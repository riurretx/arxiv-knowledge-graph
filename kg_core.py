"""
kg_core.py
==========

Módulo compartido por `backfill.py` (carga histórica en local) y `daily_job.py`
(cronjob diario en GitHub Actions). Contiene TODA la lógica del pipeline para que
ambos scripts se comporten exactamente igual:

    arXiv  ->  JEV (cribado)  ->  LLM (tripletas)  ->  Semantic Scholar (citas)  ->  Neo4j

Cada etapa vive en su propia clase para poder probarla o sustituirla por separado.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import arxiv
import openai
import requests
from neo4j import GraphDatabase
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Carga un fichero .env si existe (solo útil en local; en GitHub Actions las
# variables llegan ya como "secrets" y este paso no hace nada).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv es opcional
    pass

log = logging.getLogger("kg")


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# Categorías de arXiv que vamos a vigilar.
CATEGORIAS_ARXIV = ["cs.AI", "cs.LG", "cs.CL", "nlin.CD", "nlin.AO", "physics.comp-ph"]

# Preguntas que se le hacen a JEV. Cada clave es un "tema"; JEV devuelve, para
# cada una, la probabilidad (0-1) de que la respuesta sea "sí".
PREGUNTAS_JEV = {
    "agentes_ia": "Is the main contribution of this paper about designing, building or "
    "evaluating autonomous AI agents or multi-agent LLM systems?",
    "rag": "Is the main contribution of this paper about retrieval-augmented generation (RAG), "
    "i.e. combining retrieval over documents with a generative language model?",
    "grafos_conocimiento": "Is the main contribution of this paper about knowledge graphs "
    "(building, completing, querying or reasoning over them)?",
    "dinamica_no_lineal": "Is the main contribution of this paper a physical or data-driven model "
    "of nonlinear dynamics (chaos, complex systems, nonlinear oscillators, "
    "physics-informed or Hamiltonian neural networks)?",
}


def _env(nombre: str, defecto: str | None = None, obligatorio: bool = False) -> str | None:
    """
    Lee una variable de entorno.

    Entradas:
        nombre      -> nombre de la variable (p. ej. "NEO4J_URI").
        defecto     -> valor a devolver si la variable no existe o está vacía.
        obligatorio -> si es True y la variable no existe, se lanza un error claro.
    Salida:
        El valor de la variable como texto (o `defecto`).
    """
    valor = os.getenv(nombre, "").strip() or defecto
    if obligatorio and not valor:
        raise RuntimeError(f"Falta la variable de entorno obligatoria: {nombre}")
    return valor


@dataclass
class Config:
    """
    Agrupa toda la configuración del pipeline leída desde variables de entorno.

    Atributos:
        neo4j_uri / neo4j_user / neo4j_password / neo4j_database
            -> datos de conexión a la instancia de Neo4j AuraDB.
        jev_api_key      -> clave de la API de JEV (TypeSafe AI).
        jev_api_url      -> endpoint de JEV (TypeSafe directo o el de AIML API).
        jev_model        -> nombre del modelo JEV; solo hace falta con AIML API.
        jev_umbral       -> probabilidad mínima para aprobar un paper (0.90 = 90 %).
        openai_api_key   -> clave de la API de OpenAI.
        llm_model        -> modelo de GPT usado para extraer tripletas.
        llm_reasoning    -> esfuerzo de razonamiento ("low", "minimal"...; vacío = no se envía).
        s2_api_key       -> clave de Semantic Scholar (opcional pero muy recomendable).
        jev_workers / llm_workers -> nº de peticiones en paralelo a cada API.

    Funciones:
        desde_entorno() -> construye y devuelve un objeto Config a partir del entorno.
    """

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    jev_api_key: str
    jev_api_url: str
    jev_model: str | None
    jev_umbral: float
    openai_api_key: str
    llm_model: str
    llm_reasoning: str | None
    s2_api_key: str | None
    jev_workers: int = 8
    llm_workers: int = 4

    @classmethod
    def desde_entorno(cls) -> "Config":
        """
        Entradas: ninguna (lee el entorno del proceso).
        Salida:   un objeto Config listo para usar. Lanza RuntimeError si falta algo obligatorio.
        """
        return cls(
            neo4j_uri=_env("NEO4J_URI", obligatorio=True),
            neo4j_user=_env("NEO4J_USERNAME", "neo4j"),
            neo4j_password=_env("NEO4J_PASSWORD", obligatorio=True),
            neo4j_database=_env("NEO4J_DATABASE", "neo4j"),
            jev_api_key=_env("JEV_API_KEY", obligatorio=True),
            jev_api_url=_env("JEV_API_URL", "https://api.typesafe.ai/v1/systemone"),
            jev_model=_env("JEV_MODEL"),
            jev_umbral=float(_env("JEV_THRESHOLD", "0.90")),
            openai_api_key=_env("OPENAI_API_KEY", obligatorio=True),
            llm_model=_env("LLM_MODEL", "gpt-6-luna"),
            llm_reasoning=_env("LLM_REASONING_EFFORT", "low"),
            s2_api_key=_env("SEMANTIC_SCHOLAR_API_KEY"),
            jev_workers=int(_env("JEV_WORKERS", "8")),
            llm_workers=int(_env("LLM_WORKERS", "4")),
        )


# =============================================================================
# 2. MODELO DE DATOS INTERNO
# =============================================================================


def normalizar_arxiv_id(identificador: str) -> str:
    """
    Quita el sufijo de versión de un ID de arXiv para que "2401.01234v3" y
    "2401.01234" se consideren el mismo paper.

    Entradas:  identificador -> ID de arXiv con o sin versión.
    Salida:    el ID sin versión (p. ej. "2401.01234").
    """
    return re.sub(r"v\d+$", "", identificador.strip())


def clave_concepto(nombre: str) -> str:
    """
    Genera la clave única con la que se guarda un concepto en Neo4j. Sirve para
    que "Retrieval-Augmented Generation" y "retrieval augmented  generation"
    acaben en el MISMO nodo.

    Entradas:  nombre -> nombre del concepto tal y como lo devuelve el LLM.
    Salida:    texto en minúsculas, sin acentos, sin guiones y con espacios simples.
    """
    texto = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    texto = re.sub(r"[-_/]", " ", texto.lower())
    return re.sub(r"\s+", " ", texto).strip()


@dataclass
class Paper:
    """
    Representa un artículo de arXiv a lo largo de todo el pipeline.

    Atributos:
        arxiv_id   -> ID de arXiv sin versión (clave única en Neo4j).
        titulo     -> título del artículo.
        resumen    -> abstract.
        autores    -> lista de nombres de autores.
        fecha      -> fecha de envío (UTC).
        categorias -> categorías de arXiv (p. ej. ["cs.AI", "cs.LG"]).
        url        -> enlace a la página del paper.
        puntuaciones_jev -> {tema: probabilidad} devuelto por JEV (se rellena en la etapa 2).
        extraccion -> resultado del LLM (entidades y relaciones; etapa 3).
        referencias-> IDs de arXiv a los que cita según Semantic Scholar (etapa 4).

    Funciones:
        desde_arxiv(resultado) -> crea un Paper a partir de un resultado de la librería `arxiv`.
        temas_aprobados(umbral) -> devuelve la lista de temas cuya probabilidad supera el umbral.
        puntuacion_maxima      -> (propiedad) la mayor probabilidad devuelta por JEV.
    """

    arxiv_id: str
    titulo: str
    resumen: str
    autores: list[str]
    fecha: datetime
    categorias: list[str]
    url: str
    puntuaciones_jev: dict[str, float] = field(default_factory=dict)
    extraccion: dict | None = None
    referencias: list[str] = field(default_factory=list)

    @classmethod
    def desde_arxiv(cls, r: arxiv.Result) -> "Paper":
        """
        Entradas:  r -> objeto `arxiv.Result` tal y como lo devuelve la API de arXiv.
        Salida:    un Paper con los campos básicos rellenos.
        """
        return cls(
            arxiv_id=normalizar_arxiv_id(r.get_short_id()),
            titulo=" ".join(r.title.split()),
            resumen=" ".join(r.summary.split()),
            autores=[a.name for a in (r.authors or [])],
            fecha=r.published,
            categorias=list(r.categories or []),
            url=r.entry_id,
        )

    def temas_aprobados(self, umbral: float) -> list[str]:
        """
        Entradas:  umbral -> probabilidad mínima (0-1).
        Salida:    lista de temas (claves de PREGUNTAS_JEV) que superan el umbral.
        """
        return [t for t, p in self.puntuaciones_jev.items() if p >= umbral]

    @property
    def puntuacion_maxima(self) -> float:
        """Salida: la mayor probabilidad de JEV entre todos los temas (0 si no hay)."""
        return max(self.puntuaciones_jev.values(), default=0.0)


# =============================================================================
# 3. ETAPA 1 — INGESTA DESDE arXiv
# =============================================================================


class ArxivFetcher:
    """
    Descarga papers de arXiv para un rango de fechas.

    Atributos:
        cliente    -> cliente de la librería `arxiv`, configurado con 3 s de pausa
                      entre páginas (norma de uso de la API de arXiv).
        categorias -> categorías a consultar.

    Funciones:
        buscar(desde, hasta) -> devuelve la lista de Papers enviados entre `desde` y
                                `hasta`, sin duplicados y ordenados del más antiguo
                                al más reciente.
    """

    def __init__(self, categorias: list[str] = CATEGORIAS_ARXIV, pausa_segundos: float = 3.0):
        """
        Entradas:
            categorias     -> categorías de arXiv a vigilar.
            pausa_segundos -> espera entre peticiones a arXiv (mínimo recomendado: 3 s).
        """
        self.categorias = categorias
        self.cliente = arxiv.Client(page_size=200, delay_seconds=pausa_segundos, num_retries=5)

    def buscar(self, desde: datetime, hasta: datetime) -> list[Paper]:
        """
        Entradas:
            desde -> inicio del intervalo (UTC, incluido).
            hasta -> fin del intervalo (UTC, incluido).
        Salida:
            lista de Paper ordenada por fecha ascendente. Un paper publicado en varias
            categorías (cross-list) aparece una sola vez.
        """
        filtro_cat = " OR ".join(f"cat:{c}" for c in self.categorias)
        rango = f"submittedDate:[{desde:%Y%m%d%H%M} TO {hasta:%Y%m%d%H%M}]"
        busqueda = arxiv.Search(
            query=f"({filtro_cat}) AND {rango}",
            max_results=None,  # sin límite: la librería pagina sola respetando la pausa
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending,
        )
        vistos: dict[str, Paper] = {}
        try:
            for resultado in self.cliente.results(busqueda):
                paper = Paper.desde_arxiv(resultado)
                vistos.setdefault(paper.arxiv_id, paper)
        except arxiv.UnexpectedEmptyPageError:
            # arXiv a veces devuelve una página vacía al final de la paginación;
            # nos quedamos con lo descargado hasta ese momento.
            log.warning("arXiv devolvió una página vacía; se usa lo descargado hasta ahora.")
        return sorted(vistos.values(), key=lambda p: p.fecha)


# =============================================================================
# 4. ETAPA 2 — CRIBADO CON JEV (modelo discriminativo)
# =============================================================================


def _es_error_reintentable(exc: BaseException) -> bool:
    """
    Decide si merece la pena reintentar una petición HTTP fallida.

    Entradas:  exc -> la excepción lanzada por `requests`.
    Salida:    True para errores de red, 429 (límite de peticiones) y 5xx; False en el resto.
    """
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


_reintentar_http = retry(
    retry=retry_if_exception(_es_error_reintentable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


class JevClassifier:
    """
    Cliente de la API de JEV (TypeSafe AI). Hace una pregunta de tipo `noul`
    (sí/no, devuelve la probabilidad de "sí") por cada tema de PREGUNTAS_JEV.

    Atributos:
        url, api_key, modelo -> datos de conexión a la API.
        umbral               -> probabilidad mínima para aprobar (0.90).
        sesion               -> sesión HTTP reutilizable.

    Funciones:
        puntuar(paper) -> rellena `paper.puntuaciones_jev` y devuelve el propio paper.
        aprueba(paper) -> True si algún tema supera el umbral.
    """

    def __init__(self, cfg: Config):
        """Entradas: cfg -> configuración global (usa jev_api_url, jev_api_key, jev_model, jev_umbral)."""
        self.url = cfg.jev_api_url
        self.api_key = cfg.jev_api_key
        self.modelo = cfg.jev_model
        self.umbral = cfg.jev_umbral
        self.sesion = requests.Session()
        self.sesion.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    @_reintentar_http
    def _llamar(self, cuerpo: dict) -> dict:
        """
        Entradas:  cuerpo -> JSON de la petición.
        Salida:    JSON de la respuesta ya decodificado. Reintenta en errores temporales.
        """
        resp = self.sesion.post(self.url, json=cuerpo, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def puntuar(self, paper: Paper) -> Paper:
        """
        Entradas:  paper -> Paper con título y abstract.
        Salida:    el mismo Paper con `puntuaciones_jev` = {tema: probabilidad}.
                   Si JEV falla tras los reintentos, las puntuaciones quedan a 0 (no se aprueba).
        """
        cuerpo = {
            # "state" es el texto que JEV debe analizar.
            "state": f"Title: {paper.titulo}\n\nAbstract: {paper.resumen}",
            # Una pregunta "noul" por tema; JEV las responde todas en una sola llamada.
            "questions": {
                tema: {"type": "noul", "instructions": pregunta}
                for tema, pregunta in PREGUNTAS_JEV.items()
            },
        }
        if self.modelo:  # solo necesario si se usa el endpoint de AIML API
            cuerpo["model"] = self.modelo
        try:
            respuestas = self._llamar(cuerpo).get("answers", {})
            paper.puntuaciones_jev = {
                tema: float(respuestas.get(tema, {}).get("noul", 0.0)) for tema in PREGUNTAS_JEV
            }
        except Exception as exc:  # noqa: BLE001 - un paper fallido no debe parar el lote
            log.error("JEV falló para %s: %s", paper.arxiv_id, exc)
            paper.puntuaciones_jev = {tema: 0.0 for tema in PREGUNTAS_JEV}
        return paper

    def aprueba(self, paper: Paper) -> bool:
        """
        Entradas:  paper -> Paper ya puntuado.
        Salida:    True si al menos un tema alcanza el umbral (por defecto 90 %).
        """
        return paper.puntuacion_maxima >= self.umbral


# =============================================================================
# 5. ETAPA 3 — EXTRACCIÓN DE TRIPLETAS CON GPT (LLM generativo de OpenAI)
# =============================================================================

# Ontología cerrada: el LLM SOLO puede usar estos tipos de entidad y de relación.
TIPOS_ENTIDAD = [
    "Modelo",        # un modelo concreto: GPT-4, BERT, Lorenz-96 surrogate...
    "Metodo",        # técnica o algoritmo: RAG, PPO, Reservoir Computing...
    "Tarea",         # problema que se resuelve: question answering, forecasting...
    "Dataset",       # conjunto de datos o benchmark: HotpotQA, ERA5...
    "Metrica",       # medida de evaluación: F1, RMSE, Lyapunov exponent...
    "Herramienta",   # software o framework: LangChain, Neo4j, PyTorch...
    "SistemaFisico", # sistema dinámico estudiado: péndulo doble, Navier-Stokes...
    "Concepto",      # idea general que no encaja en lo anterior
]
TIPOS_RELACION = [
    "PROPONE",      # el trabajo propone/introduce X (origen suele ser el método principal)
    "USA",          # X utiliza Y
    "EXTIENDE",     # X amplía o modifica Y
    "MEJORA",       # X supera a Y en algo
    "COMPARA_CON",  # X se compara con Y
    "EVALUA_EN",    # X se evalúa en el dataset/métrica Y
    "RESUELVE",     # X aborda la tarea Y
    "MODELA",       # X modela el sistema físico Y
    "PARTE_DE",     # X es un componente de Y
]

PROMPT_SISTEMA = f"""You build a scientific knowledge graph from arXiv abstracts.

Extract the key entities and the relations between them from the abstract you receive.

ONTOLOGY (use ONLY these values):
- Entity types: {", ".join(TIPOS_ENTIDAD)}
- Relation types: {", ".join(TIPOS_RELACION)}

RULES TO AVOID DUPLICATE CONCEPTS:
1. Use the canonical, fully expanded English name in singular form, Title Case for methods
   and concepts (e.g. "Retrieval-Augmented Generation", not "RAG" or "retrieval augmented
   generation systems"). Put acronyms and variants in "alias".
2. Use the official spelling for named models, datasets and tools (e.g. "GPT-4o", "HotpotQA",
   "PyTorch").
3. Prefer these canonical names when the concept appears: "Large Language Model",
   "Retrieval-Augmented Generation", "Knowledge Graph", "Autonomous Agent", "Multi-Agent System",
   "Graph Neural Network", "Reinforcement Learning", "Transformer",
   "Physics-Informed Neural Network", "Hamiltonian Neural Network", "Reservoir Computing",
   "Koopman Operator", "Lyapunov Exponent", "Chaotic System".
4. Do not extract generic words ("method", "approach", "framework", "results", "paper").
5. Extract between 3 and 15 entities. Every relation must connect two entities from your list,
   using exactly the same "nombre".
6. Only state relations the abstract supports. Do not invent facts.
"""

# Esquema JSON estricto que la API obliga a cumplir (structured outputs).
ESQUEMA_EXTRACCION = {
    "type": "object",
    "properties": {
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "tipo": {"type": "string", "enum": TIPOS_ENTIDAD},
                    "alias": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["nombre", "tipo", "alias"],
                "additionalProperties": False,
            },
        },
        "relaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string"},
                    "relacion": {"type": "string", "enum": TIPOS_RELACION},
                    "destino": {"type": "string"},
                },
                "required": ["origen", "relacion", "destino"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entidades", "relaciones"],
    "additionalProperties": False,
}


class TripletExtractor:
    """
    Envía el abstract a GPT (OpenAI) y obtiene entidades y relaciones (tripletas)
    en JSON estricto, validado contra ESQUEMA_EXTRACCION (Structured Outputs).

    Atributos:
        cliente    -> cliente oficial de OpenAI (reintenta solo errores 429/5xx).
        modelo     -> ID del modelo de GPT (variable LLM_MODEL).
        razonamiento -> esfuerzo de razonamiento a pedir, o None para no enviarlo.

    Funciones:
        extraer(paper) -> rellena `paper.extraccion` con
                          {"entidades": [...], "relaciones": [...]} limpios y
                          devuelve True si salió bien, False si no.
    """

    def __init__(self, cfg: Config):
        """Entradas: cfg -> configuración global (usa openai_api_key, llm_model y llm_reasoning)."""
        self.cliente = openai.OpenAI(api_key=cfg.openai_api_key, max_retries=5)
        self.modelo = cfg.llm_model
        self.razonamiento = cfg.llm_reasoning

    def extraer(self, paper: Paper) -> bool:
        """
        Entradas:  paper -> Paper aprobado por JEV.
        Salida:    True si `paper.extraccion` quedó relleno; False si el modelo rechazó la
                   petición, se cortó o hubo un error de API (el paper se reintentará
                   en una ejecución futura porque no se guardará en Neo4j).
        """
        parametros = dict(
            model=self.modelo,
            instructions=PROMPT_SISTEMA,
            input=f"Title: {paper.titulo}\n\nAbstract: {paper.resumen}",
            max_output_tokens=16000,
            store=False,  # no guardar la conversación en OpenAI
            # Structured Outputs: la respuesta DEBE cumplir el esquema JSON.
            text={
                "format": {
                    "type": "json_schema",
                    "name": "tripletas",
                    "schema": ESQUEMA_EXTRACCION,
                    "strict": True,
                }
            },
        )
        if self.razonamiento:
            # Extracción = tarea sencilla: poco razonamiento para gastar menos tokens.
            # Si usas un modelo sin razonamiento (p. ej. gpt-4o-mini), deja
            # LLM_REASONING_EFFORT vacío o la API devolverá un error 400.
            parametros["reasoning"] = {"effort": self.razonamiento}

        try:
            respuesta = self.cliente.responses.create(**parametros)
        except openai.RateLimitError as exc:
            log.error("GPT: límite de peticiones o saldo agotado para %s: %s", paper.arxiv_id, exc)
            return False
        except openai.APIStatusError as exc:
            log.error("GPT: error %s para %s: %s", exc.status_code, paper.arxiv_id, exc)
            return False
        except openai.APIConnectionError as exc:
            log.error("GPT: error de conexión para %s: %s", paper.arxiv_id, exc)
            return False

        # "incomplete" = se quedó sin tokens o lo cortó un filtro.
        if respuesta.status != "completed":
            log.warning("GPT no completó %s (status=%s, detalle=%s)",
                        paper.arxiv_id, respuesta.status, respuesta.incomplete_details)
            return False

        # Si el modelo se niega, devuelve un bloque "refusal" en lugar de texto.
        for salida in respuesta.output:
            if salida.type == "message" and any(c.type == "refusal" for c in salida.content):
                log.warning("GPT rechazó procesar %s", paper.arxiv_id)
                return False

        texto = respuesta.output_text
        if not texto:
            return False
        paper.extraccion = self._limpiar(json.loads(texto))
        return True

    @staticmethod
    def _limpiar(datos: dict) -> dict:
        """
        Normaliza la salida del LLM y descarta lo que no encaje.

        Entradas:  datos -> JSON devuelto por GPT.
        Salida:    {"entidades": [{clave, nombre, tipo, alias}],
                    "relaciones": [{origen, relacion, destino}]} donde `origen` y
                   `destino` ya son claves normalizadas y existen en `entidades`.
        """
        entidades: dict[str, dict] = {}
        for e in datos.get("entidades", []):
            nombre = e["nombre"].strip()
            clave = clave_concepto(nombre)
            if clave and clave not in entidades:
                alias = sorted({a.strip() for a in e.get("alias", []) if a.strip()})
                entidades[clave] = {"clave": clave, "nombre": nombre, "tipo": e["tipo"], "alias": alias}

        relaciones = []
        vistas = set()
        for r in datos.get("relaciones", []):
            o, d = clave_concepto(r["origen"]), clave_concepto(r["destino"])
            triple = (o, r["relacion"], d)
            if o in entidades and d in entidades and o != d and triple not in vistas:
                vistas.add(triple)
                relaciones.append({"origen": o, "relacion": r["relacion"], "destino": d})
        return {"entidades": list(entidades.values()), "relaciones": relaciones}


# =============================================================================
# 6. ETAPA 4 — CITAS CON SEMANTIC SCHOLAR
# =============================================================================


class SemanticScholarClient:
    """
    Obtiene, para cada paper, los IDs de arXiv de los trabajos que cita.

    Atributos:
        sesion -> sesión HTTP (con la cabecera x-api-key si hay clave).
        pausa  -> segundos entre llamadas (1 petición/s con clave gratuita).

    Funciones:
        rellenar_referencias(papers) -> rellena `paper.referencias` de cada paper
                                        usando el endpoint por lotes (hasta 500 IDs
                                        por llamada). No devuelve nada.
    """

    URL_LOTE = "https://api.semanticscholar.org/graph/v1/paper/batch"
    TAM_LOTE = 400

    def __init__(self, cfg: Config):
        """Entradas: cfg -> configuración global (usa s2_api_key)."""
        self.sesion = requests.Session()
        if cfg.s2_api_key:
            self.sesion.headers["x-api-key"] = cfg.s2_api_key
        # Sin clave se comparte un cupo público muy limitado: vamos más despacio.
        self.pausa = 1.1 if cfg.s2_api_key else 3.5

    @_reintentar_http
    def _pedir_lote(self, ids: list[str]) -> list[dict | None]:
        """
        Entradas:  ids -> lista de IDs de arXiv (sin versión).
        Salida:    lista alineada con `ids`; cada elemento es el JSON del paper en
                   Semantic Scholar o None si no lo conoce.
        """
        resp = self.sesion.post(
            self.URL_LOTE,
            params={"fields": "externalIds,references.externalIds"},
            json={"ids": [f"arXiv:{i}" for i in ids]},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def rellenar_referencias(self, papers: list[Paper]) -> None:
        """
        Entradas:  papers -> Papers aprobados a los que buscar citas.
        Salida:    ninguna; modifica `paper.referencias` in situ (lista de IDs de arXiv).
                   Si Semantic Scholar falla, el paper se queda sin referencias.
        """
        for i in range(0, len(papers), self.TAM_LOTE):
            lote = papers[i : i + self.TAM_LOTE]
            try:
                datos = self._pedir_lote([p.arxiv_id for p in lote])
            except Exception as exc:  # noqa: BLE001
                log.error("Semantic Scholar falló para un lote de %d papers: %s", len(lote), exc)
                datos = [None] * len(lote)
            for paper, info in zip(lote, datos):
                refs = set()
                for ref in (info or {}).get("references") or []:
                    arxiv_ref = ((ref or {}).get("externalIds") or {}).get("ArXiv")
                    if arxiv_ref:
                        refs.add(normalizar_arxiv_id(arxiv_ref))
                paper.referencias = sorted(refs)
            time.sleep(self.pausa)


# =============================================================================
# 7. ETAPA 5 — PERSISTENCIA EN NEO4J AURADB
# =============================================================================


class Neo4jRepository:
    """
    Encapsula todas las escrituras y lecturas en Neo4j.

    Modelo del grafo:
        (:Autor)-[:ESCRIBIO]->(:Paper)
        (:Paper)-[:MENCIONA]->(:Concepto)
        (:Concepto)-[:USA|MEJORA|...{papers:[...]}]->(:Concepto)
        (:Paper)-[:CITA_A]->(:Paper)

    Atributos:
        driver   -> conexión oficial de Neo4j.
        database -> nombre de la base de datos (en AuraDB Free: "neo4j").

    Funciones:
        crear_esquema()            -> crea restricciones de unicidad e índice de búsqueda.
        ids_existentes(ids)        -> devuelve el conjunto de IDs que ya están en la BD.
        guardar_paper(paper)       -> inserta un paper con autores, conceptos y tripletas.
        crear_citas(paper)         -> crea CITA_A solo hacia papers que YA existen; devuelve cuántas.
        cerrar()                   -> cierra la conexión.
    """

    def __init__(self, cfg: Config):
        """Entradas: cfg -> configuración global (usa los datos de NEO4J_*)."""
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        self.driver.verify_connectivity()
        self.database = cfg.neo4j_database

    def cerrar(self) -> None:
        """Cierra la conexión con la base de datos. No recibe ni devuelve nada."""
        self.driver.close()

    def _q(self, cypher: str, **params):
        """
        Ejecuta una consulta Cypher.

        Entradas:  cypher -> texto de la consulta; params -> parámetros con nombre.
        Salida:    lista de registros devueltos por Neo4j.
        """
        registros, _, _ = self.driver.execute_query(cypher, params, database_=self.database)
        return registros

    def crear_esquema(self) -> None:
        """Crea (si no existen) las restricciones de unicidad y el índice full-text."""
        for cypher in (
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT autor_nombre IF NOT EXISTS FOR (a:Autor) REQUIRE a.nombre IS UNIQUE",
            "CREATE CONSTRAINT concepto_clave IF NOT EXISTS FOR (c:Concepto) REQUIRE c.clave IS UNIQUE",
            # Índice que usa el buscador de la app de Streamlit.
            "CREATE FULLTEXT INDEX busqueda_global IF NOT EXISTS "
            "FOR (n:Paper|Concepto|Autor) ON EACH [n.titulo, n.nombre, n.alias_texto]",
        ):
            self._q(cypher)

    def ids_existentes(self, ids: Iterable[str]) -> set[str]:
        """
        Entradas:  ids -> IDs de arXiv a comprobar.
        Salida:    subconjunto de esos IDs que ya existen como nodo :Paper.
        """
        registros = self._q(
            "UNWIND $ids AS id MATCH (p:Paper {arxiv_id: id}) RETURN p.arxiv_id AS id", ids=list(ids)
        )
        return {r["id"] for r in registros}

    def guardar_paper(self, paper: Paper, umbral: float) -> None:
        """
        Inserta (o actualiza) un paper y todo lo que cuelga de él en UNA transacción.

        Entradas:
            paper  -> Paper con `extraccion` rellena.
            umbral -> umbral de JEV, para guardar qué temas lo aprobaron.
        Salida:   ninguna.
        """
        ext = paper.extraccion or {"entidades": [], "relaciones": []}
        for e in ext["entidades"]:
            e["alias_texto"] = " | ".join(e["alias"])  # versión texto para el índice full-text

        def _tx(tx):
            # 1) Nodo Paper
            tx.run(
                """
                MERGE (p:Paper {arxiv_id: $id})
                SET p.titulo = $titulo, p.resumen = $resumen, p.fecha = date($fecha),
                    p.categorias = $categorias, p.url = $url, p.temas = $temas,
                    p.puntuacion_jev = $puntuacion, p.actualizado = datetime()
                """,
                id=paper.arxiv_id, titulo=paper.titulo, resumen=paper.resumen,
                fecha=paper.fecha.date().isoformat(), categorias=paper.categorias, url=paper.url,
                temas=paper.temas_aprobados(umbral), puntuacion=paper.puntuacion_maxima,
            )
            # 2) Autores
            tx.run(
                """
                MATCH (p:Paper {arxiv_id: $id})
                UNWIND $autores AS nombre
                MERGE (a:Autor {nombre: nombre})
                MERGE (a)-[:ESCRIBIO]->(p)
                """,
                id=paper.arxiv_id, autores=paper.autores,
            )
            # 3) Conceptos (MERGE por clave normalizada => sin duplicados)
            tx.run(
                """
                MATCH (p:Paper {arxiv_id: $id})
                UNWIND $entidades AS e
                MERGE (c:Concepto {clave: e.clave})
                  ON CREATE SET c.nombre = e.nombre, c.tipo = e.tipo, c.alias = e.alias,
                                c.alias_texto = e.alias_texto
                  ON MATCH SET c.alias = c.alias + [x IN e.alias WHERE NOT x IN c.alias]
                SET c.alias_texto = reduce(s = '', x IN c.alias | s + ' ' + x)
                MERGE (p)-[:MENCIONA]->(c)
                """,
                id=paper.arxiv_id, entidades=ext["entidades"],
            )
            # 4) Tripletas Concepto-RELACION->Concepto. El tipo de relación no puede ir
            #    como parámetro en Cypher, pero viene de la lista cerrada TIPOS_RELACION
            #    (validada por el esquema JSON), así que es seguro insertarlo en el texto.
            por_tipo: dict[str, list[dict]] = {}
            for r in ext["relaciones"]:
                if r["relacion"] in TIPOS_RELACION:
                    por_tipo.setdefault(r["relacion"], []).append(r)
            for tipo, rels in por_tipo.items():
                tx.run(
                    f"""
                    UNWIND $rels AS r
                    MATCH (a:Concepto {{clave: r.origen}})
                    MATCH (b:Concepto {{clave: r.destino}})
                    MERGE (a)-[x:{tipo}]->(b)
                      ON CREATE SET x.papers = [$id]
                      ON MATCH SET x.papers = CASE WHEN $id IN x.papers THEN x.papers
                                                   ELSE x.papers + $id END
                    """,
                    rels=rels, id=paper.arxiv_id,
                )

        with self.driver.session(database=self.database) as sesion:
            sesion.execute_write(_tx)

    def crear_citas(self, paper: Paper) -> int:
        """
        REGLA DE ORO: solo se crea (:Paper)-[:CITA_A]->(:Paper) si el paper citado
        YA existe en la base de datos. El MATCH lo garantiza: si el nodo no está,
        la fila se descarta y no se crea nada.

        Entradas:  paper -> Paper ya guardado, con `referencias` rellenas.
        Salida:    número de relaciones CITA_A creadas o confirmadas.
        """
        if not paper.referencias:
            return 0
        registros = self._q(
            """
            MATCH (p:Paper {arxiv_id: $id})
            UNWIND $refs AS ref
            MATCH (citado:Paper {arxiv_id: ref})
            WHERE citado <> p
            MERGE (p)-[:CITA_A]->(citado)
            RETURN count(*) AS n
            """,
            id=paper.arxiv_id, refs=paper.referencias,
        )
        return registros[0]["n"] if registros else 0


# =============================================================================
# 8. ORQUESTADOR
# =============================================================================


@dataclass
class Estadisticas:
    """
    Contadores de una ejecución del pipeline (para el log final).

    Atributos:
        descargados -> papers recibidos de arXiv.
        ya_existian -> papers que ya estaban en Neo4j y se saltaron.
        aprobados   -> papers que superaron el umbral de JEV.
        guardados   -> papers insertados en Neo4j.
        fallidos_llm-> aprobados en los que falló la extracción (se reintentarán).
        citas       -> relaciones CITA_A creadas.

    Funciones:
        sumar(otra) -> acumula los contadores de otra ejecución en esta (no devuelve nada).
    """

    descargados: int = 0
    ya_existian: int = 0
    aprobados: int = 0
    guardados: int = 0
    fallidos_llm: int = 0
    citas: int = 0

    def sumar(self, otra: "Estadisticas") -> None:
        """Entradas: otra -> Estadisticas a acumular. Salida: ninguna."""
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(otra, k))


class Pipeline:
    """
    Une todas las etapas. Lo usan tanto `backfill.py` como `daily_job.py`.

    Atributos:
        cfg, arxiv, jev, llm, s2, db -> configuración y un objeto por etapa.

    Funciones:
        procesar_intervalo(desde, hasta) -> ejecuta el pipeline completo para los
                                            papers enviados en ese intervalo y
                                            devuelve un objeto Estadisticas.
        cerrar()                         -> libera la conexión a Neo4j.
    """

    def __init__(self, cfg: Config | None = None):
        """Entradas: cfg -> configuración (si es None se lee del entorno)."""
        self.cfg = cfg or Config.desde_entorno()
        self.arxiv = ArxivFetcher()
        self.jev = JevClassifier(self.cfg)
        self.llm = TripletExtractor(self.cfg)
        self.s2 = SemanticScholarClient(self.cfg)
        self.db = Neo4jRepository(self.cfg)
        self.db.crear_esquema()

    def cerrar(self) -> None:
        """Cierra la conexión con Neo4j. No recibe ni devuelve nada."""
        self.db.cerrar()

    def procesar_intervalo(self, desde: datetime, hasta: datetime) -> Estadisticas:
        """
        Entradas:
            desde, hasta -> intervalo de fechas de envío (UTC) a procesar.
        Salida:
            Estadisticas de lo ocurrido.
        """
        est = Estadisticas()

        # --- Etapa 1: arXiv ---
        papers = self.arxiv.buscar(desde, hasta)
        est.descargados = len(papers)

        # Idempotencia: no se reprocesan (ni se paga JEV/LLM por) papers ya guardados.
        existentes = self.db.ids_existentes(p.arxiv_id for p in papers)
        papers = [p for p in papers if p.arxiv_id not in existentes]
        est.ya_existian = len(existentes)
        if not papers:
            return est

        # --- Etapa 2: JEV en paralelo ---
        with ThreadPoolExecutor(max_workers=self.cfg.jev_workers) as pool:
            papers = list(pool.map(self.jev.puntuar, papers))
        aprobados = [p for p in papers if self.jev.aprueba(p)]
        est.aprobados = len(aprobados)
        if not aprobados:
            return est

        # --- Etapa 3: GPT en paralelo ---
        with ThreadPoolExecutor(max_workers=self.cfg.llm_workers) as pool:
            exito = list(pool.map(self.llm.extraer, aprobados))
        listos = [p for p, ok in zip(aprobados, exito) if ok]
        est.fallidos_llm = len(aprobados) - len(listos)

        # --- Etapa 4: Semantic Scholar (por lotes) ---
        self.s2.rellenar_referencias(listos)

        # --- Etapa 5: Neo4j ---
        # Primero se guardan los nodos en orden cronológico y DESPUÉS las citas, de
        # modo que un paper puede citar a otro del mismo lote si ya quedó guardado.
        for p in listos:
            self.db.guardar_paper(p, self.cfg.jev_umbral)
            est.guardados += 1
        for p in listos:
            est.citas += self.db.crear_citas(p)
        return est


def configurar_logs() -> None:
    """Configura el formato de los mensajes por consola. No recibe ni devuelve nada."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # La librería arxiv es muy habladora a nivel INFO.
    logging.getLogger("arxiv").setLevel(logging.WARNING)


def ahora_utc() -> datetime:
    """Salida: fecha y hora actual en UTC (con zona horaria)."""
    return datetime.now(timezone.utc)
