"""
app.py — Frontend público en Streamlit Cloud
============================================

Explorador del grafo de conocimiento. Para no saturar el navegador el grafo NUNCA
se dibuja entero:

    1. El usuario escribe un término en el buscador (índice full-text de Neo4j).
    2. Elige uno de los resultados (un Paper, un Concepto o un Autor).
    3. Se carga SOLO ese nodo y sus vecinos a profundidad 1 o 2, con un tope de nodos.

Ejecución local:   streamlit run app.py
Credenciales:      .streamlit/secrets.toml (local) o "Secrets" de Streamlit Cloud.
"""

from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase, RoutingControl
from pyvis.network import Network

# -----------------------------------------------------------------------------
# Configuración visual
# -----------------------------------------------------------------------------
st.set_page_config(page_title="arXiv Knowledge Graph", page_icon="🕸️", layout="wide")

# Color y forma de cada tipo de nodo en el dibujo.
ESTILO_NODO = {
    "Paper": {"color": "#4C78A8", "shape": "dot", "size": 18},
    "Concepto": {"color": "#F58518", "shape": "diamond", "size": 16},
    "Autor": {"color": "#54A24B", "shape": "triangle", "size": 12},
}
MAX_NODOS_POR_DEFECTO = 150


# -----------------------------------------------------------------------------
# Conexión a Neo4j
# -----------------------------------------------------------------------------
def leer_secreto(nombre: str, defecto: str | None = None) -> str | None:
    """
    Lee una credencial primero de los Secrets de Streamlit y, si no está, del entorno.

    Entradas:  nombre -> clave del secreto (p. ej. "NEO4J_URI"); defecto -> valor de reserva.
    Salida:    el valor encontrado o `defecto`.
    """
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except FileNotFoundError:  # no existe secrets.toml en local
        pass
    return os.getenv(nombre, defecto)


@st.cache_resource(show_spinner=False)
def obtener_driver():
    """
    Crea UNA sola conexión a Neo4j y la reutiliza entre usuarios y recargas.

    Entradas:  ninguna (lee NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).
    Salida:    objeto `neo4j.Driver`.
    """
    driver = GraphDatabase.driver(
        leer_secreto("NEO4J_URI"),
        auth=(leer_secreto("NEO4J_USERNAME", "neo4j"), leer_secreto("NEO4J_PASSWORD")),
    )
    driver.verify_connectivity()
    return driver


def consultar(cypher: str, **params) -> list:
    """
    Ejecuta una consulta de SOLO LECTURA (la app nunca escribe en la base de datos).

    Entradas:  cypher -> consulta Cypher; params -> parámetros con nombre.
    Salida:    lista de registros de Neo4j.
    """
    registros, _, _ = obtener_driver().execute_query(
        cypher, params, routing_=RoutingControl.READ,
        database_=leer_secreto("NEO4J_DATABASE", "neo4j"),
    )
    return registros


# -----------------------------------------------------------------------------
# Búsqueda
# -----------------------------------------------------------------------------
def a_consulta_lucene(texto: str) -> str:
    """
    Convierte lo que escribe el usuario en una consulta segura para el índice full-text.
    Escapa los caracteres especiales de Lucene y añade "*" para buscar por prefijo.

    Entradas:  texto -> término de búsqueda libre (p. ej. "graph rag").
    Salida:    consulta Lucene (p. ej. "graph* AND rag*"), o "" si no hay palabras útiles.
    """
    palabras = re.findall(r"\w+", texto.lower())
    return " AND ".join(f"{p}*" for p in palabras[:6])


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_nodos(texto: str, limite: int = 25) -> pd.DataFrame:
    """
    Busca nodos cuyo título, nombre o alias coincida con el texto.

    Entradas:  texto -> término de búsqueda; limite -> nº máximo de resultados.
    Salida:    DataFrame con columnas id, tipo, nombre, detalle y score, ordenado por relevancia.
    """
    consulta = a_consulta_lucene(texto)
    if not consulta:
        return pd.DataFrame()
    registros = consultar(
        """
        CALL db.index.fulltext.queryNodes('busqueda_global', $q) YIELD node, score
        RETURN elementId(node) AS id, labels(node)[0] AS tipo,
               coalesce(node.titulo, node.nombre) AS nombre,
               coalesce(toString(node.fecha), node.tipo, '') AS detalle, score
        ORDER BY score DESC LIMIT $limite
        """,
        q=consulta, limite=limite,
    )
    return pd.DataFrame([r.data() for r in registros])


# -----------------------------------------------------------------------------
# Vecindario (renderizado parcial)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_vecindario(id_nodo: str, profundidad: int, max_nodos: int) -> tuple[dict, list]:
    """
    Descarga SOLO el nodo elegido y sus vecinos hasta la profundidad pedida.

    Entradas:
        id_nodo     -> elementId del nodo central en Neo4j.
        profundidad -> 1 (vecinos directos) o 2 (vecinos de los vecinos).
        max_nodos   -> tope de nodos a dibujar, para no colgar el navegador.
    Salida:
        (nodos, aristas) donde
          nodos   = {elementId: {"tipo", "nombre", "props"}}
          aristas = [(id_origen, id_destino, tipo_relacion)]
    """
    profundidad = 2 if profundidad >= 2 else 1  # validado: se inserta en el texto Cypher
    registros = consultar(
        f"""
        MATCH (centro) WHERE elementId(centro) = $id
        OPTIONAL MATCH camino = (centro)-[*1..{profundidad}]-(vecino)
        RETURN centro, camino
        LIMIT $limite
        """,
        id=id_nodo, limite=max_nodos * 3,
    )
    nodos: dict[str, dict] = {}
    aristas: set[tuple[str, str, str]] = set()

    def anadir_nodo(n) -> bool:
        """Añade un nodo si cabe. Entrada: nodo de Neo4j. Salida: True si está en el dibujo."""
        if n.element_id in nodos:
            return True
        if len(nodos) >= max_nodos:
            return False
        # Los tipos propios de Neo4j (p. ej. fechas) se pasan a texto para poder cachearlos.
        props = {k: v if v is None or isinstance(v, (str, int, float, bool, list)) else str(v)
                 for k, v in dict(n).items()}
        nodos[n.element_id] = {
            "tipo": next(iter(n.labels), "Nodo"),
            "nombre": props.get("titulo") or props.get("nombre") or "?",
            "props": props,
        }
        return True

    for r in registros:
        anadir_nodo(r["centro"])
        camino = r["camino"]
        if camino is None:
            continue
        for rel in camino.relationships:
            if anadir_nodo(rel.start_node) and anadir_nodo(rel.end_node):
                aristas.add((rel.start_node.element_id, rel.end_node.element_id, rel.type))
    return nodos, sorted(aristas)


def dibujar_grafo(nodos: dict, aristas: list, id_centro: str) -> str:
    """
    Genera el HTML interactivo del grafo con pyvis.

    Entradas:
        nodos, aristas -> salida de `cargar_vecindario`.
        id_centro      -> elementId del nodo buscado (se resalta más grande).
    Salida:
        cadena HTML lista para incrustar en Streamlit.
    """
    red = Network(height="680px", width="100%", directed=True, bgcolor="#0E1117",
                  font_color="#FAFAFA", cdn_resources="remote")
    red.barnes_hut(gravity=-6000, spring_length=140)
    for nid, n in nodos.items():
        estilo = dict(ESTILO_NODO.get(n["tipo"], {"color": "#999", "shape": "dot", "size": 10}))
        if nid == id_centro:
            estilo["size"] = estilo["size"] * 2
        etiqueta = n["nombre"] if len(n["nombre"]) <= 40 else n["nombre"][:37] + "..."
        red.add_node(nid, label=etiqueta, title=f'{n["tipo"]}: {n["nombre"]}', **estilo)
    for origen, destino, tipo in aristas:
        red.add_edge(origen, destino, title=tipo, label=tipo if tipo != "ESCRIBIO" else "",
                     font={"size": 9, "color": "#BBBBBB", "strokeWidth": 0})
    return red.generate_html(notebook=False)


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
st.title("🕸️ arXiv Knowledge Graph")
st.caption(
    "Agentes de IA, RAG, grafos de conocimiento y dinámica no lineal. "
    "Papers cribados con JEV, tripletas extraídas con GPT y citas de Semantic Scholar."
)

with st.sidebar:
    st.header("Opciones")
    profundidad = st.radio("Profundidad", [1, 2], horizontal=True,
                           help="1 = vecinos directos. 2 = también los vecinos de los vecinos.")
    max_nodos = st.slider("Máximo de nodos", 20, 400, MAX_NODOS_POR_DEFECTO, step=10)
    st.markdown("**Leyenda**  \n🔵 Paper · 🔶 Concepto · 🟢 Autor")
    try:
        total = consultar(
            "MATCH (p:Paper) WITH count(p) AS papers "
            "MATCH (c:Concepto) RETURN papers, count(c) AS conceptos"
        )[0]
        st.metric("Papers", f'{total["papers"]:,}')
        st.metric("Conceptos", f'{total["conceptos"]:,}')
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo conectar con Neo4j: {exc}")
        st.stop()

texto = st.text_input("Busca un concepto, paper o autor",
                      placeholder="p. ej. retrieval augmented generation, reservoir computing, Lorenz")
if not texto:
    st.info("Escribe un término para empezar. El grafo solo se carga alrededor de lo que busques.")
    st.stop()

resultados = buscar_nodos(texto)
if resultados.empty:
    st.warning("Sin resultados. Prueba con otro término o en inglés.")
    st.stop()

opciones = {
    f'[{fila.tipo}] {fila.nombre}' + (f' · {fila.detalle}' if fila.detalle else ''): fila.id
    for fila in resultados.itertuples()
}
eleccion = st.selectbox("Resultados", list(opciones))
id_centro = opciones[eleccion]

with st.spinner("Cargando vecindario..."):
    nodos, aristas = cargar_vecindario(id_centro, profundidad, max_nodos)

st.caption(f"{len(nodos)} nodos · {len(aristas)} relaciones"
           + (" · se alcanzó el tope de nodos" if len(nodos) >= max_nodos else ""))
components.html(dibujar_grafo(nodos, aristas, id_centro), height=700)

# Tabla con los papers del vecindario, con enlace a arXiv.
papers = [n["props"] for n in nodos.values() if n["tipo"] == "Paper"]
if papers:
    st.subheader("Papers en este vecindario")
    tabla = pd.DataFrame(
        [{"Fecha": str(p.get("fecha", "")), "Título": p.get("titulo"),
          "Temas": ", ".join(p.get("temas", [])), "JEV": round(p.get("puntuacion_jev", 0), 3),
          "arXiv": p.get("url")} for p in papers]
    ).sort_values("Fecha", ascending=False)
    st.dataframe(tabla, hide_index=True, width="stretch",
                 column_config={"arXiv": st.column_config.LinkColumn("arXiv")})
