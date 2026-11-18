"""
daily_job.py — Actualización diaria (la ejecuta GitHub Actions cada noche)
=========================================================================

Procesa los papers enviados a arXiv en las últimas 24 horas... y un poco más.

¿Por qué "un poco más"? arXiv no publica al instante: un paper enviado un viernes
puede no aparecer en la API hasta el lunes. Por eso miramos hacia atrás
LOOKBACK_DAYS días (3 por defecto). No hay riesgo de duplicar ni de pagar dos veces:
el pipeline salta todos los papers que ya están en Neo4j antes de llamar a JEV o a GPT.

Uso:
    python daily_job.py
    LOOKBACK_DAYS=5 python daily_job.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

from kg_core import Pipeline, ahora_utc, configurar_logs

log = logging.getLogger("kg.daily")


def main() -> int:
    """
    Punto de entrada del cronjob.

    Entradas:  ninguna (lee LOOKBACK_DAYS y el resto de la configuración del entorno).
    Salida:    código de salida del proceso: 0 si todo fue bien, 1 si hubo un error
               (así GitHub Actions marca la ejecución en rojo y te avisa por email).
    """
    configurar_logs()
    dias_atras = int(os.getenv("LOOKBACK_DAYS") or "3")  # "or" cubre también el valor vacío
    hasta = ahora_utc()
    desde = hasta - timedelta(days=dias_atras)
    log.info("Procesando papers enviados entre %s y %s (UTC)", f"{desde:%Y-%m-%d %H:%M}", f"{hasta:%Y-%m-%d %H:%M}")

    pipeline = Pipeline()
    try:
        est = pipeline.procesar_intervalo(desde, hasta)
    except Exception:  # noqa: BLE001 - queremos el traceback completo en el log de Actions
        log.exception("El job diario ha fallado")
        return 1
    finally:
        pipeline.cerrar()

    log.info("Resultado: %s", vars(est))
    # Si TODOS los aprobados fallaron en GPT, algo va mal (clave caducada, sin saldo...).
    if est.aprobados and est.fallidos_llm == est.aprobados:
        log.error("Todas las extracciones con el LLM fallaron. Revisa OPENAI_API_KEY y el saldo.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
