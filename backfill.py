"""
backfill.py — Carga histórica (se ejecuta UNA vez, en local)
===========================================================

Recorre el último año de arXiv DÍA A DÍA, del más antiguo al más reciente, y pasa
cada día por el pipeline completo (arXiv -> JEV -> GPT -> Semantic Scholar -> Neo4j).

¿Por qué del más antiguo al más reciente? Por la "regla de oro" de las citas:
solo se crea CITA_A si el paper citado ya está en la base de datos. Como los papers
citan casi siempre a trabajos anteriores, procesar en orden cronológico maximiza
el número de citas que se pueden enlazar.

Es reanudable: guarda en `.backfill_state.json` el último día terminado. Si se corta
(Ctrl+C, caída de red...), al volver a lanzarlo continúa donde lo dejó. Además los
papers que ya están en Neo4j se saltan sin gastar llamadas a JEV ni a GPT.

Uso:
    python backfill.py                 # último año (365 días)
    python backfill.py --days 30       # solo los últimos 30 días
    python backfill.py --reset         # ignora el progreso guardado y empieza de cero
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from kg_core import Estadisticas, Pipeline, ahora_utc, configurar_logs

log = logging.getLogger("kg.backfill")
FICHERO_ESTADO = Path(__file__).with_name(".backfill_state.json")


def leer_ultimo_dia_completado() -> date | None:
    """
    Lee el progreso guardado.

    Entradas:  ninguna (lee FICHERO_ESTADO).
    Salida:    la fecha del último día procesado por completo, o None si no hay progreso.
    """
    if not FICHERO_ESTADO.exists():
        return None
    return date.fromisoformat(json.loads(FICHERO_ESTADO.read_text())["ultimo_dia"])


def guardar_ultimo_dia_completado(dia: date) -> None:
    """
    Guarda el progreso para poder reanudar.

    Entradas:  dia -> último día procesado por completo.
    Salida:    ninguna (escribe FICHERO_ESTADO).
    """
    FICHERO_ESTADO.write_text(json.dumps({"ultimo_dia": dia.isoformat()}))


def dias_a_procesar(total_dias: int, reanudar_desde: date | None) -> list[date]:
    """
    Calcula la lista de días pendientes.

    Entradas:
        total_dias     -> cuántos días hacia atrás cubrir (365 = un año).
        reanudar_desde -> último día ya completado (o None si se empieza de cero).
    Salida:
        lista de fechas, de la más antigua a la más reciente, terminando AYER
        (el día de hoy aún no está completo y lo cubrirá el cronjob diario).
    """
    ayer = ahora_utc().date() - timedelta(days=1)
    inicio = ayer - timedelta(days=total_dias - 1)
    if reanudar_desde and reanudar_desde >= inicio:
        inicio = reanudar_desde + timedelta(days=1)
    return [inicio + timedelta(days=i) for i in range((ayer - inicio).days + 1)]


def main() -> None:
    """Punto de entrada: lee los argumentos y procesa los días pendientes uno a uno."""
    parser = argparse.ArgumentParser(description="Carga histórica del grafo de conocimiento.")
    parser.add_argument("--days", type=int, default=365, help="Días hacia atrás (por defecto 365).")
    parser.add_argument("--reset", action="store_true", help="Ignora el progreso guardado.")
    args = parser.parse_args()

    configurar_logs()
    if args.reset and FICHERO_ESTADO.exists():
        FICHERO_ESTADO.unlink()

    dias = dias_a_procesar(args.days, None if args.reset else leer_ultimo_dia_completado())
    if not dias:
        log.info("No hay días pendientes. El histórico ya está cargado.")
        return
    log.info("Días a procesar: %d (de %s a %s)", len(dias), dias[0], dias[-1])

    pipeline = Pipeline()
    total = Estadisticas()
    try:
        for n, dia in enumerate(dias, start=1):
            # Intervalo de un día completo en UTC: 00:00 -> 23:59.
            desde = datetime.combine(dia, time(0, 0), tzinfo=timezone.utc)
            hasta = datetime.combine(dia, time(23, 59), tzinfo=timezone.utc)

            est = pipeline.procesar_intervalo(desde, hasta)
            total.sumar(est)
            guardar_ultimo_dia_completado(dia)
            log.info(
                "[%d/%d] %s | descargados=%d ya_existian=%d aprobados=%d guardados=%d "
                "fallos_llm=%d citas=%d",
                n, len(dias), dia, est.descargados, est.ya_existian, est.aprobados,
                est.guardados, est.fallidos_llm, est.citas,
            )
    except KeyboardInterrupt:
        log.warning("Interrumpido por el usuario. Vuelve a ejecutar el script para reanudar.")
    finally:
        pipeline.cerrar()
        log.info("TOTAL: %s", vars(total))


if __name__ == "__main__":
    main()
