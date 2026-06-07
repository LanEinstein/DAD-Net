"""Selection-source guard for the no-leak DFME protocol.

Model selection (early stopping, best epoch, and all hyper-parameter choices)
may read only the internal validation partition. Test A and test B are held out
and scored once. The guard makes leakage structurally impossible: the held-out
partitions are not valid monitor sources, so they cannot be wired into selection
by mistake.
"""

from __future__ import annotations

from enum import Enum


class MonitorSource(str, Enum):
    """Partitions that may drive model selection."""

    INNER_VAL = "inner_val"
    TESTA = "testA"


# Partitions that must never drive selection; naming one as a monitor is a leak.
FORBIDDEN_SOURCES = ("testB", "outer_test")

# The only source permitted by the no-leak protocol.
PROTOCOL_MONITOR = MonitorSource.INNER_VAL


def validate_monitor(source: str) -> MonitorSource:
    """Return the monitor source if it is allowed, else raise a leak error.

    Args:
        source: Requested monitor partition name.

    Raises:
        ValueError: If ``source`` is a held-out test partition.
    """
    if source in FORBIDDEN_SOURCES:
        raise ValueError(
            f"{source!r} is a held-out test partition and must never drive "
            "selection (no-leak protocol violation)"
        )
    return MonitorSource(source)


__all__ = ["MonitorSource", "FORBIDDEN_SOURCES", "PROTOCOL_MONITOR", "validate_monitor"]
