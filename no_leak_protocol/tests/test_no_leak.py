"""Checks that encode the no-leak protocol guarantees."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor import FORBIDDEN_SOURCES, MonitorSource, validate_monitor  # noqa: E402
from verify_no_leak import audit_split  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"
SPLIT_FILES = sorted(SPLIT_DIR.glob("dfme_*class_internalval.json"))


def test_monitor_enum_excludes_held_out_partitions():
    values = {member.value for member in MonitorSource}
    assert "testB" not in values
    assert "outer_test" not in values


@pytest.mark.parametrize("source", FORBIDDEN_SOURCES)
def test_forbidden_monitor_raises(source):
    with pytest.raises(ValueError):
        validate_monitor(source)


def test_inner_val_is_a_valid_monitor():
    assert validate_monitor("inner_val") is MonitorSource.INNER_VAL


def test_split_files_present():
    assert SPLIT_FILES, "no DFME internal-validation split files found"


@pytest.mark.parametrize("path", SPLIT_FILES, ids=lambda p: p.name)
def test_splits_have_no_leakage(path):
    assert audit_split(path) == []


@pytest.mark.parametrize("path", SPLIT_FILES, ids=lambda p: p.name)
def test_split_paths_are_relative(path):
    csv_paths = json.loads(path.read_text())["metadata"]["csv_paths"]
    for value in csv_paths.values():
        assert not os.path.isabs(value)
