"""Unit tests for ``hi_ontop.config`` (single hyperparameter source-of-truth)."""

from __future__ import annotations

import json
from pathlib import Path

from hi_ontop.config import get_section, load_config


def test_load_config_no_file_returns_module_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg["segmenter"] == {"alpha": 1.0, "lmda": 10.0, "sigma0_sq": 0.01}


def test_load_config_with_file_overrides_section(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "segmenter": {"alpha": 5.0, "lmda": 2.0},   # partial: sigma0_sq omitted
    }))
    cfg = load_config(p)
    assert cfg["segmenter"]["alpha"] == 5.0
    assert cfg["segmenter"]["lmda"] == 2.0
    assert cfg["segmenter"]["sigma0_sq"] == 0.01     # default kept


def test_load_config_adds_unknown_section(tmp_path: Path) -> None:
    """A section not in defaults is passed through verbatim."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"extra": {"k": 7}}))
    cfg = load_config(p)
    assert cfg["extra"] == {"k": 7}
    assert cfg["segmenter"]["alpha"] == 1.0          # defaults still present


def test_load_config_strips_underscore_keys(tmp_path: Path) -> None:
    """Documented JSON: keys starting with '_' are comments."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "_comment": "top-level comment",
        "segmenter": {"_doc": "section comment", "alpha": 2.0},
    }))
    cfg = load_config(p)
    assert "_comment" not in cfg
    assert "_doc" not in cfg["segmenter"]
    assert cfg["segmenter"]["alpha"] == 2.0


def test_get_section_shortcut(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"segmenter": {"alpha": 9.0}}))
    sec = get_section("segmenter", p)
    assert sec["alpha"] == 9.0
    # other segmenter defaults preserved
    assert sec["sigma0_sq"] == 0.01


def test_get_section_unknown_returns_empty() -> None:
    assert get_section("does_not_exist") == {}


def test_repo_default_config_loadable() -> None:
    """The shipped configs/hiontop.json must parse and expose the segmenter section."""
    cfg = load_config()
    assert "segmenter" in cfg
    for key in ("alpha", "lmda", "sigma0_sq"):
        assert key in cfg["segmenter"], f"missing segmenter key: {key}"
