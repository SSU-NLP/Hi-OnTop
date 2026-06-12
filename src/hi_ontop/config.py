"""HiOnTop 알고리즘 hyperparameter config loader.

Single source of truth: ``configs/hiontop.json``. 자격증명/환경/model은
``.env`` (python-dotenv)이 담당.

Precedence (낮음 → 높음): module defaults < ``configs/hiontop.json`` < CLI 인자.

Sections:
    segmenter         : sCRP + Gaussian (alpha, lmda, sigma0_sq)

JSON에서 ``_``로 시작하는 키는 주석 — loader가 자동 strip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "hiontop.json"


_DEFAULTS: dict[str, dict[str, Any]] = {
    "segmenter": {"alpha": 1.0, "lmda": 10.0, "sigma0_sq": 0.01},
}


def _strip_comments(obj: Any) -> Any:
    """Recursively drop keys starting with '_' (documented-JSON sentinel)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def load_config(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Return merged config: defaults overridden per-section by file values.

    File-missing or section-missing 모두 안전 — 누락된 항목은 default 사용.
    """
    out: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _DEFAULTS.items()}
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        raw = _strip_comments(json.loads(p.read_text()))
        for section, values in raw.items():
            if section in out and isinstance(values, dict):
                out[section].update(values)
            elif isinstance(values, dict):
                out[section] = dict(values)
    return out


def get_section(name: str, path: Path | str | None = None) -> dict[str, Any]:
    """Convenience accessor for a single section dict."""
    return load_config(path).get(name, {})
