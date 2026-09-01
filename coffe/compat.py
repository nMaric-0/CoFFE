"""Legacy-vocabulary compatibility layer (PAPER_CANON §1, §7.3).

The repository's names were brought to the paper's vocabulary in phase 4 of the
cleanup:

===================  ==================  ==========================================
legacy               canonical           kind
===================  ==================  ==========================================
``mft_cpea``         ``coffe``           ``model.name`` config value
``enhanced``         ``simmim``          ``pretrain.objective`` config value
``spectral``         ``simmim_band``     masking-regime (variant) id
``spatial``          ``simmim_token``    masking-regime (variant) id
``both``             ``simmim_band_token``  masking-regime (variant) id
``MFTCPEACosine``    ``CoFFE``           class
``MFTOriginalCosine``  ``MFTOriginal``   class
``HyperSIGMACosine``  ``HyperSIGMAFewShot``  class
===================  ==================  ==========================================

Two guarantees, both required by PAPER_CANON §7.3:

1. **Readers accept both.** Frozen experiment trees on Nikola's machines carry
   ``pretrain_config.yaml`` with ``model.name: "mft_cpea"`` and objective
   ``"enhanced"``, and ``results.json`` with ``model_type: "MFTCPEACosine"``.
   Every reader funnels such values through the ``normalize_*`` helpers here,
   which map them and emit one :class:`DeprecationWarning` per (value, origin).
2. **Writers emit canonical only.** The one deliberate exception is
   ``scripts/reports/aggregate_significance.py``, whose emitted ``group``/``variant``
   keys reproduce the significance experiment's own directory-name components
   (see CHANGES.md).

What this module deliberately does *not* touch: string literals that *name
on-disk artifacts* (run directories like ``houston_enhanced_spatial_run1``,
``checkpoint_dir`` paths, ``_ENHANCED_CANONICAL`` in
``scripts/reproduce/sig_significance_config.py``). Those are frozen names of files that
exist, not vocabulary — see the audit's DO-NOT-RENAME list.

Class aliases are resolved lazily through the module ``__getattr__`` so that
importing a ``normalize_*`` helper never pulls in ``torch``::

    from coffe.compat import MFTCPEACosine   # -> coffe.models.coffe.CoFFE + warning

Phase 5 moved this module from the repo root into the package; the public
names are unchanged.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "LEGACY_MODEL_NAMES",
    "LEGACY_OBJECTIVES",
    "LEGACY_VARIANTS",
    "LEGACY_MODEL_TYPES",
    "LEGACY_CLASSES",
    "normalize_model_name",
    "normalize_objective",
    "normalize_variant",
    "normalize_model_type",
    "normalize_pretrain_config",
    "reset_deprecation_state",
    # lazy class aliases (see module __getattr__)
    "MFTCPEACosine",
    "MFTOriginalCosine",
    "HyperSIGMACosine",
    "EnhancedMaskedSpectralSpatialModel",
]


#: ``model.name`` config values.
LEGACY_MODEL_NAMES: Dict[str, str] = {"mft_cpea": "coffe"}

#: ``pretrain.objective`` config values.
LEGACY_OBJECTIVES: Dict[str, str] = {"enhanced": "simmim"}

#: Masking-regime ids ("variant" in the significance experiment's vocabulary).
LEGACY_VARIANTS: Dict[str, str] = {
    "spectral": "simmim_band",
    "spatial": "simmim_token",
    "both": "simmim_band_token",
}

#: ``model_type`` values written into ``results.json`` (PAPER_CANON §8 D16).
#: ``HyperSIGMADual`` is canonical already and is intentionally absent.
LEGACY_MODEL_TYPES: Dict[str, str] = {
    "MFTCPEACosine": "CoFFE",
    "MFTOriginalCosine": "MFTOriginal",
    "HyperSIGMACosine": "HyperSIGMAFewShot",
}

#: Legacy class name -> (module, canonical class name).
LEGACY_CLASSES: Dict[str, Tuple[str, str]] = {
    "MFTCPEACosine": ("coffe.models.coffe", "CoFFE"),
    "MFTOriginalCosine": ("coffe.models.mft_original", "MFTOriginal"),
    "HyperSIGMACosine": ("coffe.models.hypersigma.few_shot", "HyperSIGMAFewShot"),
    "EnhancedMaskedSpectralSpatialModel": ("coffe.pretrain.simmim", "SimMIMPretrainModel"),
}


_warned: set = set()


def reset_deprecation_state() -> None:
    """Forget which deprecations have already been reported (tests use this)."""
    _warned.clear()


def _warn_once(kind: str, legacy: str, canonical: str, origin: Optional[str]) -> None:
    key = (kind, legacy, origin)
    if key in _warned:
        return
    _warned.add(key)
    where = f" (read from {origin})" if origin else ""
    warnings.warn(
        f"legacy {kind} {legacy!r}{where} is deprecated; it maps to {canonical!r}. "
        f"Frozen artifacts keep working, but new configs and code should use "
        f"{canonical!r} (PAPER_CANON §1).",
        DeprecationWarning,
        stacklevel=4,
    )


def _normalize(mapping: Mapping[str, str], kind: str, value: Any, origin: Optional[str]) -> Any:
    """Map one legacy value; pass everything else (canonical, unknown, non-str) through.

    Unknown values are *not* rejected here: validation stays with the caller, so
    that this layer can never change which inputs a script accepts.
    """
    if not isinstance(value, str):
        return value
    canonical = mapping.get(value)
    if canonical is None:
        return value
    _warn_once(kind, value, canonical, origin)
    return canonical


def normalize_model_name(value: Any, *, origin: Optional[str] = None) -> Any:
    """``"mft_cpea"`` -> ``"coffe"``. ``origin`` names where the value came from."""
    return _normalize(LEGACY_MODEL_NAMES, "model.name", value, origin)


def normalize_objective(value: Any, *, origin: Optional[str] = None) -> Any:
    """``"enhanced"`` -> ``"simmim"``."""
    return _normalize(LEGACY_OBJECTIVES, "pretrain.objective", value, origin)


def normalize_variant(value: Any, *, origin: Optional[str] = None) -> Any:
    """``"spectral"/"spatial"/"both"`` -> ``"simmim_band"/"simmim_token"/"simmim_band_token"``.

    Only for values used as a *regime label*. Variant ids that are components of
    on-disk directory names are DO-NOT-RENAME and must not pass through here.
    """
    return _normalize(LEGACY_VARIANTS, "masking regime", value, origin)


def normalize_model_type(value: Any, *, origin: Optional[str] = None) -> Any:
    """``"MFTCPEACosine"`` -> ``"CoFFE"`` etc., for ``results.json`` readers."""
    return _normalize(LEGACY_MODEL_TYPES, "model_type", value, origin)


def normalize_pretrain_config(
    config: Mapping[str, Any], *, origin: Optional[str] = None
) -> Dict[str, Any]:
    """Return a copy of a pretrain config with its vocabulary canonicalised.

    Normalises ``model.name`` and ``pretrain.objective`` only — the two keys a
    frozen ``pretrain_config.yaml`` can carry in legacy vocabulary. Values are
    otherwise untouched (mask ratios, paths, schedules), and the input mapping
    is never mutated.
    """
    out = dict(config)
    model_cfg = out.get("model")
    if isinstance(model_cfg, Mapping) and "name" in model_cfg:
        model_cfg = dict(model_cfg)
        model_cfg["name"] = normalize_model_name(model_cfg["name"], origin=origin)
        out["model"] = model_cfg
    pretrain_cfg = out.get("pretrain")
    if isinstance(pretrain_cfg, Mapping) and "objective" in pretrain_cfg:
        pretrain_cfg = dict(pretrain_cfg)
        pretrain_cfg["objective"] = normalize_objective(pretrain_cfg["objective"], origin=origin)
        out["pretrain"] = pretrain_cfg
    return out


def __getattr__(name: str) -> Any:
    """Resolve a legacy class alias on first access, with a deprecation warning."""
    target = LEGACY_CLASSES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, canonical = target
    _warn_once("class", name, canonical, f"{module_name}.{canonical}")
    import importlib

    return getattr(importlib.import_module(module_name), canonical)


def __dir__() -> Iterable[str]:
    return sorted(set(__all__) | set(globals()))
