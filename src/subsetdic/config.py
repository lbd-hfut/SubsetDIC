"""YAML configuration parser for SubsetDIC."""

import yaml
from pathlib import Path
from typing import Optional, Union


DEFAULT_CONFIG = {
    "dic": {
        "radius": 20,
        "spacing": 1,
        "cutoff_diffnorm": 1.0e-6,
        "cutoff_iteration": 50,
        "subsettrunc": True,
        "direct_seed_grid": False,
    },
    "seeds": {
        "method": "sift",
        "n_seeds": 1,
    },
    "strain": {
        "enabled": True,
        "radius": 5,
    },
    "postprocess": {
        "enabled": True,
        "max_iterations": 8,
        "min_neighbors": 3,
        "corrcoef_threshold": 2.0,
    },
    "border": {
        "bcoef": 20,
        "interp": 20,
        "extrap": 20,
    },
}


def load_config(config: Optional[Union[str, dict]]) -> dict:
    if config is None:
        return dict(DEFAULT_CONFIG)

    if isinstance(config, dict):
        cfg = dict(DEFAULT_CONFIG)
        _deep_merge(cfg, config)
        return cfg

    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        cfg = dict(DEFAULT_CONFIG)
        if data:
            _deep_merge(cfg, data)
        return cfg

    raise TypeError(f"config must be None, str, Path, or dict, got {type(config)}")


def _deep_merge(base: dict, override: dict):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
