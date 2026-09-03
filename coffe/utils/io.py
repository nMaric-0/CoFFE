"""I/O utilities."""

import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
