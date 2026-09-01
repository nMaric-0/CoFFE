"""I/O utilities."""
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
