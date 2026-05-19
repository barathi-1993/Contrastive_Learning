"""utils/config.py — YAML config loader with base inheritance and CLI --options override."""
import os
import yaml
from pathlib import Path
from addict import Dict

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(yaml_path: str, options: list = None) -> Dict:
    """
    Load a YAML config, merge with its _base_ parent if declared,
    then apply any CLI --options KEY=VALUE overrides.

    Args:
        yaml_path : path to the model-specific YAML (e.g. configs/simclr.yaml)
        options   : list of "Section.key=value" strings from argparse --options

    Returns:
        addict.Dict — dot-accessible config object
    """
    cfg_dir = Path(yaml_path).parent
    raw = _load_yaml(yaml_path)

    # Merge with base config if declared
    base_name = raw.pop("_base_", None)
    if base_name:
        base_path = cfg_dir / base_name
        base_raw  = _load_yaml(str(base_path))
        raw = _deep_merge(base_raw, raw)

    # Apply CLI overrides
    if options:
        for opt in options:
            if "=" not in opt:
                continue
            key_path, value = opt.split("=", 1)
            keys = key_path.split(".")
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            target = keys[-1]
            existing = d.get(target)
            # Try to cast to existing type
            if existing is None:
                d[target] = value
            elif isinstance(existing, bool):
                d[target] = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                d[target] = int(value)
            elif isinstance(existing, float):
                d[target] = float(value)
            elif isinstance(existing, list):
                import ast
                d[target] = ast.literal_eval(value)
            else:
                d[target] = value

    return Dict(raw)

def print_config(cfg: Dict, indent: int = 0) -> None:
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(Dict(v), indent + 2)
        else:
            print(" " * indent + f"{k}: {v}")
