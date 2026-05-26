"""Shared pipeline configuration loader."""

from pathlib import Path


def load_config():
    """Load distill.config.yaml and resolve {{namespace}} templates."""
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml

    config_path = Path(__file__).resolve().parent.parent / "distill.config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ns = cfg["cluster"]["namespace"]
    def _resolve(val):
        if isinstance(val, str):
            return val.replace("{{namespace}}", ns)
        return val

    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = _resolve(v)
    return cfg
