import os
import yaml

_CONFIG_CACHE = {}


def load_config(path=None):
    """Load YAML config. Caches by path to avoid re-reading."""
    if path is None:
        # Find configs/base.yaml relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, 'configs', 'base.yaml')

    path = os.path.abspath(path)
    if path not in _CONFIG_CACHE:
        with open(path, 'r') as f:
            _CONFIG_CACHE[path] = yaml.safe_load(f)
    return _CONFIG_CACHE[path]


def get_config(section=None):
    """Get full config or a specific section.

    Usage:
        cfg = get_config()           # full config
        cfg = get_config('rl')       # just rl section
        cfg = get_config('data')     # just data section
    """
    config = load_config()
    if section:
        return config.get(section, {})
    return config
