"""MIRAGE Tier-1 capture service."""

# Runs before anything in this package reads a knob, so a configuration recorded
# under the pre-rename MIRAGE_* names still takes effect. See compat_env.
from .compat_env import apply_legacy_env_aliases as _apply_legacy_env_aliases

_apply_legacy_env_aliases()
