"""Legacy environment-variable aliases.

The project was renamed from its working name to MIRAGE on 2026-08-29, and every
knob it reads moved with it: ``SITARA_GAIT_PRESET`` is now ``MIRAGE_GAIT_PRESET``,
and so on for all 111 of them.

Every configuration recorded before that date - in the evaluation ledger, in shell
history, in a collaborator's run script - names the OLD variables. Those recorded
configurations are how a measurement is reproduced, so silently ignoring them would
turn a rename into a reproducibility break: the run would still succeed, quietly, at
the default value instead of the configured one.

So the old names keep working. For every ``SITARA_*`` in the environment, the matching
``MIRAGE_*`` is filled in if - and only if - it is not already set. The new name always
wins, nothing is overwritten, and no variable is ever unset.

This is a compatibility shim, not an interface. New code reads ``MIRAGE_*``.
"""

import os

__all__ = ["apply_legacy_env_aliases", "LEGACY_PREFIX", "PREFIX"]

LEGACY_PREFIX = "SITARA_"
PREFIX = "MIRAGE_"


def apply_legacy_env_aliases(environ=None):
    """Fill in ``MIRAGE_*`` from any ``SITARA_*`` that is set but has no new-name value.

    Returns the sorted list of new-name variables this call populated, so a caller
    that wants to warn about deprecated configuration can.
    """
    env = os.environ if environ is None else environ
    filled = []
    for key in list(env):
        if not key.startswith(LEGACY_PREFIX):
            continue
        new_key = PREFIX + key[len(LEGACY_PREFIX):]
        if new_key not in env:
            env[new_key] = env[key]
            filled.append(new_key)
    return sorted(filled)
