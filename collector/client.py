"""Collector-facing protocol aliases.

`common.protocol` is the single source of truth for frame schemas.  This module
keeps collector imports short while avoiding duplicate dataclass definitions.
"""

from common.protocol import ActionFrame, ObservationFrame, TransitionFrame

__all__ = ['ObservationFrame', 'ActionFrame', 'TransitionFrame']
