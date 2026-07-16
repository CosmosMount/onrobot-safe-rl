"""Debug session NDJSON logger (session d88d84)."""
from __future__ import annotations

import json
import time
from pathlib import Path

_LOG = Path(__file__).resolve().parents[1] / '.cursor' / 'debug-d88d84.log'
_SESSION = 'd88d84'


def debug_log(hypothesis_id: str, location: str, message: str,
              data: dict, *, run_id: str = 'pre-fix') -> None:
    # region agent log
    payload = {
        'sessionId': _SESSION,
        'runId': run_id,
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data,
        'timestamp': int(time.time() * 1000),
    }
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')
    except OSError:
        pass
    # endregion
