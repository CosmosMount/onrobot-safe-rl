"""Online post-selection outcome bookkeeping for SQRL replacements."""

from __future__ import annotations

from collections import deque


class ReplacementOutcomeTracker:
    """Attach 8/16/32-step observed outcomes to every action replacement."""

    def __init__(self, horizons=(8, 16, 32)):
        self.horizons = tuple(sorted({int(value) for value in horizons}))
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError('replacement horizons must be positive')
        self.pending = []
        self.completed = []
        self.replacement_steps = deque(maxlen=self.horizons[-1])
        self.total_falls = 0
        self.false_negative_falls = {
            horizon: 0 for horizon in self.horizons}

    def record_replacement(self, step: int, info: dict) -> None:
        selection_kind = str(info.get(
            '_selection_kind', 'replacement'))
        if selection_kind == 'replacement':
            self.replacement_steps.append(int(step))
        self.pending.append({
            'step': int(step),
            'nominal_Q_safe': float(info.get(
                'nominal_Q_safe_A', float('nan'))),
            'selected_Q_safe': float(info.get(
                'selected_Q_safe', float('nan'))),
            'action_distance': float(info.get(
                'selected_nominal_action_distance', float('nan'))),
            'candidate_group': int(info.get(
                'sqrl_selected_group', -1)),
            'selection_kind': selection_kind,
            'age': 0,
            'failure': False,
            'near_failure': False,
            'time_to_failure': -1,
            'outcomes': {},
        })

    def record_step(
            self, step: int, *, unsafe: bool, near_failure: bool,
            done: bool) -> None:
        if unsafe:
            self.total_falls += 1
            for horizon in self.horizons:
                recent = any(
                    0 <= int(step) - replacement_step < horizon
                    for replacement_step in self.replacement_steps)
                self.false_negative_falls[horizon] += int(not recent)
        retained = []
        for event in self.pending:
            event['age'] += 1
            event['near_failure'] = bool(
                event['near_failure'] or near_failure)
            if unsafe and not event['failure']:
                event['failure'] = True
                event['time_to_failure'] = int(event['age'])
            for horizon in self.horizons:
                key = str(horizon)
                if key in event['outcomes']:
                    continue
                if event['age'] >= horizon or event['failure'] or done:
                    event['outcomes'][key] = {
                        'failure': bool(event['failure']),
                        'near_failure': bool(event['near_failure']),
                        'time_to_failure': int(
                            event['time_to_failure']),
                        'censored': bool(
                            done and not event['failure']
                            and event['age'] < horizon),
                    }
            if len(event['outcomes']) == len(self.horizons):
                self.completed.append(event)
            else:
                retained.append(event)
        self.pending = retained

    def finalize(self) -> None:
        for event in self.pending:
            for horizon in self.horizons:
                event['outcomes'].setdefault(str(horizon), {
                    'failure': bool(event['failure']),
                    'near_failure': bool(event['near_failure']),
                    'time_to_failure': int(event['time_to_failure']),
                    'censored': True,
                })
            self.completed.append(event)
        self.pending = []

    def metrics(self) -> dict[str, float]:
        result = {}
        for horizon in self.horizons:
            uncensored_events = [
                event for event in self.completed
                if not event['outcomes'][str(horizon)]['censored']]
            outcomes = [
                event['outcomes'][str(horizon)]
                for event in uncensored_events]
            result.update({
                f'sqrl/replacement_outcomes_h{horizon}':
                    float(len(outcomes)),
                f'sqrl/replacement_failure_rate_h{horizon}': (
                    float(sum(item['failure'] for item in outcomes)
                          / len(outcomes))
                    if outcomes else float('nan')),
                f'sqrl/replacement_near_failure_rate_h{horizon}': (
                    float(sum(item['near_failure'] for item in outcomes)
                          / len(outcomes))
                    if outcomes else float('nan')),
                f'sqrl/false_negative_falls_h{horizon}':
                    float(self.false_negative_falls[horizon]),
            })
            for kind in ('replacement', 'fallback'):
                kind_outcomes = [
                    event['outcomes'][str(horizon)]
                    for event in uncensored_events
                    if event['selection_kind'] == kind
                ]
                result.update({
                    f'sqrl/{kind}_outcomes_h{horizon}':
                        float(len(kind_outcomes)),
                    f'sqrl/{kind}_failure_rate_h{horizon}': (
                        float(sum(
                            item['failure'] for item in kind_outcomes)
                              / len(kind_outcomes))
                        if kind_outcomes else float('nan')),
                })
        return result
