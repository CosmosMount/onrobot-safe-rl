# PPO same-state counterfactual supervision gate decision

Date: 2026-08-12 (Asia/Shanghai)

## Result

The pre-training informativeness gate failed, so critic training and every
later experiment were stopped. No new protected outcome was generated or
opened, and no SAC transfer experiment was started.

```text
counterfactual_supervision_informative=false
action_ranking_learnable=false
deployment_selector_supported=false
sac_transfer_authorized=false
formal_objective1_authorized=false
```

The failure is specifically a candidate-supervision result. It is not evidence
against the registered GRU/action architecture, pairwise loss, probability
calibration, or selector, because the protocol forbids training those models
after this gate fails.

## Fresh state collection

Both frozen boundary policies were loaded at `model_19.pt` and run at 2,000
parallel environments with stochastic actions, fixed +0.30 m/s command, zero
push/impulse/recovery/get-up, and first-fall terminal/reset.

| Collector | Frozen rollout transitions | Independent falls | Saved normals |
|---|---:|---:|---:|
| PPO seed 137 | 64,000,000 | 1,126 | 400 |
| PPO seed 138 | 64,000,000 | 1,299 | 400 |

The deterministic roster contained exactly 2,400 episode-disjoint states with
the preregistered train/calibration/protected and Boundary/Medium/Normal
quotas. The new protected roster contains identities and RNG plans only; its
branches have not been run. The consumed first-round 200-state cohort is in a
read-only protected directory and its 200 identities are rejected by
development code.

## Development branching

The development dataset contains 2,000 states, 16 actions per state, four
paired-CRN replicas, and H96 continuation: exactly 128,000 complete branches.
There were 8,194 branch falls. Candidate construction used one nominal plus 64
stochastic PPO proposals, physical-q-target deduplication at 0.025 normalized
RMS distance, three rank bins, and five selected actions per bin.

| Metric | Observed | Required |
|---|---:|---:|
| Boundary+Medium median empirical risk range | 0.25 | >= 0.25 |
| Boundary+Medium states with at least one strong pair | 20.87% | >= 30% |
| All-state fraction with risk range >= 0.25 | 59.45% | diagnostic |
| All-state fraction with risk range >= 0.50 | 18.70% | diagnostic |
| Mean non-tie pairs per state | 25.52 | diagnostic |
| Mean strong pairs per state | 3.34 | diagnostic |
| Development empirical oracle reduction | 4.9625 pp | diagnostic |

The risk-range threshold passed exactly, but the strong-pair state fraction
missed the fixed 30% gate. Results were also asymmetric across collectors:
24.1% for seed137 and 13.3% for seed138. Boundary states had 24.5% strong-pair
coverage, Medium 13.6%, and Normal 12.2%.

Therefore, the fixed 64-to-15 stochastic candidate generator and H96 protocol
did not create sufficiently strong within-state action-risk contrasts. The
positive development oracle value shows some candidate headroom, but it cannot
override the preregistered informativeness gate.

## Reproduction commands

The commands used for the two frozen-policy state archives were:

```bash
/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_frozen_ppo_counterfactual_states.py --checkpoint saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_19.pt --ppo-seed 137 --output saved/qsafe_counterfactual_v2/frozen-states-seed137-64m --aggregate-transitions 64000000 --normal-events 400
/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_frozen_ppo_counterfactual_states.py --checkpoint saved/qsafe_ppo_sqrl_v1/ppo-seed138-30m/model_19.pt --ppo-seed 138 --output saved/qsafe_counterfactual_v2/frozen-states-seed138-64m --aggregate-transitions 64000000 --normal-events 400
```

Roster, development branching, and decision:

```bash
python3 scripts/plan_counterfactual_ppo_states.py --seed137-root saved/qsafe_counterfactual_v2/frozen-states-seed137-64m --seed138-root saved/qsafe_counterfactual_v2/frozen-states-seed138-64m --development-output saved/qsafe_counterfactual_v2/development-roster.npz --protected-output saved/qsafe_protected/counterfactual-protected/v2-protected-identity-plan.npz --round-one-denylist config/qsafe_round_one_protected_denylist.json

/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_ppo_counterfactual_branches.py --roster saved/qsafe_counterfactual_v2/development-roster.npz --checkpoint137 saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_19.pt --checkpoint138 saved/qsafe_ppo_sqrl_v1/ppo-seed138-30m/model_19.pt --model-binary saved/qsafe_counterfactual_v2/frozen-states-seed137-64m/model.mjb --round-one-denylist config/qsafe_round_one_protected_denylist.json --replicas 4 --workers 16 --output saved/qsafe_counterfactual_v2/development-r4-h96.npz

python3 scripts/analyze_counterfactual_informativeness.py --dataset saved/qsafe_counterfactual_v2/development-r4-h96.npz --output saved/qsafe_counterfactual_v2/informativeness-gate.json
```

## Protocol consequence

The frozen stopping rule forbids model training, threshold tuning, protected
branching, SAC 2k/3k/5k transfer, fresh-SAC online A/B, formal Objective 1, and
speed expansion. A future cycle must preregister a changed candidate generator
or branching design and use a new protected cohort; it may not consume the
identity-only protected plan created in this failed cycle.
