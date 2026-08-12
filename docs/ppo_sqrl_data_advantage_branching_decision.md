# PPO safety-data advantage and independent branching decision

Date: 2026-08-12 (Asia/Shanghai)

## Decision

The first-round PPO branching gate failed. The following flags are frozen:

```text
ppo_data_advantage_supported=false
action_signal_learnable=false
ppo_to_sac_transfer_supported=false
formal_objective1_authorized=false
```

Consequently, SAC 2k/3k/5k transfer branching, fresh-SAC online testing, the
24-seed Objective 1 experiment, and Objective 2 speed expansion were not run.
This is the protocol-required fail-closed outcome, not missing experimental
work.

The machine-readable decision is stored outside Git with the large experiment
artifacts at:

```text
saved/qsafe_ppo_sqrl_v1/branching/final-decision.json
```

## Data and training completed

- Two PPO seeds (137 and 138), each with early, boundary, and mature frozen
  collectors.
- 2,000 MuJoCo-Warp environments, stochastic actions, fixed +0.30 m/s command,
  zero pushes/impulses/recovery/get-up, and first-fall terminal/reset.
- 6,000,000 aggregate environment transitions in the master dataset.
- Whole-episode nested cohorts contained 999,394, 2,999,788, and 4,999,047
  transitions for the nominal 1M, 3M, and 5M budgets.
- Matched SQRL Bellman critics were trained on the three PPO cohorts, with
  state-only and shuffled-action controls and a critic trained from the
  existing SQRL/SAC safety replay.
- The compute-matched 3M and 5M diagnostics each used 2,714 gradient updates,
  matching the 1M critic.

## Offline diagnostics

| Critic | Test AUROC | Test AUPRC | Action permutation mean absolute change |
|---|---:|---:|---:|
| PPO 1M action | 0.9784 | 0.1852 | 0.000533 |
| PPO 3M action | 0.9312 | 0.1690 | 0.000211 |
| PPO 5M action | 0.9993 | 0.2515 | 0.000118 |
| PPO 5M state-only | 0.9995 | 0.2798 | 0 |
| PPO 5M shuffled | 0.9992 | 0.3121 | 0.000074 |

The 5M action critic did not beat either control on AUPRC. Action-permutation
sensitivity also decreased from 1M to 5M. This is evidence that the models
mostly learned state danger and increasingly ignored the current action.

With the gradient count fixed at 2,714, the 3M and 5M AUPRC values were 0.0731
and 0.0982. Thus neither proportional compute nor compute-matched training
produced a clean data-size benefit.

## Protected 200-state H96 branching

The protected audit evaluated 200 unseen boundary snapshots, 16 stochastic PPO
actions per snapshot, and eight paired-CRN H96 continuations per action. Oracle
selection used R1--R4; all reported outcomes used independent R5--R8.

| Selector | H96 fall rate | Reduction vs nominal | One-sided 95% LCB | Pair accuracy | Strong-pair accuracy |
|---|---:|---:|---:|---:|---:|
| PPO nominal | 44.50% | — | — | — | — |
| Independent oracle | 41.50% | 3.00 pp | +1.50 pp | — | — |
| Existing SQRL/SAC top-1 | 45.00% | -0.50 pp | -2.625 pp | 49.85% | 51.50% |
| PPO 1M top-1 | 42.75% | 1.75 pp | -0.125 pp | 43.13% | 43.11% |
| PPO 3M top-1 | 43.63% | 0.875 pp | -1.125 pp | 45.76% | 44.31% |
| PPO 5M top-1 | 44.25% | 0.25 pp | -1.625 pp | 49.71% | 40.72% |
| PPO 5M shuffled top-1 | 42.63% | 1.875 pp | -0.125 pp | 56.58% | 53.29% |

The action space is not the immediate bottleneck: the independently evaluated
oracle has a positive LCB. The learned critics are the bottleneck. No learned
PPO selector has a positive reduction LCB, and the shuffled control has a
larger point reduction than PPO 1M/3M/5M.

Thresholds were frozen using episode-disjoint PPO calibration data, selecting
the raw-Q threshold with at least 90% empirical first-fall recall. With those
thresholds, neither SQRL rejection nor minimal intervention passed the gate.

## Answers to the five research questions

1. **Did PPO data outperform the existing SQRL/SAC data?** No supported claim.
   PPO 1M had a better point estimate than the existing baseline, but its LCB
   was negative and it did not beat shuffled controls.
2. **Was improvement caused by quantity or boundary coverage?** Neither is
   established. Results were non-monotonic: 1M ranked better than 3M and 5M,
   while action sensitivity decreased with data size. Compute matching did not
   reverse the diagnosis.
3. **Can the critic choose a less fall-prone action in one state?** The candidate
   oracle can, but the learned critic cannot do so reliably. The best learned
   selector LCB was -0.125 pp.
4. **Does the PPO action-risk structure transfer to SAC?** Not tested because
   the mandatory PPO action-ranking prerequisite failed.
5. **Does it reduce fresh SAC-from-zero falls?** Not tested or claimed. Starting
   that experiment would violate the registered stopping rule.

## Next permitted research step

Do not tune a selector threshold on these protected outcomes. The next valid
development cycle must change the critic supervision or representation while
keeping this protected result sealed—for example, collect explicit same-state
multi-action targets for critic fitting, rebalance informative action pairs, or
use a pairwise/ranking loss. A new protected cohort is required before making a
new gate claim.
