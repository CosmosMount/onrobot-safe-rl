# P16: 0.30 m/s SAC vs transferred Q_safe

Both primary runs used seed 42, a 50 Hz runtime, and exactly 15,000
policy transitions. Their initial actor and reward-critic hashes match. The
transferred Q_safe remained frozen for the full masking run.

| Group | Policy steps | Falls | Falls/1k | Episode fall rate | Avg return | Avg length | Avg velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q_safe logging source (diagnostic) | 3,689 | 11 | 2.982 | 0.550 | n/a | n/a | 0.124 |
| Pure SAC | 15,000 | 56 | 3.733 | 0.683 | 1,306.704 | 179.756 | 0.215 |
| SAC + frozen Q_safe masking | 15,000 | 85 | 5.667 | 0.552 | 638.497 | 95.760 | 0.252 |

The masking run produced 29 more falls than pure SAC, a 51.8% increase.
It replaced 2,351 actions (15.67%) and abstained because no candidate was
below the threshold on 6,034 steps (40.23%). Replacement failure rates over
8/16/32 subsequent policy steps were 0.98%/2.98%/10.90%.

This experiment does not demonstrate fall reduction. The lower episode fall
rate for masking is caused by many more, shorter episodes and must not be
interpreted as improved safety; falls per fixed policy-step budget is the
primary metric. The source AUROC was measured on balanced training batches,
not an episode-disjoint action-ranking validation set, so it did not establish
that Q_safe could safely rank same-state candidate actions.

W&B runs:

- Source logging: `7zcxznwx`
- Pure SAC: `j7t6hn83`
- Frozen Q_safe masking: `t3ad2kq6`

## Exact-snapshot replacement evaluation

To separate action-ranking quality from online learning effects, the 5k, 10k,
and 15k masking checkpoints were evaluated in an in-process MuJoCo backend.
Each evaluated state was saved with `mjSTATE_INTEGRATION`. The nominal and
Q_safe-selected branches restored the same state, received the same
pre-snapshot base-velocity disturbance, differed only in the first action, and
then followed the same deterministic frozen actor for 8/16/32 steps. Maximum
observation mismatch after restore was `4.37e-6`.

The boundary dataset contains only states where the deployed selector would
replace the nominal action. Controlled disturbances were used to obtain
failure-informative boundary states; they are identical for both branches.

| Actor checkpoint | Pairs | Mean nominal risk | Mean selected risk | Nominal falls H32 | Selected falls H32 | Improved / worsened |
|---|---:|---:|---:|---:|---:|---:|
| 5k | 300 | 0.363 | 0.100 | 53 | 57 | 2 / 6 |
| 10k | 300 | 0.339 | 0.118 | 50 | 50 | 2 / 2 |
| 15k | 500 | 0.348 | 0.115 | 60 | 57 | 4 / 1 |
| Combined | 1,100 | — | — | 163 | 164 | 8 / 9 |

Across all checkpoints Q_safe predicted a large risk decrease, but the true
32-step outcome did not improve: selected actions caused 164 failures versus
163 for nominal actions. Only 17/1,100 pairs had different binary outcomes;
Q_safe chose the safer action in 8 of those 17 (`47.1%`). At 5k, when online
training was most failure-prone, selection was harmful (57 versus 53).

This result does not support useful same-state action ranking. It also shows
that a single replacement usually has little causal effect; repeated
replacement, action discontinuity, and learning-distribution interaction can
therefore still explain the larger online masking fall count. The next method
iteration should add counterfactual branch supervision and a conservative
replacement gate before running a multi-seed online matrix.

## Online-adaptive Q_safe diagnostic

A third 15,000-step run kept every frozen-masking setting fixed but allowed the
transferred Q_safe to continue training on target-policy transitions. Initial
actor, reward-critic, and safety-critic hashes match the frozen run.

| Group | Falls | Falls/1k | Avg return | Avg length | Avg velocity | Replacements | No-safe rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure SAC | 56 | 3.733 | 1306.7 | 179.8 | 0.215 | — | — |
| Frozen Q_safe | 85 | 5.667 | 638.5 | 95.8 | 0.252 | 2351 | 40.23% |
| Adaptive Q_safe | 52 | 3.467 | 1203.9 | 169.8 | 0.189 | 2153 | 17.80% |

Adaptive Q_safe reduced falls by 33 versus frozen masking and by 4 versus the
single-seed SAC run. Falls over the three 5k-step windows were `25/18/9`,
compared with frozen masking's `51/19/15` and SAC's `11/30/15`. Thus online
adaptation fixed much of the severe early mismatch and improved with target
data, but it did not protect the earliest phase as well as SAC.

The last-5k average velocity was `0.155 m/s` for adaptive Q_safe versus
`0.201 m/s` for SAC, so the four-fall advantage over SAC is confounded by more
conservative locomotion. Exact-snapshot evaluation of the final adaptive
checkpoint also found only a tiny causal effect: 80 nominal versus 78 selected
32-step failures over 500 replacement pairs. The critic predicted a much
larger risk reduction (`0.609` to `0.052`) than the observed reduction.

This single seed supports online adaptation over freezing, but does not yet
establish a speed-matched safety improvement over SAC.

## P17 clean-restart paired run

Both processes were started from a full MuJoCo/controller/runtime restart.
The actor and reward-critic initialization hashes match. Masking V2 was
inactive for the first 5,000 policy steps, then used a 2,000-step ramp.

| Group | Falls | Falls/1k | Avg return | Avg length | Avg velocity | Last-5k velocity |
|---|---:|---:|---:|---:|---:|---:|
| Pure SAC | 33 | 2.200 | 1806.8 | 243.3 | 0.212 | 0.198 |
| Adaptive Q_safe V2 | 53 | 3.533 | 945.1 | 142.8 | 0.241 | 0.151 |

Adaptive V2 replaced only 94 actions over its 10,000 active steps. Forty-five
of 53 falls had no replacement in the preceding 32 policy steps. Its
5k-window fall counts were `35/10/8`, versus SAC's `16/10/7`. The difference
therefore existed before masking activated: online Q_safe updates reduced
policy-step throughput from about 24 to 16 steps/s and changed the real-time
interaction trajectory. V2 does not establish a safety benefit.

The exact deployed V2 selector produced no replacements in 20,000 natural
standalone MuJoCo states because its combined risk, action-distance and
reward-Q gates were too strict. V3 contracted policy candidates into a local
RMS neighborhood and reduced auxiliary updates from once per policy step to
once per five policy steps. Under strong controlled disturbances it produced
100 replacement pairs, but H32 failures increased from 20 nominal to 22
selected. Previous-action contractions were also ineffective: the best
coefficient (`0.75 nominal + 0.25 previous`) changed 136 failures to 135 over
300 paired states.

These results locate the remaining bottleneck in same-state action ranking,
not pointwise failure classification. A first counterfactual dataset contained
1,000 snapshots and 8,000 candidate outcomes but only 60 mixed-outcome
snapshots. Branch-ranking fine-tuning achieved validation AUROC `0.885` and
Brier `0.135`, while pairwise accuracy remained `0.570`. Online masking is
therefore gated off until actively sampled mixed-outcome data reaches the
`0.65` pairwise threshold and held-out exact-snapshot H32 failure delta is
negative.

### Active counterfactual follow-up

Active sampling produced 400/400 mixed-outcome snapshots (3,200 candidate
outcomes, 1,609 failures), compared with only 60 mixed snapshots in the first
1,000-state dataset. A 50-epoch branch-ranked critic fit the training split
well (`0.882` pairwise accuracy) but reached only `0.599` on held-out states
from the same disturbance seed. On a completely independent seed, the best
binary pairwise accuracy was `0.487`.

Adding seeds 143 and 144 yielded 600 training snapshots while seed 242
remained fully held out. The best held-out pairwise accuracy was `0.501`.
Replacing the binary target with discounted time-to-failure risk improved the
best binary/TTF pair accuracies only to `0.527/0.534`.

The mixed states concentrate around the arbitrary horizon boundary: mean
time-to-failure is approximately 30--31 steps for a 32-step label. Small
candidate changes often shift failure across the cutoff by only one or two
steps, so hard binary labels are unstable across disturbance trajectories.

A privileged critic experiment appended normalized base height, tilt and
contact count. It improved pointwise AUROC and Brier but reached only `0.531`
held-out pairwise accuracy. Increasing the MLP from `256x256` to
`512x512x256` or `1024x512x256` gave `0.528` and `0.529`; larger networks
increased training fit without improving cross-seed action ranking. Network
depth is therefore not the limiting factor.

Finally, exact-snapshot closed-loop evaluation repeatedly applied the shield
for all 32 rollout steps, rather than changing only the first action. Across
300 high-risk states it changed failures from 155 to 154. A separate
1,000-state seed changed failures from 425 to 427. Combined, the deployed V3
shield produced 581 failures versus 580 for the nominal policy. This
controlled evidence shows no causal fall reduction, so another online 15k
matrix is not justified. The next critic must include history/contact phase
or use a predictive dynamics/recovery formulation before masking is enabled.

A final history-aware control tested 1, 2 and 4 observation frames together
with privileged height/tilt/contact context on independently collected seeds.
Best held-out pairwise accuracies were `0.530`, `0.499` and `0.504`.
History increased training fit but did not improve cross-seed ranking. Thus
neither MLP capacity nor short observation history resolves the action-level
identifiability problem in this setup. The supported next pivot is a
predictive rollout/dynamics shield or a state-level recovery trigger, not
additional SQRL candidate-ranking sweeps.

A smoother predictive-target follow-up collected 2,000 training states
(16,000 candidate rollouts) and 500 independent validation states with future
maximum tilt, minimum height and time-to-failure. Pointwise AUROC reached
`0.96`, but only 22 validation states had mixed binary outcomes and the best
pairwise accuracy was `0.576`. A final independent set explicitly sampled 100
mixed-outcome states. Its best pairwise accuracy was `0.533`; binary with
privileged context scored `0.498`, and continuous severity with context scored
`0.528`. Continuous targets improve state-level risk prediction but do not
make candidate ordering reliable.

The current Adaptive SQRL hypothesis is therefore rejected for this setup:
clean online training increased falls (`53` versus `33`), and controlled
closed-loop shielding produced `581` versus `580` failures. Further claims of
stable superiority require a method-level change, such as learned predictive
dynamics/rollout shielding or a state-level recovery trigger, followed by a
new paired multi-seed protocol.
