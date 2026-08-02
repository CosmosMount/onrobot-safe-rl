# SQRL paper-aligned Go2 reproduction

This experiment is isolated from `safe_droq`.  The existing agent remains the
project's enhanced SQRL-style research implementation; `paper_sqrl` implements
the algorithm in *Learning to be Safe: Deep RL with a Safety Critic*
(arXiv:2010.14603) without the later Go2-specific selector extensions.

## Protocol

The primary reproduction follows the paper's MinitaurVelocity experiment:

- pre-train at 0.30 m/s for 500,000 policy steps;
- jointly learn one standard twin-Q SAC actor and its Q_safe;
- alternate ordinary task-policy collection with complete trajectories from
  the current Q_safe-constrained policy;
- keep only the latest constrained-policy trajectories in `D_safe`;
- update after each configured set of k complete trajectories, even when a
  failure makes a trajectory much shorter than the nominal episode length;
- train Q_safe only with its Bellman MSE target, using an unconstrained actor
  action for the next-action target as specified in Appendix D;
- explore the safety boundary by selecting accepted actions immediately below
  epsilon during safety rollouts;
- transfer the same actor, reward critics, and Q_safe to 0.40 m/s;
- reset the target reward replay and freeze Q_safe for the main-paper run;
- use rejection sampling plus the SQRL Lagrange actor constraint during target
  fine-tuning;
- compare against standard SAC that is independently pre-trained at 0.30 m/s
  for the same 500,000 steps and then fine-tuned at 0.40 m/s, matching the
  ablation described in Section 7.1. A same-checkpoint comparison is useful as
  a separate critic-only diagnostic, but is not the paper's primary baseline.

Paper defaults used here are a two-layer 256-unit MLP, gamma_safe=0.7 and
epsilon_safe=0.1.  The environment-specific adaptation is that Go2 reports
`I(s')` on the transition entering a terminal fall; it is stored as the
immediate failure target for `Q_safe(s,a)`.

The paper does not publish numerical values for `n_off`, `k`, the size of the
small online trajectory buffer, or the number of rejection candidates. No
public author implementation was found. These are therefore exposed rather
than silently guessed: `n_off=1000` task transitions, `k=1` complete safety
trajectory, ten recent trajectories, and 100 candidates. Algorithm 1's single
Q_safe gradient update after each set of `k` trajectories is retained. These
preserve the paper's ordering and data semantics but
are Go2 reproduction parameters, not claimed author-code constants.

The formal configurations use two reward critics, no critic dropout or layer
normalization, and UTD=1. Earlier `pretrain_strict_v4` checkpoints used the
project's five-critic high-UTD DroQ backbone and a latest-value state mailbox;
they are retained only as diagnostics and are not valid paper-reproduction
checkpoints.

## 50 Hz collection protocol

The runtime and learner are deliberately asynchronous. The runtime publishes
every 50 Hz state to a lossless ordered SPSC queue. A dedicated collector
process owns an optimizer-free actor/Q_safe copy and assigns every action and
state `action_id`, `runtime_step_id`, `episode_id`, and `episode_step`. The
learner consumes complete transitions and periodically publishes immutable
network snapshots back to the collector. Stand-up/recovery ticks have
`applied_action_id=-1` and never enter reward or safety replay.

This separation is necessary because SAC updates can exceed the 20 ms control
period. The former synchronous loop silently skipped intermediate states and
one-cycle terminal messages, producing impossible trajectory lengths. The
formal protocol fails closed on a missing/reordered state, unknown action,
queue overflow, or a second runtime attempting to reset an active queue.

## Deliberately excluded from the reproduction agent

- H-step binary failure-window labels and BCE auxiliary loss;
- 50/50 failure/normal batch balancing;
- near-failure labels;
- counterfactual branch-ranking loss;
- behavior-support gate and action contraction;
- critic-A/critic-B validation;
- reward-Q candidate margin;
- structured fallback.  When no safe candidate exists, the paper rule of
  choosing the candidate with minimum predicted failure probability is used.

## Commands

Pre-training:

```bash
micromamba run -n oss python -m train \
  --config config/go2_50hz_sqrl_paper_pretrain.yaml --seed 42
```

The standalone inference runtime must use the same overlay because episode
truncation is owned by the runtime rather than the client-side `Go2Env`:

```bash
micromamba run -n oss python -m runtime.inference.runtime \
  --config config/go2_50hz_sqrl_paper_pretrain.yaml \
  --ordered-state-queue
```

Train the paper's independent SAC baseline and then fine-tune both methods from
their own 0.30 m/s checkpoints:

```bash
SQRL_PRETRAIN=saved/experiments/sqrl_paper/seed42/pretrain_strict_async_sac_v2/step_000000500000
SAC_PRETRAIN=saved/experiments/sqrl_paper/seed42/pretrain_sac_async_v1/step_000000500000

micromamba run -n oss python -m train \
  --config config/go2_50hz_sqrl_paper_sac_pretrain.yaml --seed 42

micromamba run -n oss python -m train \
  --config config/go2_50hz_sqrl_paper_sac_finetune.yaml \
  --initialize-from "$SAC_PRETRAIN" --seed 42

micromamba run -n oss python -m train \
  --config config/go2_50hz_sqrl_paper_finetune.yaml \
  --initialize-from "$SQRL_PRETRAIN" --seed 42
```

The two target runs use fresh reward replay buffers. Each target manifest must
match the final actor/reward-critic hashes of its own pre-training source. The primary comparison is
cumulative policy-transition falls and falls per 1,000 policy steps, with
return, episode length, velocity tracking, replacement rate, no-safe rate,
runtime throughput and safety constraint metrics reported alongside them.

## Acceptance

1. Unit tests prove recent-trajectory retention, reward/safety replay
   isolation, phase switching, and target masking.
2. A short live-stack integration run completes task and safety collection,
   performs Q_safe updates, and saves/restores a full checkpoint.
3. The 500k pre-train run finishes and produces a critic with both safe and
   failure experience. If natural failure data are absent, the run is reported
   as an unsuccessful reproduction rather than relabeled artificially.
4. SAC and SQRL each initialize from their method-specific completed 0.30 m/s
   checkpoint and complete the same number of 0.40 m/s policy steps.
5. Results are reported over paired seeds; seed 42 is the pilot, followed by
   additional seeds only after the protocol passes the pilot audit.

## Verified smoke tests (not experimental results)

- Ordered transport retained all 500 states and the terminal boundary while
  the test learner was completely stalled.
- Live Go2/MuJoCo SQRL smoke: 1,600 policy steps, four correctly bounded
  episodes (500/500/500/58), one natural fall, 601 learner calls, one exact
  Algorithm-1 Q_safe update, collector interval p50 20.44 ms and p95 21.41 ms,
  runtime queue depth 0 and transition queue maximum 7.
- Live standard-SAC smoke: 1,200 policy steps, two 500-step episodes, 201 SAC
  updates, collector interval p50 20.41 ms and p95 21.37 ms, runtime queue depth
  0, and the inference copy applied learner weight version 180.

These runs validate mechanics only. They are too short to support any claim
about SQRL fall reduction.

## Seed-42 strict reproduction result (2026-08-02)

The first complete 50 Hz pilot did **not** reproduce the paper's safety
advantage.  These are policy-transition falls; recovery ticks are excluded.

| 0.30 m/s pre-training | Steps | Episodes | Falls | Falls/1k | Episode fall rate |
|---|---:|---:|---:|---:|---:|
| Independent SAC | 500,000 | 1,068 | 100 | 0.200 | 9.36% |
| Strict public-spec SQRL | 500,000 | 1,137 | 205 | 0.410 | 18.03% |

The SQRL run therefore had 105 more falls and 2.05 times as many falls as
SAC.  Timing does not explain the result: median collector intervals were
20.41 ms for SQRL and 20.39 ms for SAC, and each run repeated only about
0.002% of actions.

Fresh construction from both published YAMLs at seed 42 gives the same actor
hash `6343903e...` and reward-critic hash `63e45573...`, so their initial SAC
parameters are identical.  The SQRL manifest's later `initial_*` values were
overwritten with resume-checkpoint hashes when that run resumed at step
200,179; this provenance bug affects the label, not the saved trajectory or
checkpoint lineage.  Future resumes now preserve the original hashes and
record separate `resume_*` hashes.

At 0.40 m/s, the matched early adaptation window is also negative:

| Fine-tuning through about 51.5k steps | Episodes | Falls | Episode fall rate | Mean length | Mean return |
|---|---:|---:|---:|---:|---:|
| SAC (51,514 steps) | 105 | 3 | 2.86% | 490.61 | 4,790.84 |
| SQRL (51,205 completed-episode boundary) | 123 | 24 | 19.51% | 416.30 | 3,829.59 |

The episode fall-rate difference is +16.66 percentage points for SQRL
(risk ratio 6.83; two-sided Fisher exact p=6.23e-5).  This is one seed, so it
is evidence of a failed pilot rather than a population-level claim.  The
longer SQRL run eventually reached 33 falls in 403,493 policy steps and a
cumulative 3.96% episode fall rate, but its safe set remained empty and its
Lagrange multiplier saturated at 100.  That late number therefore reflects
policy adaptation under a strong penalty and minimum-risk fallback, not a
successful SQRL rejection set.

## Why the strict pilot produced an empty safe set

Only 354 Q_safe gradient updates occurred during 500,000 pre-training policy
steps.  Every one of the 140,589 constrained-rollout steps had no candidate
below epsilon=0.10.  The final recent buffer also contained ten all-normal
episodes, so its failure memory had been evicted.  The critic still ranked
historical near-failure transitions reasonably well, but its absolute output
was miscalibrated: final candidate minima were about 0.21, above epsilon.

The paper publishes Algorithm 1's ordering but does not report `n_off`, `k`,
the number of Q_safe minibatches per cycle, target-network tau, output
initialization, buffer size, batch size, or rejection candidate count.  The
strict pilot used one update after each 1,000 task steps plus one complete
safety trajectory, exactly matching the literal single update shown in the
public pseudocode.  The result demonstrates that this particular completion
of the unspecified settings is insufficient on Go2; it does not prove that
the paper's private settings used the same update ratio.

### Isolated update-intensity diagnostic

To isolate network capacity from optimization intensity, the actor and reward
critics at step 450,441 were held bit-identical while Q_safe received 1,000
additional Bellman updates on its existing ten-trajectory replay (four
terminal failures, 3,832 normal transitions).  This is a diagnostic variant,
not part of the strict result.

| Extra Q_safe updates | Terminal AUROC | H32 near-failure AUROC | Safety-replay coverage | Recent task-replay coverage |
|---:|---:|---:|---:|---:|
| 0 | 0.998 | 0.885 | 0.0% | 0.0% |
| 250 | 1.000 | 0.843 | 0.0% | 0.0% |
| 500 | 1.000 | 0.816 | 0.0% | 0.0% |
| 750 | 1.000 | 0.799 | 0.2% | 0.2% |
| 1,000 | 1.000 | 0.808 | 88.4% | 48.8% |

After 1,000 updates, mean terminal-failure risk was 0.895 and normal risk was
0.098.  Thus a two-layer 256-unit network can represent a nonempty epsilon=.10
safe set; insufficient update intensity, not network depth, is the primary
explanation for the original absolute calibration failure.  The 88.4% versus
48.8% coverage gap also shows a remaining distribution mismatch between the
small constrained-policy safety replay and ordinary task-policy states.

The diagnostic is reproducible with:

```bash
micromamba run -n oss python scripts/recalibrate_paper_sqrl_safety.py \
  --checkpoint saved/experiments/sqrl_paper/seed42/pretrain_strict_async_sac_v2/step_000000450441 \
  --output saved/experiments/sqrl_paper/seed42/diagnostic_qsafe_updates_1000_from_450441 \
  --updates 1000 --device cuda
```

## Exact-snapshot action consequence gate

Before allowing the diagnostic critic to control the online experiment, 300
identical in-process MuJoCo snapshots were branched into the actor's nominal
action and the SQRL-selected action.  Both a single replacement and true
closed-loop filtering (a fresh candidate set at every 20 ms step) were tested.
The disturbance schedule and all nominal-policy random seeds were paired.

| Critic | H32 nominal failures | H32 closed-loop failures | Improved / worsened pairs | Difference | Paired p |
|---|---:|---:|---:|---:|---:|
| Original step-450,441 | 116 | 128 | 2 / 14 | +12 | 0.00418 |
| +1,000 Q_safe updates | 116 | 130 | 2 / 16 | +14 | 0.00131 |

For the recalibrated critic, a one-step replacement was nearly neutral
(117 versus 116 H32 failures), while repeated closed-loop filtering was
significantly worse.  Of the 300 initial snapshots, only 23 had a nonempty
safe set; in that small subset the closed-loop result was 16 versus 15
failures.  Most of the negative closed-loop effect came from repeated
minimum-risk fallback in unsupported states (114 versus 101 failures in the
277 initially empty-set snapshots).

This evaluator uses the same Go2 XML, 50 Hz action period and 500 Hz PD gains,
and restores MuJoCo integration state to within 3.1e-6 in observation space.
It is nevertheless a separate in-process backend: its natural states had only
0.1% epsilon=.10 coverage, whereas real training task replay had 48.8% after
recalibration.  The causal result is therefore strong evidence that Q_safe's
OOD/minimum-risk ordering must not be trusted, but it is not presented as a
direct estimate of the online stack's fall rate.

The pre-declared gate failed, so no online run with the recalibrated critic was
started.  The next scientifically justified experiment is to expose the
paper's missing Q_safe update ratio as a sweep during pre-training and require
both task-state safe-set coverage and held-out same-state action ranking before
fine-tuning.  Merely running more seeds under the current 100%-empty strict
setting would only replicate a known implementation completion failure.
