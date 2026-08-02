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
