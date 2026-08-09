# Q_safe Phase 1 recovery-option triage: no-headroom report

Status: **failed; redesign the recovery action library**  
Decision date: 2026-08-09  
Protocol: `objective1_recovery_option_triage_v2`  
Generator/analyzer commit: `a30f3ed5e925a6f19dac08e4da940cc009dd5be6`  

## Executive decision

The preregistered independent-replica triage found no practically useful
causal headroom in the frozen actor-residual recovery library. The selected
one-step option reduced audit fall risk by only `0.175` percentage points, and
the discovery-to-audit ordering agreement was `0.502`. The discovery-locked
two-step option reduced audit fall risk by only `0.279` percentage points.
Both are far below the required `5 pp` reduction, and neither satisfied the
required same-positive-direction check across all three source seeds.

All fixed durations L1--L4 had audit effects below `0.3 pp`, so the
preregistered `no_headroom_stop` rule fired. The resulting decision is
`stop_model_scaling_and_redesign_recovery_action_library`.

Consequences:

- no Q_safe model may be trained on or authorized by this triage;
- selector calibration, paired closed-loop evaluation, and online SAC A/B
  evaluation remain unauthorized;
- these discovery/audit outcomes are permanently diagnostic/falsification
  evidence and cannot be reused to select a replacement library;
- Objective 1 remains incomplete and Phase 2 remains blocked;
- the next iteration must introduce a materially stronger recovery mechanism,
  preregister it before outcomes, and use fresh source seeds and a fresh
  discovery/audit cohort.

## Immutable data and provenance

No protected evaluation artifact was opened or used. The collection used the
frozen 0.30 m/s actor checkpoint declared in the protocol, fresh development
source seeds `7601`, `7602`, and `7603`, K=29 recovery options, R=64
common-random-number replicas, and H=32 policy steps. Replica indices `0:32`
were assigned to discovery and `32:64` to audit before any candidate outcome.

| Artifact | File SHA-256 | Content SHA-256 |
|---|---|---|
| merged deployable, 384 groups | `62724a18173278a160032967a80bcf2bfb9ab05ac8504a4dbdbd9160dd7541ba` | `1b72950deb39d6af6b980ddee9beae4e904c8d23c9364403a28a1ed0563c6b50` |
| aligned privileged view | `5964965fa3a3d60cc437d1fcf2d6c979a6019b10e158b27bcceff61838401ecc` | `d9b6bff4d4b46418754cf02f28d23b6eaa71efabd6a68625f3a6e2e8c8dbedc6` |
| merge report | `1965ca00556d444b4fed867846453da3e94ef947fb8919498a63ab5655834443` | n/a |
| one-shot audit report | `912b4fdb11537068755944cc52ecadb44d8654fc6b3074cd20fcf899feef83cf` | n/a |
| cohort lock | `c1d0dc615c26139aa2e01f7c9af36a23f0f074afc5aae502de5a1a6b135273f3` | n/a |
| protocol | `a3a5f07515dff8a16a0ae518c0e1a834326bad845d918e084e47d708351ca790` | n/a |

The merged data gate passed every check: 384 independent groups, exactly 128
groups per source seed, 80 trajectory clusters, 29 valid candidates per group,
64 replicas per candidate, zero duplicate state fingerprints, and an
exhaustive pre-outcome discovery/audit partition. In total, the cohort contains
11,136 group-candidates and 712,704 H32 branch rollouts.

The three source shards were generated independently but bound to the same
clean commit and cohort lock. Their deployable file/content hashes were:

| Seed | Groups / trajectories | File SHA-256 | Content SHA-256 |
|---|---:|---|---|
| 7601 | 128 / 26 | `0ec026febdc0284fb1e1bf1f418ab58c62f7018c5f87369f3da78c8bdb149a9c` | `30c12e7c96a9e248a8274d3cc248ba2ea83733ed26b752abc28d6a86a6ec3a2b` |
| 7602 | 128 / 27 | `7625e49ea0f6bf52beda5baf76bf5f3c2c13f50de792fae6ee8f1124fbc5bbbf` | `49313ca3270272ad7676d150dd09d6985418cd76a14a14a9de1a92856105ff77` |
| 7603 | 128 / 27 | `2831b7666cebc3e8c6190c15f2f4e5b4d1a0a9ac7a1e9c835b3e6d7aa26681d8` | `0d531095fc5823cc47fb02d573f5590cb30f8402d90f6017cb403bff68392a1e` |

An initial direct-path CLI invocation failed at Python import before loading
the simulator, creating a cohort lock, or observing any outcome. Collection
then ran once through the module entry point. This bootstrap failure is not a
data look or an optional-stopping event.

## Preregistered discovery/audit result

Positive reduction means nominal audit fall risk minus selected audit fall
risk. Confidence intervals are central 90% trajectory-cluster bootstrap
intervals with 20,000 replicates. Candidate ties use the order-invariant
uniform expectation declared before collection.

| Duration | Audit reduction (90% CI) | Pair agreement | Per-seed reduction (pp) | Decision |
|---|---:|---:|---|---|
| L1 | `+0.175 pp [-0.018, 0.541]` | `0.5021` | `+0.517 / 0.000 / -0.002` | A fail |
| L2 | `+0.279 pp [0.002, 0.810]` | `0.5031` | `+0.800 / 0.000 / +0.023` | discovery-locked B fail |
| L3 | `+0.275 pp [-0.001, 0.808]` | `0.5035` | `+0.787 / 0.000 / +0.023` | diagnostic only |
| L4 | `+0.271 pp [-0.006, 0.801]` | `0.5024` | `+0.772 / 0.000 / +0.025` | diagnostic only |

The nominal audit fall risk was `3.008%`. The discovery-locked L2 selection
had audit risk `2.729%`. Its improvement over the locked L1 selection was only
`0.104 pp` (90% CI `0.008--0.266 pp`), far below the required `2 pp`.

The L1 gate failed all four substantive checks:

- absolute reduction `0.175 pp < 5 pp`;
- confidence lower bound `-0.018 pp <= 0`;
- discovery-to-audit pair agreement `0.502 < 0.58`;
- source-seed effects were not all positive.

The multistep gate failed absolute reduction, the `2 pp` improvement-over-L1
threshold, and the all-seeds-positive requirement. Its positive lower bounds
are evidence only for a very small effect and do not rescue practical
headroom. Per the locked failure policy, no runner-up duration or template was
tried on audit replicas.

## Mechanistic interpretation

The independent audit rules out replica-minimum winner's curse as the main
explanation for this result, but it also shows why a scorer is not yet useful.
For L1, 379 of 384 groups had a discovery minimum tie, with a mean tied winner
set of 7.90 candidates. Pair comparisons were tied in 10,671 of 10,752 cases.
L2--L4 were similarly tied and their discovery-to-audit ordering agreement
remained at chance. There is therefore almost no stable action ordering for a
Q_safe to learn from this library.

The narrow conclusion is that linearly decaying residuals around the same
frozen locomotion actor, applied for one to four 20 ms policy steps, do not
produce the required recovery authority at the sampled 0.30 m/s disturbed
states. Increasing network capacity, epochs, or replicas cannot turn the
observed sub-0.3 pp intervention effect into the required 5 pp reduction.

This does not rule out Q_safe as a safety-ranking architecture. It rules out
training another Q_safe before introducing recovery choices with a larger,
independently auditable causal effect, such as a genuinely separate backup
controller/policy or longer-horizon state-dependent recovery maneuver.

## Required next iteration

The next protocol must be written and committed before collecting any fresh
outcome. It must:

1. replace actor-local residual options with materially distinct recovery
   behaviors rather than extend L beyond four inside the same family;
2. use fresh source seeds and a new atomic cohort lock;
3. retain pre-outcome assignment, independent discovery/audit replicas,
   common random numbers, trajectory-cluster confidence intervals, and
   all-seed direction checks;
4. run a bounded mechanism-only screen before any new Q_safe training;
5. authorize a fresh train/calibration/test protocol only if a selected
   recovery behavior clears an absolute-effect and confidence gate;
6. keep Objective 1, paired closed-loop evaluation, online SAC evaluation,
   and Phase 2 blocked until their own preregistered gates pass.

