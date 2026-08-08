# Q_safe Phase 1, Iteration 1: falsification report

Status: **failed; no fall-reduction claim**  
Decision date: 2026-08-09  
Method: `selective_advantage_qsafe_v1`  
Evidence protocol: `config/qsafe_evidence_protocol.yaml`  
Generator/training commit: `c417fd0`  

## Executive decision

Iteration 1 did not establish that the learned Q_safe can rank locally safer
actions. The preregistered primary deployable model failed the model gate on
the one-shot held-out development test: pair accuracy was `0.5062`, selected
top-1 risk was **0.319 percentage points worse** than nominal, and its 95%
trajectory-cluster bootstrap interval included zero. Pointwise and privileged
diagnostics were also at chance.

The failure is not a dead action head or an optimization failure. A post-hoc
replica audit shows that the apparent 6.6--7.3 pp empirical Oracle reduction is
almost entirely the winner's curse from selecting the minimum of 9--16 noisy
R=8 candidate estimates and evaluating it on the same replicas. Selecting on
four replicas and evaluating on the other four gives approximately zero or
negative benefit on train, calibration, and test. The action-effect ordering
between halves is indistinguishable from chance.

Consequences:

- the primary model is rejected;
- selector calibration, paired closed-loop evaluation, and online SAC A/B
  evaluation are **not authorized** for this artifact;
- the consumed test seeds and their outcomes cannot be reused to choose or
  support a replacement method;
- Objective 1 remains incomplete and Phase 2 remains mechanically blocked;
- Iteration 2 must pass an independent discovery/audit-replica label gate
  before any model is trained.

## Immutable inputs and one-shot consumption

All files below are development evidence artifacts. No protected evaluation
artifact was opened or used.

### Grouped data

| Split | Groups / trajectories / source seeds | File SHA-256 | Content SHA-256 |
|---|---:|---|---|
| train | 1002 / 201 / 3 | `b6f27bf9300e43014bf3f6831e95b8540e8b290aec00f3edc17301f77743975b` | `064dad3e9a5e136413144a657fe0dd33d8e25e43816c05182145e7483021d16f` |
| calibration | 501 / 102 / 3 | `d231dd79dda5977c3c6749af171f0620b45f3091b24987f92aef9d58ae881648` | `f13a47dc472e747c24d91f28e0aa1ed6f8ede723d6d2f2ae05f4dd369de7bae1` |
| held-out test | 1002 / 201 / 3 | `48bfb0390b1aeb6c29b23cd5054199c7d36769f76246a6d1f0be10a5c54fd93f` | `1695008a8055d3f060ee3965beaabb270e425b7ed02ad2b125e0e7d087dd79be` |
| held-out privileged view | 1002 aligned groups | `c038d456cbbffd2ebd3a26b167c317f690e5707568b849e7cad58fea2a104827` | `88aaa69d36d957a2568b278552fbfdca86b4328e6ad3a8b4f287a97fef6d6722` |

Each split used K>=9 valid candidates, R=8 common-random-number replicas,
three source seeds, zero duplicate state fingerprints, and at least 75% mixed
groups. The train and merged test sets passed the preregistered data gate. The
three-way split audit checked all three split pairs and found no identity or
seed leakage. The calibration set intentionally had 501 groups: it passed the
selector-calibration minimum but was not represented as a 1,000-group training
set.

The merged held-out test used source seeds `7501`, `7502`, and `7503`. It was
consumed exactly once for each of the four preregistered run IDs. Durable
one-shot ledger markers were created before loading its arrays. These outcomes
are now permanently diagnostic/falsification evidence only.

### Runtime binding

- actor fingerprint: `f01e7cc36b9020631171c3dfe502d7426877aee0c33debd0a5c197206efc0908`
- actor state hash: `2074a3f00152df8e96cddf380623a9eb4bb63f84538dafca59cdf509f9559409`
- config hash: `ebf312ef27f64326a6ef478e0f86273af9f9cbaf61dd09b632fb10868524f726`
- causal contract hash: `3dfe8ede15636dadc8c5eef45fa7fdfe98a2da15ffc278d906cd8b01283a23de`
- consumed protocol hash: `f820c34a8964ca7ebe970ee48f2a684396f083d024d264db5555b12bf6071399`

## Preregistered one-shot results

Top-1 values are nominal fall risk minus the risk of the candidate selected by
the model. Positive is safer. Only the first row was claim-eligible; the other
rows were locked diagnostics and cannot become claims through multiple-model
selection.

| Run | Pair accuracy (95% CI where primary) | Strong pair | Top-1 reduction (95% CI where primary) | ECE | Decision |
|---|---:|---:|---:|---:|---|
| primary selective, deployable | `0.5062 [0.4880, 0.5240]` | `0.5243` | `-0.319 pp [-0.890, 0.225]` | `0.0570` | fail |
| pointwise, deployable | `0.5041` | `0.5141` | `-0.447 pp` | `0.0569` | diagnostic fail |
| state-only, deployable | `0.5000` | `0.5000` | `0.000 pp` | `0.0503` | diagnostic only |
| selective, privileged | `0.5074` | `0.5267` | `-0.345 pp` | `0.0566` | diagnostic fail |

The primary nominal empirical fall risk was `0.42362`; selected risk was
`0.42681`. Its same-replica empirical Oracle risk was `0.35343`, apparently a
`7.019 pp` reduction, but the independent-replica analysis below invalidates
that quantity as mechanism evidence. Primary test AUROC was `0.5274`; AUROC
cannot pass an action-ranking model by itself.

Artifact and marker hashes:

| Run | Artifact manifest SHA-256 | One-shot marker SHA-256 |
|---|---|---|
| primary | `8c51cab387f201ab5a5ec8de3dacc288f22ed554b11649b6cd235d72e0cad7d5` | `0d7194f5bc5bf15639330d35de0cf91a29e74f7941105af1a612959421190bc4` |
| pointwise | `c6a555b4a8372c96eb1bd45ed00b32f9cb75dae3a968a5e784554813536c57ab` | `ee9578ec08f0e266490f4660cd4a7e408a4db584a32947cd7c913586e7d7f5e8` |
| state-only | `3259676a2881e4f742cb167df54a03c9969def4dd8b2e41fdb829dc02a9810c9` | `0e8e609a224e2f5eedfbf4a6a7f70470c49883132edcd655ae2309697a8b537a` |
| privileged | `a92a9b95a993525a5efa90dcf7b15d9e08c5d4c7991bc4c978f1c8785ed74577` | `4dc7a652af022c5ea823144f628f1d66fee77ee73e82c281c3543fdecfe5397c` |

## Why the apparent Oracle was false headroom

### Fixed 4/4 independent-replica audit

For every group, replicas `0:4` selected all candidates tied for minimum
empirical risk; replicas `4:8` evaluated that selection. The direction was then
reversed and the two effects averaged. Ties used their order-invariant uniform
selection expectation. Confidence intervals are percentile intervals from
20,000 trajectory-cluster bootstrap samples.

| Split | Same-R full empirical Oracle | Symmetric cross-fit effect (95% CI) | IPW cross-fit effect (95% CI) |
|---|---:|---:|---:|
| train | `+6.599 pp` | `-0.112 pp [-0.491, 0.268]` | `-0.087 pp [-0.476, 0.300]` |
| calibration | `+7.285 pp` | `+0.177 pp [-0.382, 0.749]` | `+0.175 pp [-0.384, 0.739]` |
| test | `+6.999 pp` | `-0.275 pp [-0.663, 0.123]` | `-0.256 pp [-0.652, 0.149]` |
| stratified pooled | `+6.896 pp [6.555, 7.234]` | `-0.119 pp [-0.369, 0.124]` | `-0.102 pp [-0.355, 0.142]` |

Centered candidate-effect correlations between halves were
`0.00085/0.01022/0.00203` for train/calibration/test. Pair-order agreement,
including strong pairs, was approximately `0.500` in every split. The 95%
upper bound of the pooled reproducible effect was only `0.12--0.14 pp`, about
`1.7--2.0%` of the biased full-R Oracle.

This conclusion is insensitive to the tie rule. A first-index `np.argmin`
sensitivity analysis also produced zero/negative symmetric effects:

- train: `-0.187 pp [-0.525, 0.162]`;
- calibration: `-0.175 pp [-0.678, 0.347]`;
- test: `-0.212 pp [-0.574, 0.162]`.

Across all 70 oriented 4/4 complementary partitions, selection on a half
looked like a `+7.46--8.05 pp` improvement in that same half, but evaluation on
the complementary replicas was `-0.074--0.171 pp`; ordering agreement was
`0.5018--0.5032`. Candidate-versus-nominal fall outcomes were discordant in
only about 4.7--5.0% of replicas, so R=8 supplied only `0.38--0.40` discordant
events per candidate comparison on average. Taking a minimum over K candidates
amplified these individual random flips.

### The model learned the noise rather than collapsing

The primary full-train scores were pair `0.778`, strong pair `0.816`, and
top-1 `+3.650 pp`. When each training trajectory was predicted only by ensemble
members for which it was bootstrap-out-of-bag, 907 groups fell to pair `0.520`,
strong pair `0.524`, and top-1 `+0.373 pp`. Calibration and test were likewise
at chance.

Action-head parameter movement was 46--49% relative to initialization, its
gradient norm was `0.10--0.15`, and raw within-group advantage standard
deviation was about `0.60`. There were no NaNs, exploding values, or dead
gradients. Temperature scaling improved calibration but could not repair
ordering. The state-only model had the best held-out Brier score (`0.24489`),
which is consistent with learnable state danger but unidentifiable action
benefit.

## Root cause and scope of the negative result

The direct finding is narrow but decisive: under the declared one-step
candidate library, strong future branch impulses, H32 continuation, and R=8
measurement, there is no independently replicated candidate ordering that a
Q_safe can learn. This does not prove that every one-step action has exactly
zero physical effect, nor does it rule out a higher-replica estimate of a small
effect. It does prove that the same-R empirical Oracle and derived Oracle-gap
capture are invalid evidence for this iteration.

Likely contributors, in order:

1. R=8 is far too noisy for a roughly 5% paired-discordance process and K-way
   minimum selection.
2. A single 20 ms action intervention has too little persistent causal effect
   over H32 under the current continuation.
3. Branch impulses at steps 8 and 16 inject large future variation unrelated
   to the initial candidate.
4. The 5-member high-capacity model can memorize 201 training trajectories and
   candidate-kind extremes.

Changing model capacity, epochs, temperature, or privileged inputs cannot
create missing causal headroom, so no further architecture search is allowed
on these outcomes.

## Preregistered direction for Iteration 2

Iteration 2 uses a fresh development-only protocol,
`objective1_recovery_option_triage_v2`, and fresh source seeds
`7601--7603`. It will:

1. declare R=64 as 32 discovery plus 32 audit replicas before outcomes;
2. remove future branch-impulse magnitude while retaining disturbed source
   boundary states;
3. compare fixed 1--4 step linearly decayed residual recovery options;
4. choose the duration on discovery replicas exactly once and report raw audit
   falls without shrinkage;
5. require a positive trajectory-bootstrap lower bound before training any
   scorer;
6. train no model and consume no new held-out test if the label gate fails.

The exact candidate templates, seeds, budgets, tie rule, bootstrap seed, and
triage gates live in the new machine-readable protocol. A successful triage is
only authorization to preregister and collect a fresh v2 train/calibration/test
experiment. It is not Objective 1 evidence by itself. Objective 1 still
requires a new model gate, natural paired closed-loop benefit, and fall
reduction during SAC learning or at both small speed shifts. Phase 2 remains
blocked until every Objective 1 gate passes.
