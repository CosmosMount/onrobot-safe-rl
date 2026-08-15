# Paper alignment

| SQRL paper component | Go2 reproduction | Alignment |
|---|---|---|
| Maximum-entropy task learner | Independent vanilla twin-Q SAC | Exact concept |
| Tanh-Gaussian actor, target Q, learned alpha | `algo/sac.py` | Faithful |
| Sparse unsafe terminal state | Existing Go2 first-fall predicate | Adapted embodiment |
| Safety Bellman critic | `c[t+1] + (1-c[t+1]) gamma_safe Q_target` | Faithful transition-index adaptation |
| Policy evaluated by Q_safe | Current safety-constrained `bar_pi` | Frozen reproduction convention |
| Small on-policy replay | FIFO of latest complete `bar_pi` trajectories | Faithful |
| Offline task replay | Independent `D_task` populated only by `pi` | Faithful |
| Projection `Gamma_safe(pi)` | Finite candidate rejection sampling | Approximation |
| No accepted candidate | Minimum predicted Q_safe among sampled candidates | Finite-sample fallback |
| Target initialization | Same pre-trained actor for all three branches | Faithful paired design |
| Target reward critics | Reinitialized | Declared Go2 convention |
| Target alpha and replay | Reinitialized | Declared Go2 convention |
| Target Q_safe | Transferred and frozen | Faithful to Algorithm 2 |
| Fine-tune behavior | Masked `bar_pi` for SQRL branches | Faithful |
| Fine-tune safety objective | Nonnegative dual `nu` on `Q_safe-epsilon` | Faithful |
| Robot | Unitree Go2 | Embodiment change |
| Pre-training task | +0.30 m/s | Go2 analogue |
| Target task | +0.40 m/s | Go2 analogue |
| Observation | Five frames of the deployable 46D observation, flattened | Go2 adaptation |
| Action | 12D normalized absolute-joint-target command | Go2 adaptation |

## Explicit implementation decisions

The paper's theory and main-text Bellman equation evaluate the next action
under the policy whose risk is estimated. Algorithm 1 jointly trains Q_safe
with safety-constrained rollouts. Appendix D discloses that the authors'
experiments instead sampled the *unconstrained* policy for the Bellman target
because early Q_safe was too pessimistic. The frozen objective for this
reproduction explicitly requires `a[t+1] ~ bar_pi`, so the first version uses
the constrained next action. This is a deliberate protocol choice, not an
undisclosed claim about author code.

The paper does not publish `n_off`, recent-buffer size, rejection candidate
count, or safety updates per cycle. Their values in `config/base.yaml` are Go2
reproduction parameters. The development screening froze the run budgets at
25,000 pre-training and 10,000 target steps before any formal paired seeds.

No BCE, H96 label, ranking loss, ensemble, GRU, OOD abstention, reward-Q
selector, near-fall cost, adaptive target-phase Q_safe, DroQ, or FlashSAC
feature is present.
