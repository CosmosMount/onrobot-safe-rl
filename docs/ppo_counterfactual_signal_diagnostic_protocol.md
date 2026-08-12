# PPO counterfactual supervision-signal diagnostic protocol

Date: 2026-08-12 (Asia/Shanghai)

This diagnostic explains why the previous Boundary+Medium strong-pair state
coverage was only 20.87%. It cannot train a safety critic, open or branch a
protected cohort, tune a selector, or run SAC transfer.

Before any new branch outcome is generated, 400 states are selected from the
previous development cohort using only state identity, original development
split, collector seed, and risk stratum. The fixed allocation is 320 train and
80 calibration states; within each split it preserves the original 2:1:1
Boundary/Medium/Normal ratio and a 1:1 seed137/seed138 ratio. Existing branch
risk, oracle result, fall count, and strong-pair membership are unavailable to
the selection function.

The exact 16 candidates and their physical q-targets are copied from the
previous development dataset. Replicas 5--16 use new paired-CRN streams. No PPO
candidate is sampled again.

R4, R8, and R16 use prefix replicas. Pair-ordering reliability compares
disjoint halves: 1--2 vs 3--4, 1--4 vs 5--8, and 1--8 vs 9--16. Independent
oracle action discovery and evaluation use the same disjoint halves. Horizon
diagnostics derive H16/H32/H64/H96 labels only from first-fall step; H96 remains
the final fall horizon and no horizon is selected post hoc.

The three diagnostic flags and their exact conditions are frozen in
`config/qsafe_counterfactual_signal_diagnostic_v1.yaml`. All uncertainty uses
state-group bootstrap. Results can inform a future protocol, but cannot alter
this diagnostic or authorize protected/SAC work.

