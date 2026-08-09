"""Grouped Selective Advantage Q_safe training and deployment primitives."""

from rl.qsafe.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    LoadedQSafeArtifact,
    load_qsafe_artifact,
    save_qsafe_artifact,
)
from rl.qsafe.data import (
    ActionView,
    GroupBatch,
    NormalizationStats,
    TorchGroupedView,
    trajectory_bootstrap_indices,
)
from rl.qsafe.loss import QSafeLossConfig, QSafeLossResult, qsafe_group_loss
from rl.qsafe.network import (
    EnsemblePrediction,
    QSafeEnsemble,
    QSafeNetworkConfig,
    QSafeOutput,
    SelectiveAdvantageQSafe,
)
from rl.qsafe.runtime import QSafeRuntimeResult, run_qsafe_step
from rl.qsafe.recovery_inference import (
    RecoveryQSafeInference,
    run_recovery_qsafe_inference,
)
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256,
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_VIEW,
    RecoveryProgramFeatures,
    bind_recovery_program_manifest,
    build_recovery_program_features,
    make_recovery_program_feature_manifest,
    validate_recovery_program_binding,
)
from rl.qsafe.recovery_runtime import (
    BoundActionProjectionProvider,
    CounterBasedActorShadow,
    PersistentRecoveryController,
    RecoveryActionOwner,
    RecoveryReplayAction,
    RecoveryRuntimeState,
    StageCActorCounterDomain,
    StageCActorCounterKey,
    StageDActorCounterDomain,
    StageDActorCounterKey,
)
from rl.qsafe.recovery_selector import (
    RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION,
    RecoveryConformalOffsets,
    RecoverySelection,
    RecoverySelectorBundle,
    RecoverySelectorConfig,
    select_recovery_program,
)

from rl.qsafe.selector import (
    CandidateBatch,
    SelectionResult,
    SelectorConfig,
    select_candidate,
)
from rl.qsafe.training import (
    QSafeTrainingConfig,
    TrainedQSafeEnsemble,
    TrainedQSafeMember,
    fit_temperature,
    predict_qsafe_ensemble,
    train_qsafe_ensemble,
    train_qsafe_member,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ActionView",
    "CandidateBatch",
    "EnsemblePrediction",
    "GroupBatch",
    "LoadedQSafeArtifact",
    "NormalizationStats",
    "QSafeEnsemble",
    "QSafeLossConfig",
    "QSafeLossResult",
    "QSafeNetworkConfig",
    "QSafeOutput",
    "QSafeRuntimeResult",
    "QSafeTrainingConfig",
    "SelectionResult",
    "BoundActionProjectionProvider",
    "CounterBasedActorShadow",
    "PersistentRecoveryController",
    "RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM",
    "RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256",
    "RECOVERY_PROGRAM_VIEW",
    "RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION",
    "RecoveryActionOwner",
    "RecoveryConformalOffsets",
    "RecoveryProgramFeatures",
    "RecoveryQSafeInference",
    "RecoveryReplayAction",
    "RecoveryRuntimeState",
    "RecoverySelection",
    "RecoverySelectorBundle",
    "RecoverySelectorConfig",
    "StageCActorCounterDomain",
    "StageCActorCounterKey",
    "StageDActorCounterDomain",
    "StageDActorCounterKey",
    "SelectiveAdvantageQSafe",
    "SelectorConfig",
    "TorchGroupedView",
    "TrainedQSafeEnsemble",
    "TrainedQSafeMember",
    "fit_temperature",
    "load_qsafe_artifact",
    "predict_qsafe_ensemble",
    "qsafe_group_loss",
    "run_qsafe_step",
    "run_recovery_qsafe_inference",
    "build_recovery_program_features",
    "bind_recovery_program_manifest",
    "make_recovery_program_feature_manifest",
    "validate_recovery_program_binding",
    "select_recovery_program",
    "select_candidate",
    "save_qsafe_artifact",
    "train_qsafe_ensemble",
    "train_qsafe_member",
    "trajectory_bootstrap_indices",
]
