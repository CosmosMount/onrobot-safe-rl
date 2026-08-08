"""Evidence-safe grouped counterfactual data and evaluation utilities."""

from safety_data.metrics import evaluate_predictions
from safety_data.phase1_stats import (
    CommonGateStatus,
    OnlineGateThresholds,
    OnlineRun,
    Phase1EvidenceDecision,
    Phase1EvidenceError,
    RouteEvidence,
    RouteSpec,
    compile_phase1_evidence,
    evaluate_online_route,
)
from safety_data.paths import ProtectedEvidencePathError, assert_development_path
from safety_data.schema import (
    DatasetValidationError,
    GroupedBranchDataset,
    PrivilegedBranchView,
    SCHEMA_VERSION,
    audit_split_disjointness,
)

__all__ = [
    "DatasetValidationError",
    "CommonGateStatus",
    "GroupedBranchDataset",
    "OnlineGateThresholds",
    "OnlineRun",
    "Phase1EvidenceDecision",
    "Phase1EvidenceError",
    "PrivilegedBranchView",
    "ProtectedEvidencePathError",
    "SCHEMA_VERSION",
    "RouteEvidence",
    "RouteSpec",
    "assert_development_path",
    "audit_split_disjointness",
    "compile_phase1_evidence",
    "evaluate_online_route",
    "evaluate_predictions",
]
