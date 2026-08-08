"""Evidence-safe grouped counterfactual data and evaluation utilities."""

from safety_data.metrics import evaluate_predictions
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
    "GroupedBranchDataset",
    "PrivilegedBranchView",
    "ProtectedEvidencePathError",
    "SCHEMA_VERSION",
    "assert_development_path",
    "audit_split_disjointness",
    "evaluate_predictions",
]
