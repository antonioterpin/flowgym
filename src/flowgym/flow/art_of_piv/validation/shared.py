"""Shared validation types and constants for Art of PIV."""

from typing import Literal, TypeAlias

TrainType: TypeAlias = Literal["supervised", "rl"]
TrainingMode: TypeAlias = Literal["bandits", "regression", "classification"]

NORM_KINDS: tuple[str, ...] = ("batch", "group", "instance", "none")
TRAIN_TYPES: tuple[TrainType, ...] = ("supervised", "rl")
TRAINING_MODES: tuple[TrainingMode, ...] = (
    "bandits",
    "regression",
    "classification",
)
CLASSIFICATION_LOSSES: tuple[str, ...] = (
    "softmax_cross_entropy_with_integer_labels",
    "multiclass_hinge",
    "multiclass_perceptron",
)
GREEDY_ONLY_MODES: tuple[str, ...] = ("regression", "classification")


def _format_allowed(values: tuple[str, ...]) -> str:
    return ", ".join(repr(v) for v in values)
