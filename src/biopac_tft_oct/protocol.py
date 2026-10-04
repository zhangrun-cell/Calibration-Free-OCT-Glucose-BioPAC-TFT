"""Paper-level protocol definitions for OCT-conditioned blood glucose prediction.

The constants in this module make the two historical quantities in the study
explicitly distinct:

* ``initial_reference_length`` (L0) is the short reference blood glucose
  history used to initialize the shared temporal model.
* ``input_chunk_length`` (W) is the history required by the temporal model
  before it emits the first multi-step forecast.

Consequently, L0 = 10 does not imply that a prediction is available after ten
minutes. With W = 50 and H = 10, the first forecastable target starts after the
first 50 observed time points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionProtocol:
    """Fixed protocol used for the reported subject-wise prediction analyses."""

    initial_reference_length: int = 10
    input_chunk_length: int = 50
    output_chunk_length: int = 10
    n_subject_folds: int = 5
    aggregation_keep_ratio: float = 0.30

    def __post_init__(self) -> None:
        if self.initial_reference_length < 1:
            raise ValueError("initial_reference_length must be at least 1.")
        if self.input_chunk_length < self.initial_reference_length:
            raise ValueError("input_chunk_length must be at least initial_reference_length.")
        if self.output_chunk_length < 1:
            raise ValueError("output_chunk_length must be at least 1.")
        if self.n_subject_folds < 2:
            raise ValueError("n_subject_folds must be at least 2.")

    @property
    def first_forecastable_index(self) -> int:
        """First target index that can receive a forecast under the W=50 model."""

        return self.input_chunk_length

    def reference_history_mask(self, n_timepoints: int) -> np.ndarray:
        """Return True only for the L0 reference blood glucose input samples."""

        mask = np.zeros(int(n_timepoints), dtype=bool)
        mask[: min(mask.size, self.initial_reference_length)] = True
        return mask

    def evaluable_target_mask(self, n_timepoints: int) -> np.ndarray:
        """Return target points potentially evaluable after model-history loss.

        The returned mask does not encode session-specific aggregation edge
        exclusions. Those exclusions are applied after overlapping forecasts are
        aggregated.
        """

        mask = np.zeros(int(n_timepoints), dtype=bool)
        mask[self.first_forecastable_index :] = True
        return mask


@dataclass(frozen=True)
class SubjectFold:
    """One rotation of subject-wise five-fold cross-validation."""

    fold_index: int
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def subjectwise_fivefold_splits(
    subjects: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[SubjectFold]:
    """Yield deterministic train/validation/test rotations grouped by subject.

    Subjects, rather than minute-level samples or sessions, are shuffled and
    distributed across five folds. In each rotation one fold is held out for
    testing, the next fold is used for validation, and the other folds are used
    for training. Thus every participant serves as an unseen test participant
    exactly once, and all sessions from a participant remain together.
    """

    names = sorted({str(subject) for subject in subjects})
    if len(names) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique subjects, received {len(names)}.")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(names, dtype=object)
    rng.shuffle(shuffled)
    buckets = [tuple(str(x) for x in bucket) for bucket in np.array_split(shuffled, n_splits)]

    for fold_index in range(n_splits):
        test_subjects = buckets[fold_index]
        validation_subjects = buckets[(fold_index + 1) % n_splits]
        train_subjects = tuple(
            subject
            for bucket_index, bucket in enumerate(buckets)
            if bucket_index not in {fold_index, (fold_index + 1) % n_splits}
            for subject in bucket
        )
        yield SubjectFold(
            fold_index=fold_index,
            train_subjects=train_subjects,
            validation_subjects=validation_subjects,
            test_subjects=test_subjects,
        )
