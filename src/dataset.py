"""BirdSet HSN audio loading, annotation, and domain splits."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 21
SAMPLE_RATE = 32_000
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS


def _value(record: Any, *names: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    else:
        for name in names:
            if hasattr(record, name):
                return getattr(record, name)
    return default


def _records(records: Any) -> list[Any]:
    if isinstance(records, (str, Path)):
        with Path(records).open(newline="") as file:
            return list(csv.DictReader(file))
    if hasattr(records, "to_dict"):
        return list(records.to_dict("records"))
    return list(records)


def _as_label_indices(value: Any, class_to_index: Mapping[str, int]) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    indices: list[int] = []
    for label in value:
        if isinstance(label, str):
            if label in class_to_index:
                indices.append(class_to_index[label])
            elif label.isdigit() and 0 <= int(label) < NUM_CLASSES:
                indices.append(int(label))
        elif isinstance(label, int) and 0 <= label < NUM_CLASSES:
            indices.append(label)
    return sorted(set(indices))


def _timestamp_intervals(value: Any, class_to_index: Mapping[str, int]) -> list[tuple[int, float, float]]:
    """Normalize common timestamp forms to (class index, start, end) tuples."""
    if value is None:
        return []
    intervals: list[tuple[int, float, float]] = []
    if isinstance(value, Mapping):
        for label, timestamps in value.items():
            index = _as_label_indices(label, class_to_index)
            if not index:
                continue
            if (
                isinstance(timestamps, (list, tuple))
                and len(timestamps) >= 2
                and all(isinstance(item, (int, float)) for item in timestamps[:2])
            ):
                timestamps = [timestamps]
            for timestamp in timestamps if isinstance(timestamps, (list, tuple)) else [timestamps]:
                if isinstance(timestamp, Mapping):
                    start = timestamp.get("start", timestamp.get("begin"))
                    end = timestamp.get("end", timestamp.get("stop"))
                else:
                    start, end = timestamp[:2]
                if start is not None and end is not None:
                    intervals.append((index[0], float(start), float(end)))
        return intervals
    for timestamp in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(timestamp, Mapping):
            label = timestamp.get("label", timestamp.get("class", timestamp.get("species")))
            start = timestamp.get("start", timestamp.get("begin"))
            end = timestamp.get("end", timestamp.get("stop"))
        else:
            label, start, end = timestamp[0], timestamp[1], timestamp[2]
        indices = _as_label_indices(label, class_to_index)
        if start is not None and end is not None:
            intervals.extend((index, float(start), float(end)) for index in indices)
    return intervals


class BirdSetHSNDataset(Dataset):
    """Dataset of fixed-length BirdSet HSN clips and log-mel features.

    Records may be dictionaries, dataclass-like objects, or rows from a
    pandas DataFrame. They must provide ``audio_path`` (or ``path``). Labels
    can be an iterable in ``labels``/``target`` or a 21-element binary vector.
    Timestamp annotations may be supplied in ``timestamps`` or
    ``call_timestamps`` as ``{label: [(start, end), ...]}``.
    """

    def __init__(
        self,
        records: Sequence[Any],
        class_to_index: Mapping[str, int] | None = None,
        sample_rate: int = SAMPLE_RATE,
        clip_seconds: float = CLIP_SECONDS,
    ) -> None:
        self.records = list(records)
        self.class_to_index = dict(class_to_index or {})
        self.sample_rate = sample_rate
        self.num_samples = int(round(sample_rate * clip_seconds))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=128,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self) -> int:
        return len(self.records)

    def _target(self, record: Any) -> torch.Tensor:
        labels = _value(record, "labels", "target", "species", "classes")
        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if isinstance(labels, (list, tuple)) and len(labels) == NUM_CLASSES:
            try:
                target[:] = torch.as_tensor(labels, dtype=torch.float32).clamp(0, 1)
                return target
            except (TypeError, ValueError):
                pass
        target[_as_label_indices(labels, self.class_to_index)] = 1.0
        return target

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = _value(record, "audio_path", "path", "filepath", "file")
        if path is None:
            raise KeyError("Each BirdSet record must contain audio_path or path")
        waveform, input_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if input_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_rate, self.sample_rate)
        waveform = waveform[:, : self.num_samples]
        if waveform.shape[-1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[-1]))
        spectrogram = self.to_db(self.mel(waveform)).squeeze(0)

        target = self._target(record)
        timestamps = _value(record, "call_timestamps", "timestamps", "temporal_annotations")
        mask = torch.zeros((NUM_CLASSES, spectrogram.shape[-1]), dtype=torch.float32)
        for class_index, start, end in _timestamp_intervals(timestamps, self.class_to_index):
            first = max(0, math.floor(start * self.sample_rate / 320))
            last = min(mask.shape[-1], math.ceil(end * self.sample_rate / 320) + 1)
            if last > first:
                mask[class_index, first:last] = 1.0
        recording_id = _value(record, "recording_id", "record_id", "id", default=Path(path).stem)
        return {
            "spectrogram": spectrogram,
            "target": target,
            "y": target,
            "mask": mask,
            "temporal_mask": mask,
            "mask_available": timestamps is not None,
            "recording_id": recording_id,
        }


def split_recordings(
    records: Sequence[Any],
    source_recording_ids: Sequence[str] | None = None,
    target_recording_ids: Sequence[str] | None = None,
    *,
    source_val_fraction: float = 0.2,
    target_val_fraction: float = 0.2,
    target_adapt_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, list[Any]]:
    """Partition whole recordings without allowing clips from one ID to leak."""
    rows = _records(records)
    ids = [str(_value(row, "recording_id", "record_id", "id")) for row in rows]
    source_ids = set(map(str, source_recording_ids or []))
    target_ids = set(map(str, target_recording_ids or []))
    if not source_ids and not target_ids:
        for row, recording_id in zip(rows, ids):
            domain = str(_value(row, "domain", "split", default="")).lower()
            (source_ids if domain in {"source", "s"} else target_ids).add(recording_id)
    if source_ids & target_ids:
        raise ValueError("Source and target recording IDs must be disjoint")
    unknown = set(ids) - source_ids - target_ids
    if unknown:
        raise ValueError(f"Unassigned recording IDs: {sorted(unknown)}")

    rng = random.Random(seed)

    def partition(domain_ids: set[str], val_fraction: float, names: tuple[str, str]) -> None:
        grouped = sorted(domain_ids)
        rng.shuffle(grouped)
        val_count = min(len(grouped), max(1 if grouped and val_fraction > 0 else 0, round(len(grouped) * val_fraction)))
        assignment = {recording_id: names[1] for recording_id in grouped[:val_count]}
        assignment.update({recording_id: names[0] for recording_id in grouped[val_count:]})
        for row, recording_id in zip(rows, ids):
            if recording_id in assignment:
                result[assignment[recording_id]].append(row)

    result: dict[str, list[Any]] = {name: [] for name in ("source_train", "source_val", "target_adapt", "target_val", "target_test")}
    partition(source_ids, source_val_fraction, ("source_train", "source_val"))
    target_group = sorted(target_ids)
    rng.shuffle(target_group)
    adapt_count = min(len(target_group), round(len(target_group) * target_adapt_fraction))
    adapt_ids, evaluation_ids = set(target_group[:adapt_count]), set(target_group[adapt_count:])
    partition(adapt_ids, 0, ("target_adapt", "target_adapt"))
    partition(evaluation_ids, target_val_fraction, ("target_test", "target_val"))
    return result


def split_loaders(
    records: Sequence[Any],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    **split_kwargs: Any,
) -> dict[str, DataLoader]:
    """Create DataLoaders for the five BirdSet transfer-learning partitions."""
    class_to_index = split_kwargs.pop("class_to_index", None)
    datasets = {
        name: BirdSetHSNDataset(rows, class_to_index=class_to_index)
        for name, rows in split_recordings(records, **split_kwargs).items()
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train if name in {"source_train", "target_adapt"} else False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }


create_split_loaders = split_loaders
get_split_loaders = split_loaders
get_dataloaders = split_loaders
split_loader = split_loaders
BirdSetDataset = BirdSetHSNDataset

__all__ = [
    "BirdSetHSNDataset",
    "BirdSetDataset",
    "NUM_CLASSES",
    "SAMPLE_RATE",
    "split_recordings",
    "split_loaders",
    "create_split_loaders",
    "get_split_loaders",
    "get_dataloaders",
    "split_loader",
]
