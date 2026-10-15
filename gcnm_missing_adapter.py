from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset


@dataclass
class GCNMMissingConfig:
    history: int
    horizon: int
    observed_ratio: float = 0.8
    missing_ratio: Optional[float] = None
    mask_option: str = "RM"
    spatial_k: int = 5
    nh: int = 3
    nd: int = 0
    nw: int = 0
    tau: Optional[int] = None
    day_span_steps: Optional[int] = None
    week_span_steps: Optional[int] = None
    missing_enabled: bool = True
    temporal_block_length: Optional[int] = None
    spatial_node_block: int = 1
    spatiotemporal_block_length: Optional[int] = None
    spatiotemporal_node_block: int = 1
    seed: int = 42


@dataclass
class GCRNODEConfig:
    history: int
    horizon: int
    observed_ratio: float = 0.8
    missing_ratio: Optional[float] = None
    mask_option: str = "RM"
    replay_day_size: int = 16
    replay_week_size: int = 64
    memory_dim: int = 32
    diffusion_depth: int = 2
    topk: int = 8
    ode_steps: int = 1
    ode_solver: str = "euler"
    kl_weight: float = 0.1
    missing_enabled: bool = True
    temporal_block_length: Optional[int] = None
    spatial_node_block: int = 1
    spatiotemporal_block_length: Optional[int] = None
    spatiotemporal_node_block: int = 1
    seed: int = 42


@dataclass
class MissingPatternConfig:
    enabled: bool = True
    pattern: str = "RM"
    observed_ratio: float = 0.8
    missing_ratio: Optional[float] = None
    temporal_block_length: Optional[int] = None
    spatial_node_block: int = 1
    spatiotemporal_block_length: Optional[int] = None
    spatiotemporal_node_block: int = 1
    seed: int = 42


def scale_gcnm_data(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    train_end: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_scale = np.max(np.abs(x_raw[: max(train_end, 1)]), axis=(0, 1), keepdims=True)
    y_scale = np.max(np.abs(y_raw[: max(train_end, 1)]), axis=(0, 1), keepdims=True)
    x_scale = np.where(x_scale < eps, 1.0, x_scale).astype(np.float32)
    y_scale = np.where(y_scale < eps, 1.0, y_scale).astype(np.float32)
    return (
        (x_raw / x_scale).astype(np.float32),
        (y_raw / y_scale).astype(np.float32),
        x_scale.astype(np.float32),
        y_scale.astype(np.float32),
    )


def compute_graph_distance_matrix(a_hat: np.ndarray) -> np.ndarray:
    adj = np.asarray(a_hat, dtype=np.float32)
    binary = (adj > 0).astype(np.int64)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            nbrs = np.flatnonzero(binary[node])
            for nbr in nbrs:
                cand = dist[src, node] + 1.0
                if cand < dist[src, nbr]:
                    dist[src, nbr] = cand
                    queue.append(int(nbr))
    finite = np.isfinite(dist)
    if not np.any(finite):
        return np.zeros((n, n), dtype=np.float32)
    fill_value = float(np.max(dist[finite]) + 1.0)
    dist[~finite] = fill_value
    return dist.astype(np.float32)


def _resolve_missing_ratio(config: object) -> float:
    explicit = getattr(config, "missing_ratio", None)
    if explicit is not None:
        return float(np.clip(float(explicit), 0.0, 1.0))
    observed_ratio = float(getattr(config, "observed_ratio", 1.0))
    return float(np.clip(1.0 - observed_ratio, 0.0, 1.0))


def _resolve_pattern(config: object) -> str:
    raw = str(getattr(config, "mask_option", "RM")).strip().upper()
    aliases = {
        "RANDOM": "RM",
        "RAND": "RM",
        "RM": "RM",
        "TEMPORAL": "TM",
        "TM": "TM",
        "SPATIAL": "SM",
        "SM": "SM",
        "SPATIOTEMPORAL": "STM",
        "SPATIO_TEMPORAL": "STM",
        "SPATIO-TEMPORAL": "STM",
        "STM": "STM",
        "NONE": "NONE",
    }
    return aliases.get(raw, raw)


def _connected_nodes(
    seed_node: int,
    node_count: int,
    graph_dists: Optional[np.ndarray],
) -> np.ndarray:
    if node_count <= 1 or graph_dists is None:
        return np.array([seed_node], dtype=np.int64)
    dists = np.asarray(graph_dists[seed_node], dtype=np.float32)
    order = np.argsort(dists, kind="stable")
    picked: List[int] = []
    for idx in order:
        if len(picked) >= node_count:
            break
        if not np.isfinite(dists[idx]):
            continue
        picked.append(int(idx))
    if seed_node not in picked:
        picked.insert(0, seed_node)
    return np.array(picked[: max(node_count, 1)], dtype=np.int64)


def _pick_block_length(length: int, requested: Optional[int], fallback_ratio: float) -> int:
    if requested is not None:
        return int(np.clip(requested, 1, length))
    return int(np.clip(round(max(fallback_ratio, 1.0 / max(length, 1)) * length), 1, length))


def _mask_random_per_node(
    length: int,
    num_nodes: int,
    num_features: int,
    missing_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((length, num_nodes, num_features), dtype=np.float32)
    entries_per_node = length * num_features
    missing_per_node = int(round(missing_ratio * entries_per_node))
    if missing_per_node <= 0:
        return mask
    for node in range(num_nodes):
        flat_idx = rng.choice(entries_per_node, size=min(missing_per_node, entries_per_node), replace=False)
        rows = flat_idx // num_features
        cols = flat_idx % num_features
        mask[rows, node, cols] = 0.0
    return mask


def _mask_temporal_stream_blocks(
    length: int,
    num_nodes: int,
    num_features: int,
    missing_ratio: float,
    block_length: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((length, num_nodes, num_features), dtype=np.float32)
    target_missing = int(round(missing_ratio * length * num_nodes * num_features))
    if target_missing <= 0:
        return mask
    block = _pick_block_length(length, block_length, missing_ratio)
    current_missing = 0
    max_attempts = max(target_missing * 4, 32)
    attempts = 0
    while current_missing < target_missing and attempts < max_attempts:
        attempts += 1
        node = int(rng.integers(0, num_nodes))
        feat = int(rng.integers(0, num_features))
        start = int(rng.integers(0, max(length - block + 1, 1)))
        end = min(start + block, length)
        before = mask[start:end, node, feat].sum()
        mask[start:end, node, feat] = 0.0
        current_missing += int(before)
    return mask


def _mask_spatial_node_blocks(
    length: int,
    num_nodes: int,
    num_features: int,
    missing_ratio: float,
    node_block: int,
    graph_dists: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((length, num_nodes, num_features), dtype=np.float32)
    target_missing = int(round(missing_ratio * length * num_nodes * num_features))
    if target_missing <= 0:
        return mask
    node_block = int(np.clip(node_block, 1, num_nodes))
    unit = num_features
    current_missing = 0
    max_attempts = max(target_missing * 2, 32)
    attempts = 0
    while current_missing < target_missing and attempts < max_attempts:
        attempts += 1
        t = int(rng.integers(0, length))
        seed_node = int(rng.integers(0, num_nodes))
        nodes = _connected_nodes(seed_node, node_block, graph_dists)
        before = mask[t, nodes, :].sum()
        mask[t, nodes, :] = 0.0
        current_missing += int(before)
    return mask


def _mask_spatiotemporal_node_blocks(
    length: int,
    num_nodes: int,
    num_features: int,
    missing_ratio: float,
    block_length: Optional[int],
    node_block: int,
    graph_dists: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((length, num_nodes, num_features), dtype=np.float32)
    target_missing = int(round(missing_ratio * length * num_nodes * num_features))
    if target_missing <= 0:
        return mask
    block = _pick_block_length(length, block_length, missing_ratio)
    node_block = int(np.clip(node_block, 1, num_nodes))
    current_missing = 0
    max_attempts = max(target_missing * 2, 32)
    attempts = 0
    while current_missing < target_missing and attempts < max_attempts:
        attempts += 1
        seed_node = int(rng.integers(0, num_nodes))
        start = int(rng.integers(0, max(length - block + 1, 1)))
        end = min(start + block, length)
        nodes = _connected_nodes(seed_node, node_block, graph_dists)
        before = mask[start:end, nodes, :].sum()
        mask[start:end, nodes, :] = 0.0
        current_missing += int(before)
    return mask


class MissingnessGenerator:
    def __init__(self, config: object, graph_dists: Optional[np.ndarray] = None):
        self.config = config
        self.graph_dists = None if graph_dists is None else np.asarray(graph_dists, dtype=np.float32)

    def generate(
        self,
        length: int,
        num_nodes: int,
        num_features: int,
        sample_seed: int,
    ) -> np.ndarray:
        if not bool(getattr(self.config, "missing_enabled", True)):
            return np.ones((length, num_nodes, num_features), dtype=np.float32)
        pattern = _resolve_pattern(self.config)
        missing_ratio = _resolve_missing_ratio(self.config)
        if pattern == "NONE" or missing_ratio <= 0.0:
            return np.ones((length, num_nodes, num_features), dtype=np.float32)
        rng = np.random.default_rng(sample_seed)
        if pattern == "RM":
            return _mask_random_per_node(length, num_nodes, num_features, missing_ratio, rng)
        if pattern == "TM":
            return _mask_temporal_stream_blocks(
                length,
                num_nodes,
                num_features,
                missing_ratio,
                getattr(self.config, "temporal_block_length", None),
                rng,
            )
        if pattern == "SM":
            return _mask_spatial_node_blocks(
                length,
                num_nodes,
                num_features,
                missing_ratio,
                int(getattr(self.config, "spatial_node_block", 1)),
                self.graph_dists,
                rng,
            )
        if pattern == "STM":
            return _mask_spatiotemporal_node_blocks(
                length,
                num_nodes,
                num_features,
                missing_ratio,
                getattr(self.config, "spatiotemporal_block_length", None),
                int(getattr(self.config, "spatiotemporal_node_block", 1)),
                self.graph_dists,
                rng,
            )
        raise ValueError(f"Unsupported missing pattern: {pattern}")


def apply_missing_pattern(config: object, pattern: MissingPatternConfig) -> object:
    if hasattr(config, "missing_enabled"):
        setattr(config, "missing_enabled", pattern.enabled)
    if hasattr(config, "mask_option"):
        setattr(config, "mask_option", pattern.pattern)
    if hasattr(config, "missing_ratio"):
        setattr(config, "missing_ratio", pattern.missing_ratio)
    if hasattr(config, "observed_ratio"):
        observed = pattern.observed_ratio
        if pattern.missing_ratio is not None:
            observed = 1.0 - float(pattern.missing_ratio)
        setattr(config, "observed_ratio", observed)
    if hasattr(config, "temporal_block_length"):
        setattr(config, "temporal_block_length", pattern.temporal_block_length)
    if hasattr(config, "spatial_node_block"):
        setattr(config, "spatial_node_block", pattern.spatial_node_block)
    if hasattr(config, "spatiotemporal_block_length"):
        setattr(config, "spatiotemporal_block_length", pattern.spatiotemporal_block_length)
    if hasattr(config, "spatiotemporal_node_block"):
        setattr(config, "spatiotemporal_node_block", pattern.spatiotemporal_node_block)
    if hasattr(config, "seed"):
        setattr(config, "seed", pattern.seed)
    return config


def _window_mask(
    length: int,
    num_nodes: int,
    num_features: int,
    config: object,
    sample_seed: int,
    graph_dists: Optional[np.ndarray] = None,
) -> np.ndarray:
    generator = MissingnessGenerator(config, graph_dists=graph_dists)
    return generator.generate(length, num_nodes, num_features, sample_seed).astype(np.float32)


def _build_stat_channels(
    x_window: np.ndarray,
    mask: np.ndarray,
    graph_dists: np.ndarray,
    spatial_k: int,
    step_delta: float,
) -> np.ndarray:
    length, num_nodes, num_features = x_window.shape
    observed = x_window * mask

    x_last = np.zeros_like(observed, dtype=np.float32)
    x_mean_t = np.zeros_like(observed, dtype=np.float32)
    delta_t = np.zeros_like(observed, dtype=np.float32)
    x_closest = observed.copy()
    x_mean_s = np.zeros_like(observed, dtype=np.float32)
    delta_s = np.zeros_like(observed, dtype=np.float32)

    obs_count = mask.sum(axis=0, keepdims=True)
    obs_sum = observed.sum(axis=0, keepdims=True)
    mean_t = np.divide(obs_sum, np.maximum(obs_count, 1.0), where=np.maximum(obs_count, 1.0) > 0)
    x_mean_t[:] = mean_t

    last_val = np.zeros((num_nodes, num_features), dtype=np.float32)
    gaps = np.zeros((num_nodes, num_features), dtype=np.float32)
    for t in range(length):
        miss = mask[t] == 0.0
        x_last[t][miss] = last_val[miss]
        delta_t[t][miss] = gaps[miss]
        obs = ~miss
        last_val[obs] = observed[t][obs]
        gaps[obs] = step_delta
        gaps[miss] += step_delta

    sorted_neighbors = np.argsort(graph_dists, axis=1)
    for t in range(length):
        observed_now = mask[t] == 1.0
        for node in range(num_nodes):
            neighbor_ids = [idx for idx in sorted_neighbors[node] if idx != node]
            if spatial_k > 0:
                neighbor_ids = neighbor_ids[: max(spatial_k, 1)]
            for feat in range(num_features):
                if observed_now[node, feat]:
                    continue
                found = False
                spatial_vals: List[float] = []
                for nbr in neighbor_ids:
                    if observed_now[nbr, feat]:
                        spatial_vals.append(float(observed[t, nbr, feat]))
                        if not found:
                            x_closest[t, node, feat] = observed[t, nbr, feat]
                            delta_s[t, node, feat] = graph_dists[node, nbr]
                            found = True
                if spatial_vals:
                    x_mean_s[t, node, feat] = float(np.mean(spatial_vals))

    stacked = np.stack(
        [observed, mask, x_last, x_mean_t, delta_t, x_closest, x_mean_s, delta_s],
        axis=0,
    )
    return stacked.astype(np.float32)


def _retrieve_hist(
    target_start: int,
    full_x: np.ndarray,
    config: GCNMMissingConfig,
) -> np.ndarray:
    history = config.history
    tau = config.tau or history
    start = target_start - history
    end = target_start - 1
    offsets = np.arange(history, dtype=np.int64)
    full_len = full_x.shape[0]

    def stack_recent(start_anchor: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, history, full_x.shape[1], full_x.shape[2]), dtype=np.float32)
        start_hist = start_anchor - count
        end_hist = end - history
        if start_hist < 0:
            current = full_x[start : end + 1]
            return np.tile(current[None, ...], (count, 1, 1, 1)).astype(np.float32)
        seqs = []
        for t in range(start_hist, end_hist + 1):
            seqs.append(full_x[t + offsets])
        return np.stack(seqs, axis=0).astype(np.float32)

    recent = stack_recent(start, config.nh * tau)
    blocks = [recent]

    def periodic_block(multiplier: int, count: int, gap_steps: Optional[int]) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, history, full_x.shape[1], full_x.shape[2]), dtype=np.float32)
        if gap_steps is None or gap_steps <= 0:
            current = full_x[start : end + 1]
            return np.tile(current[None, ...], (count * tau, 1, 1, 1)).astype(np.float32)
        block_list = []
        current = np.tile(full_x[start : end + 1][None, ...], (tau, 1, 1, 1)).astype(np.float32)
        for k in range(1, count + 1):
            ref_start = start - multiplier * k * gap_steps - tau // 2
            ref_end = end - multiplier * k * gap_steps - history + tau // 2
            if ref_start < 0 or ref_end >= full_len:
                block_list.append(current)
                continue
            seqs = []
            for t in range(ref_start, ref_end + 1):
                if t < 0 or (t + history) > full_len:
                    continue
                seqs.append(full_x[t + offsets])
            if len(seqs) == tau:
                current = np.stack(seqs, axis=0).astype(np.float32)
            block_list.append(current)
        if not block_list:
            return np.zeros((0, history, full_x.shape[1], full_x.shape[2]), dtype=np.float32)
        return np.concatenate(block_list, axis=0)

    blocks.append(periodic_block(1, config.nd, config.day_span_steps))
    blocks.append(periodic_block(1, config.nw, config.week_span_steps))
    non_empty = [block for block in blocks if block.shape[0] > 0]
    if not non_empty:
        current = full_x[start : end + 1]
        return current[None, ...].astype(np.float32)
    return np.concatenate(non_empty, axis=0).astype(np.float32)


class GCNMMissingWindowDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        graph_dists: np.ndarray,
        start: int,
        end: int,
        config: GCNMMissingConfig,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.graph_dists = graph_dists.astype(np.float32)
        self.config = config
        first_target_start = max(int(start), config.history)
        last_target_start = int(end) - config.horizon
        if last_target_start < first_target_start:
            self.target_starts: List[int] = []
        else:
            self.target_starts = list(range(first_target_start, last_target_start + 1))
        diffs = np.diff(np.arange(config.history + 1, dtype=np.float32))
        self.step_delta = 1.0 if len(diffs) == 0 else 1.0

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_start = self.target_starts[index]
        x_window = self.x_full[target_start - self.config.history : target_start]
        y_window = self.y_full[target_start : target_start + self.config.horizon]
        mask = _window_mask(
            self.config.history,
            x_window.shape[1],
            x_window.shape[2],
            self.config,
            self.config.seed + target_start,
            graph_dists=self.graph_dists,
        )
        x_stat = _build_stat_channels(
            x_window=x_window,
            mask=mask,
            graph_dists=self.graph_dists,
            spatial_k=self.config.spatial_k,
            step_delta=self.step_delta,
        )
        x_hist = _retrieve_hist(target_start, self.x_full, self.config)
        return (
            torch.tensor(x_stat, dtype=torch.float32),
            torch.tensor(x_hist, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32),
        )


class SBUWindowDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        graph_dists: Optional[np.ndarray],
        start: int,
        end: int,
        config: GCNMMissingConfig,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.graph_dists = None if graph_dists is None else graph_dists.astype(np.float32)
        self.config = config
        first_target_start = max(int(start), config.history)
        last_target_start = int(end) - config.horizon
        if last_target_start < first_target_start:
            self.target_starts: List[int] = []
        else:
            self.target_starts = list(range(first_target_start, last_target_start + 1))

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int):
        target_start = self.target_starts[index]
        x_window = self.x_full[target_start - self.config.history : target_start]
        y_window = self.y_full[target_start : target_start + self.config.horizon]
        mask = _window_mask(
            self.config.history,
            x_window.shape[1],
            x_window.shape[2],
            self.config,
            self.config.seed + target_start,
            graph_dists=self.graph_dists,
        )
        x_masked = x_window * mask
        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32),
        )


class GCRNODEWindowDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        graph_dists: Optional[np.ndarray],
        start: int,
        end: int,
        config: GCRNODEConfig,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.graph_dists = None if graph_dists is None else graph_dists.astype(np.float32)
        self.config = config
        first_target_start = max(int(start), config.history)
        last_target_start = int(end) - config.horizon
        if last_target_start < first_target_start:
            self.target_starts: List[int] = []
        else:
            self.target_starts = list(range(first_target_start, last_target_start + 1))

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int):
        target_start = self.target_starts[index]
        x_window = self.x_full[target_start - self.config.history : target_start]
        y_window = self.y_full[target_start : target_start + self.config.horizon]
        mask = _window_mask(
            self.config.history,
            x_window.shape[1],
            x_window.shape[2],
            self.config,  # type: ignore[arg-type]
            self.config.seed + target_start,
            graph_dists=self.graph_dists,
        )
        x_masked = x_window * mask
        hist_idx = np.arange(target_start - self.config.history, target_start, dtype=np.int64)
        fut_idx = np.arange(target_start, target_start + self.config.horizon, dtype=np.int64)
        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32),
            torch.tensor(hist_idx, dtype=torch.long),
            torch.tensor(fut_idx, dtype=torch.long),
        )


class MaskedWindowDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        graph_dists: Optional[np.ndarray],
        start: int,
        end: int,
        config: object,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.graph_dists = None if graph_dists is None else graph_dists.astype(np.float32)
        self.config = config
        history = int(getattr(config, "history"))
        horizon = int(getattr(config, "horizon"))
        first_target_start = max(int(start), history)
        last_target_start = int(end) - horizon
        if last_target_start < first_target_start:
            self.target_starts: List[int] = []
        else:
            self.target_starts = list(range(first_target_start, last_target_start + 1))

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_start = self.target_starts[index]
        history = int(getattr(self.config, "history"))
        horizon = int(getattr(self.config, "horizon"))
        x_window = self.x_full[target_start - history : target_start]
        y_window = self.y_full[target_start : target_start + horizon]
        mask = _window_mask(
            history,
            x_window.shape[1],
            x_window.shape[2],
            self.config,
            int(getattr(self.config, "seed", 42)) + target_start,
            graph_dists=self.graph_dists,
        )
        x_masked = x_window * mask
        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32),
        )


def build_augmented_sequence(
    x_window: np.ndarray,
    mask: np.ndarray,
    graph_dists: Optional[np.ndarray] = None,
    spatial_k: int = 1,
    neighbor_matrix: Optional[np.ndarray] = None,
    sorted_neighbors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build lightweight missingness-aware inputs for MFiLM-style models.

    Output channels are concatenated as:
    [x_masked, obs_mask, delta_norm, last_observed, temporal_mean, spatial_mean,
     closest_spatial, temporal_count, spatial_count].
    """
    x_masked = x_window * mask
    length, num_nodes, num_features = x_window.shape
    x_last = np.zeros_like(x_window, dtype=np.float32)
    delta = np.zeros_like(x_window, dtype=np.float32)
    x_mean_t = np.zeros_like(x_window, dtype=np.float32)
    x_mean_s = np.zeros_like(x_window, dtype=np.float32)
    x_closest = np.zeros_like(x_window, dtype=np.float32)
    t_count = np.zeros_like(x_window, dtype=np.float32)
    s_count = np.zeros_like(x_window, dtype=np.float32)
    last_val = np.zeros((num_nodes, num_features), dtype=np.float32)
    gap = np.zeros((num_nodes, num_features), dtype=np.float32)
    observed = x_window * mask
    obs_count = mask.sum(axis=0, keepdims=True)
    obs_sum = observed.sum(axis=0, keepdims=True)
    denom_t = np.maximum(obs_count, 1.0)
    mean_t = np.zeros_like(obs_sum, dtype=np.float32)
    np.divide(obs_sum, denom_t, out=mean_t, where=denom_t > 0)
    for t in range(length):
        x_last[t] = last_val
        delta[t] = gap
        x_mean_t[t] = mean_t
        t_count[t] = obs_count
        observed = mask[t] > 0.0
        last_val = np.where(observed, x_window[t], last_val)
        gap = np.where(observed, 0.0, gap + 1.0)
    if neighbor_matrix is None and graph_dists is not None:
        neighbor_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        sorted_neighbors = np.argsort(graph_dists, axis=1)
        for node in range(num_nodes):
            neighbor_ids = [idx for idx in sorted_neighbors[node] if idx != node]
            if spatial_k > 0:
                neighbor_ids = neighbor_ids[: max(spatial_k, 1)]
            if neighbor_ids:
                neighbor_matrix[node, neighbor_ids] = 1.0
    if neighbor_matrix is not None:
        weights = neighbor_matrix.astype(np.float32)
        for t in range(length):
            observed_vals = x_window[t] * mask[t]
            observed_counts = weights @ mask[t]
            observed_sums = weights @ observed_vals
            s_count[t] = observed_counts
            np.divide(
                observed_sums,
                np.maximum(observed_counts, 1.0),
                out=x_mean_s[t],
                where=observed_counts > 0,
            )
            if sorted_neighbors is not None:
                observed_now = mask[t] == 1.0
                for node in range(num_nodes):
                    neighbor_ids = [idx for idx in sorted_neighbors[node] if idx != node]
                    if spatial_k > 0:
                        neighbor_ids = neighbor_ids[: max(spatial_k, 1)]
                    if not neighbor_ids:
                        continue
                    for feat in range(num_features):
                        for nbr in neighbor_ids:
                            if observed_now[nbr, feat]:
                                x_closest[t, node, feat] = x_window[t, nbr, feat]
                                break
    delta_norm = delta / max(float(length - 1), 1.0)
    t_count_norm = t_count / max(float(length), 1.0)
    s_count_norm = s_count / max(float(spatial_k), 1.0)
    x_aug = np.concatenate(
        [x_masked, mask, delta_norm, x_last, x_mean_t, x_mean_s, x_closest, t_count_norm, s_count_norm],
        axis=-1,
    ).astype(np.float32)
    return np.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)


class MFilmWindowDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        graph_dists: Optional[np.ndarray],
        start: int,
        end: int,
        config: object,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.graph_dists = None if graph_dists is None else graph_dists.astype(np.float32)
        self.config = config
        self.neighbor_matrix = None
        self.sorted_neighbors = None
        if self.graph_dists is not None:
            num_nodes = self.graph_dists.shape[0]
            spatial_k = int(getattr(self.config, "spatial_k", 1))
            neighbor_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            sorted_neighbors = np.argsort(self.graph_dists, axis=1)
            for node in range(num_nodes):
                neighbor_ids = [idx for idx in sorted_neighbors[node] if idx != node]
                if spatial_k > 0:
                    neighbor_ids = neighbor_ids[: max(spatial_k, 1)]
                if neighbor_ids:
                    neighbor_matrix[node, neighbor_ids] = 1.0
            self.neighbor_matrix = neighbor_matrix
            self.sorted_neighbors = sorted_neighbors
        history = int(getattr(config, "history"))
        horizon = int(getattr(config, "horizon"))
        first_target_start = max(int(start), history)
        last_target_start = int(end) - horizon
        if last_target_start < first_target_start:
            self.target_starts: List[int] = []
        else:
            self.target_starts = list(range(first_target_start, last_target_start + 1))

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_start = self.target_starts[index]
        history = int(getattr(self.config, "history"))
        horizon = int(getattr(self.config, "horizon"))
        x_window = self.x_full[target_start - history : target_start]
        y_window = self.y_full[target_start : target_start + horizon]
        mask = _window_mask(
            history,
            x_window.shape[1],
            x_window.shape[2],
            self.config,
            int(getattr(self.config, "seed", 42)) + target_start,
            graph_dists=self.graph_dists,
        )
        x_aug = build_augmented_sequence(
            x_window,
            mask,
            graph_dists=self.graph_dists,
            spatial_k=int(getattr(self.config, "spatial_k", 1)),
            neighbor_matrix=self.neighbor_matrix,
            sorted_neighbors=self.sorted_neighbors,
        )
        return (
            torch.tensor(x_aug, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32),
        )


def build_gcnm_loaders(
    x_full: np.ndarray,
    y_full: np.ndarray,
    graph_dists: np.ndarray,
    split: Dict[str, int],
    batch_size: int,
    config: GCNMMissingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    train_ds = GCNMMissingWindowDataset(x_full, y_full, graph_dists, split["train_core_start"], split["train_core_end"], config)
    val_ds = None
    if split["val_end"] > split["val_start"]:
        val_ds = GCNMMissingWindowDataset(x_full, y_full, graph_dists, split["val_start"], split["val_end"], config)
    test_ds = GCNMMissingWindowDataset(x_full, y_full, graph_dists, split["test_start"], split["test_end"], config)
    sim_ds = GCNMMissingWindowDataset(x_full, y_full, graph_dists, split["simulation_start"], split["simulation_end"], config)
    if len(train_ds) == 0 or len(test_ds) == 0 or len(sim_ds) == 0:
        raise ValueError("Not enough time steps for GCN-M missing-data windows.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None if val_ds is None or len(val_ds) == 0 else DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, sim_loader


def build_sbu_loaders(
    x_full: np.ndarray,
    y_full: np.ndarray,
    graph_dists: Optional[np.ndarray],
    split: Dict[str, int],
    batch_size: int,
    config: GCNMMissingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    train_ds = SBUWindowDataset(x_full, y_full, graph_dists, split["train_core_start"], split["train_core_end"], config)
    val_ds = None
    if split["val_end"] > split["val_start"]:
        val_ds = SBUWindowDataset(x_full, y_full, graph_dists, split["val_start"], split["val_end"], config)
    test_ds = SBUWindowDataset(x_full, y_full, graph_dists, split["test_start"], split["test_end"], config)
    sim_ds = SBUWindowDataset(x_full, y_full, graph_dists, split["simulation_start"], split["simulation_end"], config)
    if len(train_ds) == 0 or len(test_ds) == 0 or len(sim_ds) == 0:
        raise ValueError("Not enough time steps for SBU-LSTM windows.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None if val_ds is None or len(val_ds) == 0 else DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, sim_loader


def build_gcrnode_loaders(
    x_full: np.ndarray,
    y_full: np.ndarray,
    graph_dists: Optional[np.ndarray],
    split: Dict[str, int],
    batch_size: int,
    config: GCRNODEConfig,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    train_ds = GCRNODEWindowDataset(x_full, y_full, graph_dists, split["train_core_start"], split["train_core_end"], config)
    val_ds = None
    if split["val_end"] > split["val_start"]:
        val_ds = GCRNODEWindowDataset(x_full, y_full, graph_dists, split["val_start"], split["val_end"], config)
    test_ds = GCRNODEWindowDataset(x_full, y_full, graph_dists, split["test_start"], split["test_end"], config)
    sim_ds = GCRNODEWindowDataset(x_full, y_full, graph_dists, split["simulation_start"], split["simulation_end"], config)
    if len(train_ds) == 0 or len(test_ds) == 0 or len(sim_ds) == 0:
        raise ValueError("Not enough time steps for GCRNODE windows.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None if val_ds is None or len(val_ds) == 0 else DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, sim_loader


def build_masked_loaders(
    x_full: np.ndarray,
    y_full: np.ndarray,
    graph_dists: Optional[np.ndarray],
    split: Dict[str, int],
    batch_size: int,
    config: object,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    train_ds = MaskedWindowDataset(x_full, y_full, graph_dists, split["train_core_start"], split["train_core_end"], config)
    val_ds = None
    if split["val_end"] > split["val_start"]:
        val_ds = MaskedWindowDataset(x_full, y_full, graph_dists, split["val_start"], split["val_end"], config)
    test_ds = MaskedWindowDataset(x_full, y_full, graph_dists, split["test_start"], split["test_end"], config)
    sim_ds = MaskedWindowDataset(x_full, y_full, graph_dists, split["simulation_start"], split["simulation_end"], config)
    if len(train_ds) == 0 or len(test_ds) == 0 or len(sim_ds) == 0:
        raise ValueError("Not enough time steps for masked-window scenario loaders.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None if val_ds is None or len(val_ds) == 0 else DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, sim_loader


def build_mfilm_loaders(
    x_full: np.ndarray,
    y_full: np.ndarray,
    graph_dists: Optional[np.ndarray],
    split: Dict[str, int],
    batch_size: int,
    config: object,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    train_ds = MFilmWindowDataset(x_full, y_full, graph_dists, split["train_core_start"], split["train_core_end"], config)
    val_ds = None
    if split["val_end"] > split["val_start"]:
        val_ds = MFilmWindowDataset(x_full, y_full, graph_dists, split["val_start"], split["val_end"], config)
    test_ds = MFilmWindowDataset(x_full, y_full, graph_dists, split["test_start"], split["test_end"], config)
    sim_ds = MFilmWindowDataset(x_full, y_full, graph_dists, split["simulation_start"], split["simulation_end"], config)
    if len(train_ds) == 0 or len(test_ds) == 0 or len(sim_ds) == 0:
        raise ValueError("Not enough time steps for MFilm windows.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None if val_ds is None or len(val_ds) == 0 else DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, sim_loader


class linear2d(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class linear3d(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv3d(c_in, c_out, kernel_size=(1, 1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MemoryModule(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp_q = linear2d(c_in, c_out)
        self.mlp_m = linear3d(c_in, c_out)
        self.mlp_c = linear3d(c_in, c_out)

    def forward(self, x_local: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        x_local = x_local.permute(0, 3, 1, 2).contiguous()
        x_hist = x_hist.permute(0, 4, 2, 1, 3).contiguous()
        q = self.mlp_q(x_local)
        q = torch.unsqueeze(q, dim=-1)
        m = self.mlp_m(x_hist)
        c = self.mlp_c(x_hist)
        m = m.permute(0, 1, 3, 2, 4).contiguous()
        c = c.permute(0, 1, 3, 2, 4).contiguous()
        mq = torch.matmul(m, q).squeeze(-1)
        q = q.squeeze(-1)
        attention_scores = F.softmax(F.relu(mq), dim=-1)
        o = torch.einsum("bflt,bfltd->bfld", attention_scores, c).contiguous()
        return q + o


class FilterLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, filter_square_matrix: torch.Tensor, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("filter_square_matrix", filter_square_matrix.clone().detach())
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)


class LocalFeatureModule(nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        identity = torch.eye(num_nodes)
        self.register_buffer("identity", identity)
        self.gamma_x_t = FilterLinear(num_nodes, num_nodes, identity)
        self.gamma_x_s = FilterLinear(num_nodes, num_nodes, identity)

    def step(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        x_last_obsv: torch.Tensor,
        x_mean_t: torch.Tensor,
        delta_t: torch.Tensor,
        x_closest_obsv: torch.Tensor,
        x_mean_s: torch.Tensor,
        delta_s: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, num_features = x.shape
        zeros = torch.zeros(batch_size * num_features, num_nodes, device=x.device, dtype=x.dtype)

        def flatten(v: torch.Tensor) -> torch.Tensor:
            return v.permute(0, 2, 1).reshape(batch_size * num_features, num_nodes)

        def unflatten(v: torch.Tensor) -> torch.Tensor:
            return v.reshape(batch_size, num_features, num_nodes).permute(0, 2, 1).contiguous()

        delta_t_flat = flatten(delta_t)
        delta_s_flat = flatten(delta_s)
        gama_t = torch.exp(-torch.max(zeros, self.gamma_x_t(delta_t_flat)))
        gama_s = torch.exp(-torch.max(zeros, self.gamma_x_s(delta_s_flat)))

        x_flat = flatten(x)
        mask_flat = flatten(mask)
        x_last_flat = flatten(x_last_obsv)
        x_mean_t_flat = flatten(x_mean_t)
        x_close_flat = flatten(x_closest_obsv)
        x_mean_s_flat = flatten(x_mean_s)

        out = mask_flat * x_flat + (1.0 - mask_flat) * (
            gama_t * x_last_flat
            + (1.0 - gama_t) * x_mean_t_flat
            + gama_s * x_close_flat
            + (1.0 - gama_s) * x_mean_s_flat
        )
        return unflatten(out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input[:, 0]
        mask = input[:, 1]
        x_last = input[:, 2]
        x_mean_t = input[:, 3]
        delta_t = input[:, 4]
        x_closest = input[:, 5]
        x_mean_s = input[:, 6]
        delta_s = input[:, 7]

        outputs = []
        for step in range(input.shape[2]):
            outputs.append(
                self.step(
                    x[:, step],
                    mask[:, step],
                    x_last[:, step],
                    x_mean_t[:, step],
                    delta_t[:, step],
                    x_closest[:, step],
                    x_mean_s[:, step],
                    delta_s[:, step],
                ).unsqueeze(1)
            )
        return torch.cat(outputs, dim=1)


class nconv_gwnet(nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", x, a).contiguous()


class nconv_gcnm_dynamic(nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,nvwl->ncwl", x, a).contiguous()


class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class gcn_gwnet(nn.Module):
    def __init__(self, c_in: int = 32, c_out: int = 32, dropout: float = 0.3, support_len: int = 3, order: int = 2):
        super().__init__()
        self.nconv = nconv_gwnet()
        self.order = order
        self.dropout = dropout
        self.mlp = linear((order * support_len + 1) * c_in, c_out)

    def forward(self, x: torch.Tensor, support: List[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class gcn_gcnm_dynamic(nn.Module):
    def __init__(self, c_in: int = 32, c_out: int = 32, dropout: float = 0.3, support_len: int = 3, order: int = 2):
        super().__init__()
        self.nconv_dynamic = nconv_gcnm_dynamic()
        self.nconv_static = nconv_gwnet()
        self.order = order
        self.dropout = dropout
        self.mlp = linear((order * support_len + 1) * c_in, c_out)

    def forward(self, x: torch.Tensor, support: List[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in support[:2]:
            x1 = self.nconv_dynamic(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv_dynamic(x1, a)
                out.append(x1)
        if len(support) > 2:
            for a in support[2:]:
                x1 = self.nconv_static(x, a)
                out.append(x1)
                for _ in range(2, self.order + 1):
                    x1 = self.nconv_static(x1, a)
                    out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GCNMDynamicMulti(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        in_dim: int,
        out_dim: int,
        horizon: int,
        dropout: float = 0.3,
        supports: Optional[List[torch.Tensor]] = None,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.horizon = horizon
        self.out_dim = out_dim
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.supports = supports or []
        self.local_feature_model = LocalFeatureModule(num_nodes)
        self.memory_model = MemoryModule(in_dim, residual_channels)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.supports_len = 2
        node_dim = residual_channels
        self.alpha = 3
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.idx = torch.arange(self.num_nodes, device=self.device)

        self.GCN1_1 = gcn_gwnet(c_in=residual_channels, c_out=residual_channels, dropout=self.dropout, support_len=1)
        self.GCN2_1 = gcn_gwnet(c_in=residual_channels, c_out=residual_channels, dropout=self.dropout, support_len=1)

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    gcn_gcnm_dynamic(
                        c_in=dilation_channels,
                        c_out=residual_channels,
                        dropout=dropout,
                        support_len=self.supports_len,
                    )
                )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, horizon * out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field
        if horizon > self.receptive_field:
            self.skip0 = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, horizon), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, horizon - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1), bias=True)

    def preprocessing(self, adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.num_nodes, device=self.device, dtype=adj.dtype)
        adj = adj + eye
        return adj / torch.unsqueeze(adj.sum(-1).clamp_min(1e-6), -1)

    def forward(self, input: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        z = self.local_feature_model(input)
        x_hist = x_hist.transpose(1, 2).contiguous()
        e = self.memory_model(z, x_hist)
        x = e.permute(0, 1, 3, 2).contiguous()

        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        skip = self.skip0(x)

        nodevec_init_1 = self.emb1(self.idx)
        nodevec_init_2 = self.emb2(self.idx)

        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            filter1 = self.GCN1_1(x, [self.supports[0]]).permute(0, 3, 2, 1).contiguous()
            filter2 = self.GCN2_1(x, [self.supports[1]]).permute(0, 3, 2, 1).contiguous()
            nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec_init_1, filter1))
            nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec_init_2, filter2))

            a = torch.matmul(nodevec1, nodevec2.transpose(2, 3)) - torch.matmul(nodevec2, nodevec1.transpose(2, 3))
            adj = F.relu(torch.tanh(self.alpha * a))
            mask = torch.zeros_like(adj)
            s1, t1 = adj.topk(min(20, adj.size(-1)), -1)
            mask.scatter_(-1, t1, s1.fill_(1))
            adj = adj * mask

            adp = self.preprocessing(adj).permute(0, 2, 3, 1).contiguous()
            adp_t = self.preprocessing(adj.transpose(2, 3)).permute(0, 2, 3, 1).contiguous()
            new_supports = [adp, adp_t]

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except Exception:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).squeeze(-1)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.horizon, self.out_dim, self.num_nodes).permute(0, 1, 3, 2).contiguous()
        return x


class LSTMICell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.imp_from_c = nn.Linear(hidden_dim, input_dim, bias=False)
        self.imp_from_h = nn.Linear(hidden_dim, input_dim, bias=True)

        self.w_f = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_f = nn.Linear(input_dim, hidden_dim, bias=True)

        self.w_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_i = nn.Linear(input_dim, hidden_dim, bias=True)

        self.w_o = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_o = nn.Linear(input_dim, hidden_dim, bias=True)

        self.w_c = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_c = nn.Linear(input_dim, hidden_dim, bias=True)

    def forward(
        self,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_pred = torch.tanh(self.imp_from_c(c_prev) + self.imp_from_h(h_prev))
        x_completed = m_t * x_t + (1.0 - m_t) * x_pred

        f_t = torch.sigmoid(self.w_f(x_completed) + self.u_f(h_prev) + self.v_f(m_t))
        i_t = torch.sigmoid(self.w_i(x_completed) + self.u_i(h_prev) + self.v_i(m_t))
        o_t = torch.sigmoid(self.w_o(x_completed) + self.u_o(h_prev) + self.v_o(m_t))
        c_bar = torch.tanh(self.w_c(x_completed) + self.u_c(h_prev) + self.v_c(m_t))
        c_t = f_t * c_prev + i_t * c_bar
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, x_pred, x_completed


class BDLSTMILayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forward_cell = LSTMICell(input_dim, hidden_dim)
        self.backward_cell = LSTMICell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        h_f = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)
        c_f = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)
        h_b = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)
        c_b = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)

        outs_f: List[torch.Tensor] = []
        preds_f: List[torch.Tensor] = []
        for t in range(seq_len):
            h_f, c_f, x_pred, _ = self.forward_cell(x[:, t], mask[:, t], h_f, c_f)
            outs_f.append(h_f.unsqueeze(1))
            preds_f.append(x_pred.unsqueeze(1))

        outs_b_rev: List[torch.Tensor] = []
        preds_b_rev: List[torch.Tensor] = []
        for step in range(seq_len - 1, -1, -1):
            h_b, c_b, x_pred, _ = self.backward_cell(x[:, step], mask[:, step], h_b, c_b)
            outs_b_rev.append(h_b.unsqueeze(1))
            preds_b_rev.append(x_pred.unsqueeze(1))

        outs_b = list(reversed(outs_b_rev))
        preds_b = list(reversed(preds_b_rev))
        h_f_seq = torch.cat(outs_f, dim=1)
        h_b_seq = torch.cat(outs_b, dim=1)
        x_pred_f = torch.cat(preds_f, dim=1)
        x_pred_b = torch.cat(preds_b, dim=1)
        y = 0.5 * (h_f_seq + h_b_seq)
        return y, x_pred_f, x_pred_b


class BDLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forward_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.backward_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_f, _ = self.forward_lstm(x)
        x_rev = torch.flip(x, dims=[1])
        y_b_rev, _ = self.backward_lstm(x_rev)
        y_b = torch.flip(y_b_rev, dims=[1])
        return 0.5 * (y_f + y_b)


class SBULSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, horizon: int, reg_lambda: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.reg_lambda = reg_lambda
        self.first = BDLSTMILayer(input_dim=input_dim, hidden_dim=hidden_dim)
        self.second = BDLSTMLayer(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon * out_dim)
        self.last_reg_loss: Optional[torch.Tensor] = None

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor, x_true: Optional[torch.Tensor] = None) -> torch.Tensor:
        y1, x_pred_f, x_pred_b = self.first(x_masked, mask)
        y2 = self.second(y1)
        final_state = y2[:, -1]
        pred = self.head(final_state).view(x_masked.shape[0], self.horizon, self.out_dim)

        if x_true is not None:
            observed = mask
            observed_count = observed.sum().clamp_min(1.0)
            reg_f = torch.abs((x_pred_f - x_true) * observed).sum() / observed_count
            reg_b = torch.abs((x_pred_b - x_true) * observed).sum() / observed_count
            self.last_reg_loss = 0.5 * (reg_f + reg_b)
        else:
            self.last_reg_loss = None
        return pred


class STMemoryGraph(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        replay_day_size: int,
        replay_week_size: int,
        topk: int,
        graph_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.replay_day_size = replay_day_size
        self.replay_week_size = replay_week_size
        self.topk = topk
        self.replay_day = nn.Parameter(torch.randn(replay_day_size, memory_dim) * 0.02)
        self.replay_week = nn.Parameter(torch.randn(replay_week_size, memory_dim) * 0.02)
        self.pattern = nn.Parameter(torch.randn(num_nodes, memory_dim) * 0.02)
        self.query_proj = nn.Linear(2 * memory_dim, memory_dim)
        if graph_mask is None:
            self.register_buffer("graph_mask", torch.ones(num_nodes, num_nodes))
        else:
            self.register_buffer("graph_mask", graph_mask.float())

    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        bsz, steps = time_idx.shape
        day_idx = torch.remainder(time_idx, self.replay_day_size)
        week_idx = torch.remainder(time_idx, self.replay_week_size)
        e_day = self.replay_day[day_idx]  # [B,T,m]
        e_week = self.replay_week[week_idx]  # [B,T,m]
        e_rem = torch.cat([e_day, e_week], dim=-1)
        q_rem = self.query_proj(e_rem).unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # [B,T,N,m]
        e_pm = self.pattern.unsqueeze(0).unsqueeze(0).expand(bsz, steps, -1, -1)  # [B,T,N,m]
        scores = torch.einsum("btim,btjm->btij", q_rem, e_pm)
        a_tilde = torch.softmax(scores, dim=-1)
        if self.topk < self.num_nodes:
            topk_vals, topk_idx = torch.topk(a_tilde, k=self.topk, dim=-1)
            sparse_mask = torch.zeros_like(a_tilde)
            sparse_mask.scatter_(-1, topk_idx, 1.0)
            a_tilde = a_tilde * sparse_mask
        graph_mask = self.graph_mask.unsqueeze(0).unsqueeze(0)
        a = a_tilde * graph_mask
        denom = a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return a / denom


class DynamicGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.depth_weights = nn.Parameter(torch.ones(depth + 1))
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], a: [B,N,N]
        states = [x]
        h_prev = x
        for _ in range(self.depth):
            h_prev = self.alpha * x + self.beta * torch.einsum("bij,bjc->bic", a, h_prev)
            states.append(h_prev)
        stacked = torch.stack(states, dim=0)
        weights = torch.softmax(self.depth_weights, dim=0).view(-1, 1, 1, 1)
        out = (weights * stacked).sum(dim=0)
        return self.out_proj(out)


class GCRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gc_z = DynamicGraphConv(input_dim + hidden_dim, hidden_dim, depth)
        self.gc_r = DynamicGraphConv(input_dim + hidden_dim, hidden_dim, depth)
        self.gc_g = DynamicGraphConv(input_dim + hidden_dim, hidden_dim, depth)

    def gate_values(self, x_t: torch.Tensor, h_prev: torch.Tensor, a_t: torch.Tensor):
        xh = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.gc_z(xh, a_t))
        r = torch.sigmoid(self.gc_r(xh, a_t))
        xrh = torch.cat([x_t, r * h_prev], dim=-1)
        g = torch.tanh(self.gc_g(xrh, a_t))
        return z, r, g

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, a_t: torch.Tensor):
        z, _, g = self.gate_values(x_t, h_prev, a_t)
        return z * h_prev + (1.0 - z) * g


class GCRUODEFunc(nn.Module):
    def __init__(self, cell: GCRUCell):
        super().__init__()
        self.cell = cell

    def forward(self, h_t: torch.Tensor, x_t: torch.Tensor, a_t: torch.Tensor):
        z, _, g = self.cell.gate_values(x_t, h_t, a_t)
        return (1.0 - z) * (g - h_t)


def _diag_gaussian_kl(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    return 0.5 * (
        logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0
    ).sum(dim=-1)


class GCRNODEModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        horizon: int,
        stmg: STMemoryGraph,
        depth: int,
        ode_steps: int,
        ode_solver: str,
        kl_weight: float,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.stmg = stmg
        self.ode_steps = max(int(ode_steps), 1)
        self.ode_solver = ode_solver.lower()
        self.kl_weight = kl_weight

        self.encoder_in = nn.Linear(input_dim, hidden_dim)
        self.cell = GCRUCell(hidden_dim, hidden_dim, depth)
        self.ode_func = GCRUODEFunc(self.cell)
        self.readout = nn.Linear(hidden_dim, output_dim)

        self.prior_mu = nn.Linear(hidden_dim, hidden_dim)
        self.prior_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.obs_mu = nn.Linear(hidden_dim, hidden_dim)
        self.obs_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.post_mu = nn.Linear(hidden_dim, hidden_dim)
        self.post_logvar = nn.Linear(hidden_dim, hidden_dim)

        self.last_aux_loss: Optional[torch.Tensor] = None

    def _ode_step(self, h: torch.Tensor, x: torch.Tensor, a: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        if self.ode_solver == "rk4":
            k1 = self.ode_func(h, x, a)
            k2 = self.ode_func(h + 0.5 * dt * k1, x, a)
            k3 = self.ode_func(h + 0.5 * dt * k2, x, a)
            k4 = self.ode_func(h + dt * k3, x, a)
            return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h + dt * self.ode_func(h, x, a)

    def _ode_solve(self, h: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / float(self.ode_steps)
        h_curr = h
        for _ in range(self.ode_steps):
            h_curr = self._ode_step(h_curr, x, a, dt=dt)
        return h_curr

    def _bayes_alignment(
        self,
        h_pred: torch.Tensor,
        h_post: torch.Tensor,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        mu_pre = self.prior_mu(h_pred)
        logvar_pre = self.prior_logvar(h_pred).clamp(-5.0, 5.0)
        x_embed = self.encoder_in(x_obs)
        mu_obs = self.obs_mu(x_embed)
        logvar_obs = self.obs_logvar(x_embed).clamp(-5.0, 5.0)
        var_pre = torch.exp(logvar_pre)
        var_obs = torch.exp(logvar_obs)
        var_bayes = 1.0 / (1.0 / var_pre.clamp_min(1e-6) + 1.0 / var_obs.clamp_min(1e-6))
        mu_bayes = var_bayes * (mu_pre / var_pre.clamp_min(1e-6) + mu_obs / var_obs.clamp_min(1e-6))
        logvar_bayes = torch.log(var_bayes.clamp_min(1e-6))

        mu_post = self.post_mu(h_post)
        logvar_post = self.post_logvar(h_post).clamp(-5.0, 5.0)
        kl = _diag_gaussian_kl(mu_bayes, logvar_bayes, mu_post, logvar_post)
        kl = kl * observed_mask
        denom = observed_mask.sum().clamp_min(1.0)
        return kl.sum() / denom

    def forward(
        self,
        x_masked: torch.Tensor,
        mask: torch.Tensor,
        hist_idx: torch.Tensor,
        fut_idx: torch.Tensor,
    ) -> torch.Tensor:
        # x_masked [B,T,N,F], mask [B,T,N,F]
        hist_graphs = self.stmg(hist_idx)  # [B,T,N,N]
        fut_graphs = self.stmg(fut_idx)  # [B,H,N,N]
        x_enc = self.encoder_in(x_masked)
        h = torch.zeros(x_masked.size(0), self.num_nodes, self.hidden_dim, device=x_masked.device, dtype=x_masked.dtype)
        aux_losses: List[torch.Tensor] = []
        observed_node_mask = (mask.sum(dim=-1) > 0).float()  # [B,T,N]

        for t in range(x_masked.size(1)):
            a_t = hist_graphs[:, t]
            h_pred = self._ode_solve(h, x_enc[:, t], a_t)
            h_upd = self.cell(x_enc[:, t], h_pred, a_t)
            obs = observed_node_mask[:, t].unsqueeze(-1)
            h = obs * h_upd + (1.0 - obs) * h_pred
            aux_losses.append(self._bayes_alignment(h_pred, h_upd, x_masked[:, t], observed_node_mask[:, t]))

        preds = []
        h_dec = h
        zero_x = torch.zeros_like(x_enc[:, 0])
        for t in range(self.horizon):
            a_t = fut_graphs[:, t]
            h_dec = self._ode_solve(h_dec, zero_x, a_t)
            preds.append(self.readout(h_dec).unsqueeze(1))
        self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return torch.cat(preds, dim=1)
