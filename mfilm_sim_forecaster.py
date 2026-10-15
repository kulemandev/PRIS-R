from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gcnm_missing_adapter import compute_graph_distance_matrix


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        stdev = x.std(dim=1, keepdim=True, unbiased=False)
        stdev = torch.clamp(stdev, min=self.eps)
        x_norm = (x - mean) / stdev
        return x_norm, mean, stdev

    def denormalize(
        self,
        y: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor,
        feature_indices: Tuple[int, ...],
    ) -> torch.Tensor:
        idx = torch.as_tensor(feature_indices, device=y.device, dtype=torch.long)
        mean_sel = torch.index_select(mean, dim=-1, index=idx).squeeze(1).unsqueeze(1)
        stdev_sel = torch.index_select(stdev, dim=-1, index=idx).squeeze(1).unsqueeze(1)
        return y * stdev_sel + mean_sel


class _SafeTensor:
    @staticmethod
    def clean(x: torch.Tensor, clamp_value: float = 10.0) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(-clamp_value, clamp_value)


class _CondFiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )

    def gamma_beta(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        return gamma, beta

    @staticmethod
    def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class _FiLMLinearSTAttention(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.film_qkv = _CondFiLM(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mode: str) -> torch.Tensor:
        batch_size, steps, nodes, channels = x.shape
        if mode == "spatial":
            xt = x.reshape(batch_size * steps, nodes, channels)
            cond_bt = cond.unsqueeze(1).expand(batch_size, steps, nodes, cond.size(-1)).reshape(batch_size * steps, nodes, -1)
            q0 = self.wq(xt)
            k0 = self.wk(xt)
            v0 = self.wv(xt)
            gamma, beta = self.film_qkv.gamma_beta(cond_bt)
            q = _CondFiLM.apply_film(q0, gamma, beta) / (nodes ** 0.5)
            k = _CondFiLM.apply_film(k0, gamma, beta) / (nodes ** 0.5)
            v = _CondFiLM.apply_film(v0, gamma, beta)
            out = (q @ (k.transpose(1, 2) @ v)).reshape(batch_size, steps, nodes, channels)
            return self.drop(out)
        if mode == "temporal":
            xn = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * nodes, steps, channels)
            cond_bn = cond.reshape(batch_size * nodes, -1)
            q0 = self.wq(xn)
            k0 = self.wk(xn)
            v0 = self.wv(xn)
            gamma, beta = self.film_qkv.gamma_beta(cond_bn)
            q = _CondFiLM.apply_film(q0, gamma, beta) / (steps ** 0.5)
            k = _CondFiLM.apply_film(k0, gamma, beta) / (steps ** 0.5)
            v = _CondFiLM.apply_film(v0, gamma, beta)
            out = (q @ (k.transpose(1, 2) @ v)).reshape(batch_size, nodes, steps, channels).permute(0, 2, 1, 3).contiguous()
            return self.drop(out)
        raise ValueError("mode must be 'spatial' or 'temporal'")


class _STGMFiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.film_x = _CondFiLM(cond_dim, dim)
        self.att = _FiLMLinearSTAttention(dim, cond_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_x.gamma_beta(cond)
        x = _CondFiLM.apply_film(x, gamma, beta)
        attn = self.att(x, cond, "spatial") + self.att(x, cond, "temporal")
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ff(x))
        return x


class STGMFiLMFormerCompat(nn.Module):
    """Checkpoint-compatible MFilmFormer used by the simulation notebook."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        history: int,
        ces_index: int,
        traffic_indices: Tuple[int, ...],
        dim: int = 64,
        k_hops: int = 3,
        dropout: float = 0.1,
        cond_dim: int = 4,
        recon_lambda: float = 0.05,
    ):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.history = history
        self.in_dim = in_dim
        self.ces_index = ces_index
        self.traffic_indices = tuple(int(i) for i in traffic_indices)
        self.target_input_indices = tuple(self.traffic_indices) + (int(self.ces_index),)
        self.dim = dim
        self.k_hops = k_hops
        self.recon_lambda = recon_lambda

        self.revin = _RevIN(in_dim)
        self.x_proj = nn.Linear(in_dim, dim)
        self.repair_weight_proj = nn.Linear(in_dim * 5, in_dim * 3)
        self.miss_proj = nn.Linear(in_dim * 5, dim)
        self.miss_gate = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.Sigmoid(),
        )
        self.time_embed = nn.Parameter(torch.zeros(1, history, 1, dim))
        nn.init.normal_(self.time_embed, std=0.02)

        self.block = _STGMFiLMBlock(dim, cond_dim=cond_dim, dropout=dropout)
        self.gate = nn.Linear(dim, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon * out_dim))
        self.recon_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, in_dim),
        )
        self.last_reg_loss: Optional[torch.Tensor] = None

    def _split_augmented_input(
        self, x_aug: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = self.in_dim * 6
        if x_aug.size(-1) != expected:
            raise ValueError(
                f"STG_MFilmFormer expects augmented inputs with {expected} channels, got {x_aug.size(-1)}."
            )
        return torch.split(x_aug, self.in_dim, dim=-1)

    def _build_condition(self, x_signal: torch.Tensor, obs_mask: torch.Tensor, delta_norm: torch.Tensor) -> torch.Tensor:
        ces = x_signal[..., self.ces_index].amax(dim=1)
        if self.traffic_indices:
            traffic = x_signal[..., list(self.traffic_indices)].mean(dim=(1, 3))
        else:
            traffic = torch.zeros_like(ces)
        miss_ratio = 1.0 - obs_mask.mean(dim=(1, 3))
        gap_ratio = delta_norm.amax(dim=1).mean(dim=-1)
        return torch.stack([ces, traffic, miss_ratio, gap_ratio], dim=-1)

    def _graph_propagate(self, x: torch.Tensor, a_hat: torch.Tensor) -> list[torch.Tensor]:
        outs = [x]
        xk = x
        for _ in range(self.k_hops):
            xk = torch.matmul(a_hat, xk)
            outs.append(xk)
        return outs

    def forward(self, x_aug: torch.Tensor, a_hat: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del obs_mask
        batch_size, steps, nodes, _ = x_aug.shape
        x_hist, obs_mask, delta_norm, x_last, x_mean_t, x_mean_s = self._split_augmented_input(x_aug)
        x_base = x_hist * obs_mask + x_last * (1.0 - obs_mask)
        x_base_norm, revin_mean, revin_std = self.revin.normalize(x_base)
        x_hist_norm = _SafeTensor.clean((x_hist - revin_mean) / revin_std)
        x_last_norm = _SafeTensor.clean((x_last - revin_mean) / revin_std)
        x_mean_t_norm = _SafeTensor.clean((x_mean_t - revin_mean) / revin_std)
        x_mean_s_norm = _SafeTensor.clean((x_mean_s - revin_mean) / revin_std)
        x_base_norm = _SafeTensor.clean(x_base_norm)

        repair_context = torch.cat(
            [1.0 - obs_mask, delta_norm, x_last_norm, x_mean_t_norm, x_mean_s_norm],
            dim=-1,
        )
        repair_context = _SafeTensor.clean(repair_context)
        repair_logits = self.repair_weight_proj(repair_context).view(batch_size, steps, nodes, self.in_dim, 3)
        repair_logits = torch.nan_to_num(repair_logits, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        repair_weights = torch.softmax(repair_logits, dim=-1)
        repair_candidates = torch.stack([x_last_norm, x_mean_t_norm, x_mean_s_norm], dim=-1)
        x_fill_norm = (repair_weights * repair_candidates).sum(dim=-1)
        x_imputed_norm = x_hist_norm * obs_mask + x_fill_norm * (1.0 - obs_mask)
        x_imputed_norm = _SafeTensor.clean(x_imputed_norm)

        cond = _SafeTensor.clean(self._build_condition(x_imputed_norm, obs_mask, delta_norm))
        miss_ratio = 1.0 - obs_mask.mean(dim=(1, 3), keepdim=True)
        gap_ratio = delta_norm.amax(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        missing_strength = torch.clamp(miss_ratio + gap_ratio, min=0.0, max=1.0)

        x_main = self.x_proj(x_imputed_norm)
        if float(missing_strength.max().detach().cpu()) > 0.0:
            x_missing = self.miss_proj(repair_context)
            miss_gate = self.miss_gate(cond).unsqueeze(1) * missing_strength
            x = x_main + miss_gate * x_missing + self.time_embed[:, :steps]
        else:
            x = x_main + self.time_embed[:, :steps]

        props = self._graph_propagate(x, a_hat)
        h = self.block(props[0], cond)
        for prop in props[1:]:
            hk = self.block(prop, cond)
            gate = torch.sigmoid(self.gate(h))
            h = h + gate * hk

        recon = self.recon_head(h)
        observed_denom = torch.clamp(obs_mask.sum(), min=1.0)
        if float(missing_strength.max().detach().cpu()) > 0.0:
            recon_err = (recon - x_hist) ** 2
            self.last_reg_loss = ((recon_err * obs_mask).sum() / observed_denom).detach()
        else:
            self.last_reg_loss = torch.zeros((), device=x_aug.device, dtype=x_aug.dtype)

        h_last = h[:, -1]
        y = self.head(h_last).view(batch_size, nodes, self.horizon, self.out_dim)
        y = y.permute(0, 2, 1, 3).contiguous()
        y = self.revin.denormalize(y, revin_mean, revin_std, self.target_input_indices)
        return y


def build_ci_list_and_ahat(cell_gnb_path: Path, device: torch.device) -> Tuple[list[int], torch.Tensor]:
    cell_df = pd.read_csv(cell_gnb_path)
    cell_df.columns = [c.strip() for c in cell_df.columns]
    if not {"CI", "gnb"}.issubset(cell_df.columns):
        raise ValueError("cell_gnb_relation.csv must contain 'CI' and 'gnb' columns.")
    cell_df["CI"] = pd.to_numeric(cell_df["CI"], errors="coerce")
    cell_df["gnb"] = pd.to_numeric(cell_df["gnb"], errors="coerce")
    cell_df = cell_df.dropna(subset=["CI", "gnb"]).copy()
    cell_df["CI"] = cell_df["CI"].astype(int)
    cell_df["gnb"] = cell_df["gnb"].astype(int)

    by_gnb = cell_df.groupby("gnb")["CI"].apply(list).to_dict()
    gnb_ids = sorted(by_gnb.keys())
    ci_list = sorted(cell_df["CI"].unique().tolist())
    ci_to_idx = {ci: i for i, ci in enumerate(ci_list)}
    n = len(ci_list)
    a = np.zeros((n, n), dtype=np.float32)

    def add_undirected(ci_u: int, ci_v: int) -> None:
        if ci_u in ci_to_idx and ci_v in ci_to_idx:
            u = ci_to_idx[ci_u]
            v = ci_to_idx[ci_v]
            a[u, v] = 1.0
            a[v, u] = 1.0

    for _, cells in by_gnb.items():
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                add_undirected(cells[i], cells[j])

    for g in gnb_ids:
        if (g + 1) in by_gnb:
            for c1 in by_gnb[g]:
                for c2 in by_gnb[g + 1]:
                    add_undirected(c1, c2)

    a = a + np.eye(n, dtype=np.float32)
    d = np.sum(a, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=(d > 0))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    a_hat = (d_inv_sqrt[:, None] * a) * d_inv_sqrt[None, :]
    return ci_list, torch.tensor(a_hat, dtype=torch.float32, device=device)


class STGMFiLMForecaster:
    def __init__(self, ckpt_path: Path, dataset_csv: Path, cell_gnb_path: Path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.history = int(ckpt.get("history", ckpt.get("config", {}).get("HISTORY", 16)))
        self.horizon = int(ckpt.get("horizon", ckpt.get("config", {}).get("HORIZON", 4)))
        self.feat_cols = list(ckpt.get("feature_cols", ckpt.get("config", {}).get("FEAT_COLS", [])))
        self.target_cols = list(ckpt.get("target_cols", ckpt.get("config", {}).get("TARGET_COLS", [])))
        self.use_log1p = bool(ckpt.get("use_log1p", ckpt.get("config", {}).get("USE_LOG1P", False)))
        self.use_zscore = bool(ckpt.get("use_zscore", ckpt.get("config", {}).get("USE_ZSCORE", False)))
        if not self.feat_cols or not self.target_cols:
            raise ValueError("Checkpoint is missing feature or target column metadata.")

        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        hidden_dim = int(state_dict["x_proj.weight"].shape[0])

        df = pd.read_csv(dataset_csv)
        df.columns = [c.strip() for c in df.columns]
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["CI"] = pd.to_numeric(df["CI"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["time_s", "CI"]).copy()
        df["CI"] = df["CI"].astype(int)
        df = df.sort_values(["time_s", "CI"], kind="stable").reset_index(drop=True)

        self.ci_list, self.A_hat = build_ci_list_and_ahat(cell_gnb_path, self.device)
        self.graph_dists = compute_graph_distance_matrix(self.A_hat.detach().cpu().numpy())
        self.adj_binary = (self.A_hat.detach().cpu().numpy() > 0).astype(np.float32)
        np.fill_diagonal(self.adj_binary, 0.0)
        df = df[df["CI"].isin(self.ci_list)].copy()

        time_vals = np.array(sorted(df["time_s"].unique()))
        self.time_vals = time_vals
        self.time_to_idx = {float(t): i for i, t in enumerate(time_vals)}

        def pivot(col: str) -> np.ndarray:
            p = df.pivot(index="time_s", columns="CI", values=col).reindex(index=time_vals, columns=self.ci_list)
            return p.to_numpy()

        x_np = np.stack([pivot(c) for c in self.feat_cols], axis=-1).astype(np.float32)
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_log1p:
            log_idxs = [i for i, c in enumerate(self.feat_cols) if c not in {"avg_speed_kmh"}]
            x_np[..., log_idxs] = np.log1p(np.maximum(x_np[..., log_idxs], 0.0))
        self.X = x_np

        idx_ces = self.feat_cols.index("CES_count")
        traffic_indices = tuple(
            self.feat_cols.index(col)
            for col in ["critical_DL_sum", "critical_UL_sum", "perform_DL_sum", "perform_UL_sum"]
            if col in self.feat_cols
        )

        self.model = STGMFiLMFormerCompat(
            in_dim=len(self.feat_cols),
            out_dim=len(self.target_cols),
            horizon=self.horizon,
            history=self.history,
            ces_index=idx_ces,
            traffic_indices=traffic_indices,
            dim=hidden_dim,
            k_hops=3,
            dropout=0.1,
            recon_lambda=0.05,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _make_augmented_window(self, t_idx: int) -> torch.Tensor:
        x_hist = self.X[t_idx - self.history + 1 : t_idx + 1]
        mask = np.ones_like(x_hist, dtype=np.float32)
        x_last = x_hist.copy()
        x_mean_t = np.repeat(x_hist.mean(axis=0, keepdims=True), x_hist.shape[0], axis=0)
        if self.adj_binary.sum() > 0:
            denom = np.maximum(self.adj_binary.sum(axis=1, keepdims=True), 1.0)
            x_mean_s = np.einsum("nm,tmf->tnf", self.adj_binary, x_hist) / denom[None, :, :]
        else:
            x_mean_s = np.zeros_like(x_hist, dtype=np.float32)
        delta_norm = np.zeros_like(x_hist, dtype=np.float32)
        x_aug = np.concatenate([x_hist, mask, delta_norm, x_last, x_mean_t, x_mean_s], axis=-1).astype(np.float32)
        x_aug = np.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(x_aug, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def predict_tplus1_totals(self, time_s: float) -> Dict[str, float]:
        t_idx = self.time_to_idx.get(float(time_s), None)
        if t_idx is None or t_idx < self.history - 1:
            return {k: 0.0 for k in self.target_cols}
        pred = self.model(self._make_augmented_window(t_idx), self.A_hat)
        pred_t1 = torch.clamp(pred[0, 0], min=0.0)
        if self.use_log1p:
            pred_t1 = torch.clamp(torch.expm1(pred_t1), min=0.0)
        totals = pred_t1.detach().cpu().numpy().sum(axis=0)
        return {self.target_cols[i]: float(totals[i]) for i in range(len(self.target_cols))}

    @torch.no_grad()
    def predict_tplus1_per_cell(self, time_s: float) -> Dict[int, Dict[str, float]]:
        t_idx = self.time_to_idx.get(float(time_s), None)
        if t_idx is None or t_idx < self.history - 1:
            return {}
        pred = self.model(self._make_augmented_window(t_idx), self.A_hat)
        pred_t1 = torch.clamp(pred[0, 0], min=0.0)
        if self.use_log1p:
            pred_t1 = torch.clamp(torch.expm1(pred_t1), min=0.0)
        arr = pred_t1.detach().cpu().numpy()
        out: Dict[int, Dict[str, float]] = {}
        for i, ci in enumerate(self.ci_list):
            out[int(ci)] = {self.target_cols[j]: float(arr[i, j]) for j in range(len(self.target_cols))}
        return out
