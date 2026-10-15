from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


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


def _safe_feature_tensor(x: torch.Tensor, clamp_value: float = 10.0) -> torch.Tensor:
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


class STGMFiLMFormer(nn.Module):
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
        num_blocks: int = 1,
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
        self.num_blocks = int(num_blocks)

        self.revin = _RevIN(in_dim)
        # Main forecasting path sees only the observed traffic/history signal.
        self.x_proj = nn.Linear(in_dim, dim)
        # Missingness stays in a separate conditional branch.
        self.repair_weight_proj = nn.Linear(in_dim * 8, in_dim * 4)
        self.miss_proj = nn.Linear(in_dim * 8, dim)
        self.miss_gate = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.Sigmoid(),
        )
        self.time_embed = nn.Parameter(torch.zeros(1, history, 1, dim))
        nn.init.normal_(self.time_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_STGMFiLMBlock(dim, cond_dim=cond_dim, dropout=dropout) for _ in range(self.num_blocks)]
        )
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon * out_dim))
        self.recon_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, in_dim),
        )
        self.last_reg_loss: Optional[torch.Tensor] = None

    def _split_augmented_input(
        self, x_aug: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = self.in_dim * 9
        if x_aug.size(-1) != expected:
            raise ValueError(
                f"STG_MFilmFormer expects augmented inputs with {expected} channels, "
                f"got {x_aug.size(-1)}."
            )
        x_hist, obs_mask, delta_norm, x_last, x_mean_t, x_mean_s, x_closest, t_count, s_count = torch.split(
            x_aug, self.in_dim, dim=-1
        )
        return x_hist, obs_mask, delta_norm, x_last, x_mean_t, x_mean_s, x_closest, t_count, s_count

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
        del obs_mask  # kept only for backward-compatible call sites during notebook migration
        batch_size, steps, nodes, _ = x_aug.shape
        x_hist, obs_mask, delta_norm, x_last, x_mean_t, x_mean_s, x_closest, t_count, s_count = self._split_augmented_input(x_aug)
        # Build a stable baseline repair for normalization, then learn how to
        # blend temporal and spatial repair candidates dynamically.
        x_base = x_hist * obs_mask + x_last * (1.0 - obs_mask)
        x_base_norm, revin_mean, revin_std = self.revin.normalize(x_base)
        x_hist_norm = _safe_feature_tensor((x_hist - revin_mean) / revin_std)
        x_last_norm = _safe_feature_tensor((x_last - revin_mean) / revin_std)
        x_mean_t_norm = _safe_feature_tensor((x_mean_t - revin_mean) / revin_std)
        x_mean_s_norm = _safe_feature_tensor((x_mean_s - revin_mean) / revin_std)
        x_closest_norm = _safe_feature_tensor((x_closest - revin_mean) / revin_std)
        x_base_norm = _safe_feature_tensor(x_base_norm)
        repair_context = torch.cat(
            [1.0 - obs_mask, delta_norm, x_last_norm, x_mean_t_norm, x_mean_s_norm, x_closest_norm, t_count, s_count],
            dim=-1,
        )
        repair_context = _safe_feature_tensor(repair_context)
        repair_logits = self.repair_weight_proj(repair_context).view(batch_size, steps, nodes, self.in_dim, 4)
        repair_logits = torch.nan_to_num(repair_logits, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        repair_weights = torch.softmax(repair_logits, dim=-1)
        repair_candidates = torch.stack([x_last_norm, x_mean_t_norm, x_mean_s_norm, x_closest_norm], dim=-1)
        x_fill_norm = (repair_weights * repair_candidates).sum(dim=-1)
        x_imputed_norm = x_hist_norm * obs_mask + x_fill_norm * (1.0 - obs_mask)
        x_imputed_norm = _safe_feature_tensor(x_imputed_norm)
        cond = _safe_feature_tensor(self._build_condition(x_imputed_norm, obs_mask, delta_norm))
        miss_ratio = 1.0 - obs_mask.mean(dim=(1, 3), keepdim=True)
        gap_ratio = delta_norm.amax(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        missing_strength = torch.clamp(miss_ratio + gap_ratio, min=0.0, max=1.0)

        x_main = self.x_proj(x_imputed_norm)
        if float(missing_strength.max().detach().cpu()) > 0.0:
            miss_inputs = repair_context
            x_missing = self.miss_proj(miss_inputs)
            miss_gate = self.miss_gate(cond).unsqueeze(1) * missing_strength
            x = x_main + miss_gate * x_missing + self.time_embed[:, :steps]
        else:
            x = x_main + self.time_embed[:, :steps]
        h = x
        for block, gate_layer in zip(self.blocks, self.gates):
            props = self._graph_propagate(h, a_hat)
            h_block = block(props[0], cond)
            for prop in props[1:]:
                hk = block(prop, cond)
                gate = torch.sigmoid(gate_layer(h_block))
                h_block = h_block + gate * hk
            h = h_block

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
        return torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
