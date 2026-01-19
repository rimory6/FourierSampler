# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def rfft_lowpass_sliding_window_1d(x: torch.Tensor,
                                    window_ratio: float,
                                    progress: float,
                                    dim: int = 1) -> torch.Tensor:

    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    L = x32.shape[dim]

    Xf = torch.fft.rfft(x32, dim=dim)
    K = Xf.shape[dim]  

    window_size = max(1, int(window_ratio * K))

    max_center = K - window_size
    center = int(progress * max_center)
    center = max(0, min(center, max_center))

    window_start = center
    window_end = center + window_size

    H = torch.zeros(K, device=x.device, dtype=Xf.real.dtype)
    H[window_start:window_end] = 1.0

    view_shape = [1] * Xf.ndim
    view_shape[dim] = K
    H = H.view(*view_shape).to(Xf.dtype)

    Xf_filt = Xf * H
    y = torch.fft.irfft(Xf_filt, n=L, dim=dim)

    return y.to(orig_dtype)


class FFTLowPassController:
    def __init__(self,
                 window_ratio: float = 0.3,
                 enable: bool = True,
                 boost_beta_min: float = 0.0,
                 boost_beta_max: float = 0.3,
                 variance_scale: float = 1.0,
                 enforce_lowfreq_gate: bool = False,
                 gate_keep_ratio: float = 0.5):
        self.window_ratio = float(window_ratio)
        self.enable = bool(enable)
        self.beta_min = float(boost_beta_min)
        self.beta_max = float(boost_beta_max)
        self.variance_scale = float(variance_scale)
        self.enforce_gate = bool(enforce_lowfreq_gate)
        self.gate_keep_ratio = float(gate_keep_ratio)

        self.start = None
        self.end = None
        self.i = 0
        self.steps = 1
        self.block_id = 0

        self.last_in = None
        self.last_out = None
        self.current_beta = self.beta_max 

        self.variance_history = []

    def set_region(self, start: int, end: int):
        self.start, self.end = int(start), int(end)

    def set_progress(self, i: int, steps: int, block_id: int):
        self.i, self.steps, self.block_id = int(i), int(steps), int(block_id)

    def progress_ratio(self) -> float:
        if self.steps <= 1:
            return 1.0
        return float(self.i / (self.steps - 1))

    def pre_hook(self, module: nn.Module, inputs):
        if (not self.enable) or self.start is None or self.end is None:
            return None
        (h,) = inputs
        if h.dim() != 3:
            return None

        s, e = self.start, self.end
        if not (0 <= s < e <= h.shape[1]):
            return None

        seg = h[:, s:e, :]
        progress = self.progress_ratio()

        seg_f = rfft_lowpass_sliding_window_1d(
            seg,
            window_ratio=self.window_ratio,
            progress=progress,
            dim=1
        )

        h_mod = h.clone()
        h_mod[:, s:e, :] = seg_f.to(h.dtype)

        self.last_in = h.detach()
        self.last_out = h_mod.detach()

        return (h,)

    def lowfreq_scores(self) -> Optional[torch.Tensor]:
        if self.last_in is None or self.last_out is None or self.start is None:
            return None

        s, e = self.start, self.end
        H_low = self.last_out[:, s:e, :].to(torch.float32)

        energy = torch.sum(H_low ** 2, dim=-1)
        energy_max = energy.max(dim=1, keepdim=True)[0] + 1e-8
        s_normalized = energy / energy_max

        return s_normalized

    def _compute_percentile(self, value: float, history: list) -> float:
        if not history:
            return 50.0

        sorted_history = sorted(history)
        n = len(sorted_history)
        count_less = sum(1 for v in sorted_history if v < value)
        percentile = (count_less / n) * 100

        return percentile

    def compute_variance_adaptive_beta(self, logits: torch.Tensor, mask_index: torch.Tensor) -> float:
        if not mask_index.any():
            return self.beta_min

        masked_logits = logits[mask_index]
        probs = F.softmax(masked_logits.float(), dim=-1)

        confidence, _ = torch.max(probs, dim=-1) 
        current_var = confidence.var().item()

        if not hasattr(self, 'variance_history'):
            self.variance_history = []

        self.variance_history.append(current_var)
        if len(self.variance_history) > 20:
            self.variance_history.pop(0)

        percentile = self._compute_percentile(current_var, self.variance_history)

        normalized_percentile = (percentile - 50) / 50  
        z_score = normalized_percentile * 3 

        weight = 0.5 * (1.0 + torch.erf(torch.tensor(z_score / np.sqrt(2))).item())
        weight = np.clip(weight, 0.0, 1.0)

        adaptive_beta = self.beta_min + (1-weight) * (self.beta_max - self.beta_min)  

        return adaptive_beta


def get_lm_head_module(model: nn.Module) -> nn.Module:

    if hasattr(model, "lm_head") and isinstance(getattr(model, "lm_head"), nn.Module):
        return model.lm_head
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    candidate = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if vocab_size is None or getattr(m, "out_features", None) == vocab_size:
                candidate = m
    if candidate is None:
        raise RuntimeError("fail to locate lm_head")
    return candidate


@torch.no_grad()
def generate_fft(model, prompt, tokenizer, steps=128, gen_length=128, block_length=32,
                 temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
                 fft_cfg: Optional[dict] = None):

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if fft_cfg is None:
        fft_cfg = {}
    controller = FFTLowPassController(
        window_ratio=fft_cfg.get("window_ratio", 0.2),
        enable=fft_cfg.get("enable", True),
        boost_beta_min=fft_cfg.get("boost_beta_min", 0.4),
        boost_beta_max=fft_cfg.get("boost_beta_max", 0.6),
        variance_scale=fft_cfg.get("variance_scale", 1.0),
        enforce_lowfreq_gate=fft_cfg.get("enforce_lowfreq_gate", False),
        gate_keep_ratio=fft_cfg.get("gate_keep_ratio", 1.0),
    )

    handle = None
    try:
        if controller.enable:
            lm_head = get_lm_head_module(model)
            handle = lm_head.register_forward_pre_hook(controller.pre_hook)
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Warn] FFT hook can't work: {e}")
        controller.enable = False

    for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0), desc="FFT Generation"):
        b_start = prompt.shape[1] + num_block * block_length
        b_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, b_start:b_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        controller.set_region(b_start, b_end)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            controller.set_progress(i=i, steps=steps_per_block, block_id=num_block)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if controller.enable:
                adaptive_beta = controller.compute_variance_adaptive_beta(logits, mask_index)
                controller.current_beta = adaptive_beta

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, b_end:] = -np.inf

            if controller.enable and controller.current_beta > 0.0:
                lf = controller.lowfreq_scores()
                if lf is not None:
                    pos_boost = torch.zeros_like(x0_p)
                    pos_boost[:, b_start:b_end] = lf
                    x0_p = x0_p + controller.current_beta * pos_boost

                    if controller.enforce_gate:
                        keep_ratio = controller.gate_keep_ratio + (1.0 - controller.gate_keep_ratio) * ((i + 1) / steps_per_block)
                        keep_ratio = float(min(1.0, max(0.0, keep_ratio)))
                        B = x0_p.shape[0]
                        for bj in range(B):
                            cur = lf[bj]
                            k = max(1, int(keep_ratio * cur.numel()))
                            topk_vals, _ = torch.topk(cur, k=k, sorted=True)
                            cutoff_val = topk_vals.min()
                            keep_mask = (pos_boost[bj, b_start:b_end] >= cutoff_val)
                            x0_p[bj, b_start:b_end][~keep_mask] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    if handle is not None:
        handle.remove()

    return x
