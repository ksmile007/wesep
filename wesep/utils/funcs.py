# Created on 2018/12
# Author: Kaituo XU

import math

import torch
import torchaudio.compliance.kaldi as kaldi


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames
    by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions
                may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be
                    less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added
        frames of signal's inner-most two dimensions.

        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/
             contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length,
                               frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame,
                                                     subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes,
                              subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3:  # [B, C, T]
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


# <<<<< 더한 것 - clip_gradients 와 **같은 연산**을 GPU 동기화 없이 함 (#93, clip_grad_mode: nosync).
#       원본은 텐서마다 .item() 과 `if clip_coef < 1` 로 GPU 를 기다려 스텝당 약 1,070회 동기화했음.
#       노름·나눗셈·곱셈이 원본과 같은 연산이라 결과가 **비트 동일**함(실측, 2026-09-24)
def clip_gradients_nosync(model, clip):
    for p in model.parameters():
        if p.grad is not None:
            coef = clip / (p.grad.norm(2) + 1e-6)
            # 원본의 `if clip_coef < 1` 을 GPU 에서 — 1 이상(또는 NaN)이면 1.0 을 곱해 값이 그대로임
            p.grad.mul_(torch.where(coef < 1, coef, torch.ones_like(coef)))


# <<<<< 더한 것 - 같은 파라미터별 자르기를, 노름까지 한 번에 구하는 판 (#93, clip_grad_mode: foreach).
#       _foreach_norm 은 원본 .norm(2) 와 노름이 비트 동일하지 않아 자르기가 걸린 텐서의
#       기울기가 상대오차 약 3e-7 만큼 달라짐(실측, 2026-09-24) — 학습 조건이 미세하게 바뀜
def clip_gradients_foreach(model, clip):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return []
    norms = torch._foreach_norm(grads, 2)     # 텐서별 L2 노름을 한 번에 — 파이썬 float 로 안 꺼냄
    for g, n in zip(grads, norms):
        coef = clip / (n + 1e-6)
        # 원본의 `if clip_coef < 1` 을 GPU 에서 — 1 이상(또는 NaN)이면 1.0 을 곱해 값이 그대로임
        g.mul_(torch.where(coef < 1, coef, torch.ones_like(coef)))
    return norms


def compute_fbank(
    data,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=1.0,
    sample_rate=16000,
):
    """Extract fbank"""
    fbank_list = []
    for index_ in range(data.shape[0]):
        waveform = data[index_, :].unsqueeze(0)
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        fbank_list.append(mat.unsqueeze(0))
    np_fbank = torch.cat(fbank_list, 0)
    return np_fbank


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """Apply CMVN

    Args:
        data: Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1',
        'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']

    Returns:
        Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1',
        'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']
    """
    mat_list = []
    for index_ in range(data.shape[0]):
        mat = data[index_, :, :]
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        mat = mat.unsqueeze(0)
        mat_list.append(mat)
    np_mat = torch.cat(mat_list, 0)
    return np_mat


if __name__ == "__main__":
    torch.manual_seed(123)
    M, C, K, N = 2, 2, 3, 4
    frame_step = 2
    signal = torch.randint(5, (M, C, K, N))
    result = overlap_and_add(signal, frame_step)
    print(signal)
    print(result)
