"""브랜치 profiler/93-clip_grad_mode 전용 (clip_gradients_nosync · clip_gradients_foreach 가 거기에만 있음).

#93 개선안 ② — clip_gradients(loop) 와 clip_gradients_foreach 가 같은 기울기에 비트 동일한 결과를 내는가."""
import sys
from pathlib import Path

import pandas as pd
import torch

W = Path(__file__).resolve().parents[2]   # wesep 루트
sys.path.insert(0, str(W))
from wesep.models import get_model  # noqa: E402
from wesep.utils.funcs import clip_gradients, clip_gradients_foreach, clip_gradients_nosync  # noqa: E402
from wesep.utils.utils import parse_config_or_kwargs  # noqa: E402

torch.backends.cudnn.benchmark = False


dev = torch.device("cuda")
rows = []
for name in ("bsrnn_ecapa_sdfilm_spkeval", "bsrnn_ecapa_FiLM_spkeval"):
    cfg = parse_config_or_kwargs(str(W / f"examples/librimix/tse/v2/confs/{name}.yaml"))
    args = dict(cfg["model_args"]["tse_model"]); args["spk_model_init"] = False
    torch.manual_seed(0)
    model = get_model(cfg["model"]["tse_model"])(**args).to(dev).train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for amp_dtype, use_scaler in ((torch.float16, True), (torch.bfloat16, False)):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        g = torch.Generator(device="cpu").manual_seed(1)
        mix = (torch.randn(4, 48000, generator=g) * 0.1).to(dev)
        enroll = torch.randn(4, 376, 80, generator=g).to(dev)
        target = (torch.randn(4, 48000, generator=g) * 0.1).to(dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(mix, enroll)
            out = out[0] if isinstance(out, (list, tuple)) else out
            loss = ((out.float() - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                       # 학습 루프와 같은 순서 — 자르기 직전 상태
        params = [p for p in model.parameters() if p.grad is not None]
        g0 = [p.grad.detach().clone() for p in params]
        for clip in (5.0, 1e-3, 1e-6):
            for p, g_ in zip(params, g0):
                p.grad.copy_(g_)
            clip_gradients(model, clip)
            a = [p.grad.detach().clone() for p in params]
            for p, g_ in zip(params, g0):
                p.grad.copy_(g_)
            norms_b = clip_gradients_foreach(model, clip)
            b = [p.grad.detach().clone() for p in params]
            for p, g_ in zip(params, g0):
                p.grad.copy_(g_)
            clip_gradients_nosync(model, clip)
            c = [p.grad.detach().clone() for p in params]
            norms_a = [g_.norm(2) for g_ in g0]
            clipped = sum(int((clip / (n + 1e-6)) < 1) for n in norms_a)
            rows.append({
                "config": name, "amp": str(amp_dtype).split(".")[-1], "clip": clip,
                "텐서": len(params), "잘린 텐서": clipped,
                "노름 비트동일": sum(torch.equal(x, y) for x, y in zip(norms_a, norms_b)),
                "foreach_기울기_비트동일": sum(torch.equal(x, y) for x, y in zip(a, b)),
                "foreach_max_rel_diff": f"{max(float(((x - y).abs() / y.abs().clamp_min(1e-30)).max()) for x, y in zip(a, b)):.1e}",
                "nosync_기울기_비트동일": sum(torch.equal(x, y) for x, y in zip(a, c)),
            })
print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)}")
print(pd.DataFrame(rows).to_string(index=False))
