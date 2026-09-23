"""브랜치 profiler/93-clip_grad_mode 전용 (clip_gradients_nosync · clip_gradients_foreach 가 거기에만 있음).

#93 개선안 ② — clip 함수 하나의 시간 (loop · nosync · foreach). 학습 루프 없이 실제 모델 기울기로 잼.

앞뒤로 torch.cuda.synchronize() 를 해서, 앞선 backward 가 남긴 GPU 작업을 빼고 **자르기 자체의 비용**만 봄.
"""
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

W = Path(__file__).resolve().parents[2]   # wesep 루트
sys.path.insert(0, str(W))
from wesep.models import get_model  # noqa: E402
from wesep.utils.funcs import clip_gradients, clip_gradients_foreach, clip_gradients_nosync  # noqa: E402
from wesep.utils.utils import parse_config_or_kwargs  # noqa: E402

dev = torch.device("cuda")
FNS = {"loop": clip_gradients, "nosync": clip_gradients_nosync, "foreach": clip_gradients_foreach}
rows = []
for name in ("bsrnn_ecapa_sdfilm_spkeval", "bsrnn_ecapa_FiLM_spkeval"):
    cfg = parse_config_or_kwargs(str(W / f"examples/librimix/tse/v2/confs/{name}.yaml"))
    args = dict(cfg["model_args"]["tse_model"]); args["spk_model_init"] = False
    torch.manual_seed(0)
    model = get_model(cfg["model"]["tse_model"])(**args).to(dev).train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(1)
    mix, enroll = (torch.randn(8, 48000, generator=g) * 0.1).to(dev), torch.randn(8, 376, 80, generator=g).to(dev)
    with torch.amp.autocast("cuda", dtype=torch.float16):          # 학습과 같은 16-mixed
        out = model(mix, enroll); out = out[0] if isinstance(out, (list, tuple)) else out
        loss = (out.float() ** 2).mean()
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    params = [p for p in model.parameters() if p.grad is not None]
    g0 = [p.grad.detach().clone() for p in params]
    for clip in (5.0, 1e-6):                                          # 실제 값 · 모든 텐서가 잘리는 경우
        for mode, fn in FNS.items():
            ts = []
            for it in range(25):
                for p, x in zip(params, g0):
                    p.grad.copy_(x)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                fn(model, clip)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                if it >= 5:                                           # 앞 5회는 예열
                    ts.append((t1 - t0) * 1000)
            rows.append({"config": name.replace("bsrnn_ecapa_", ""), "clip": clip, "mode": mode,
                         "텐서": len(params), "중앙값_ms": round(statistics.median(ts), 2),
                         "최소_ms": round(min(ts), 2), "최대_ms": round(max(ts), 2)})
print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · 20회 측정 (예열 5회 제외)")
print(pd.DataFrame(rows).to_string(index=False))
