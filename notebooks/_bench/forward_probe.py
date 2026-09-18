"""학습 없이 forward 만 해서 가중치·출력이 tracker 설정에 따라 달라지는지 봄.

    python notebooks/_bench/forward_probe.py <config> <tracker> <exp_dir>

backward 를 안 하므로 스텝 간 누적이 없음 — 순수 연산 차이만 보임.
데이터 로더도 안 씀(고정 난수 입력)이라 run 하나가 몇 초로 끝남.
결과는 notebooks/_runs/forward_probe.csv 에 한 줄씩 append 함.
"""
import csv
import fcntl
import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wesep.models import get_model                     # noqa: E402
from wesep.utils.utils import set_seed                  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "_runs/forward_probe.csv"
LEFT = ["tracker", "cudnn_benchmark", "gpu_name", "rep"]
RIGHT = ["n_params", "weight_sha", "weight_sum",
         "out_sha", "out_sum", "sisdr_loss", "fwd_repeat_same"]


def sha(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def si_sdr(est, ref, eps=1e-8):
    """음수 SI-SDR — 값이 작을수록 좋음. wesep 의 SISDR 과 같은 정의."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(
        -1, keepdim=True) + eps)
    noise = est - proj
    return -(10 * torch.log10(proj.pow(2).sum(-1) /
                              (noise.pow(2).sum(-1) + eps) + eps)).mean()


def main(cfg_path, tracker, exp_dir, rep="1"):
    configs = yaml.safe_load(open(cfg_path))
    set_seed(configs["seed"])          # train.py:90 과 같은 자리

    # tracker 블록 — train.py 와 **순서가 다름.** 일부러 그렇게 둔 것임
    #   train.py : set_seed(90) -> 모델(221) -> DDP(245) -> Executor(335)
    #              tracker 는 9afd0ee 이후 Executor 안으로 들어갔음
    #   이 probe : set_seed     -> tracker        -> 모델 생성
    # tracker 를 앞에 두면 tracker 가 전역 RNG 를 소비할 때 초기 가중치가 달라져
    # weight_sha 가 바로 갈림 — train.py 순서보다 **더 엄격한** 검사임
    writer = None
    if tracker != "none":
        if tracker == "both":
            import wandb
            wandb.init(project="probe", mode="offline", dir=exp_dir)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(exp_dir, "tb"))

    model = get_model(
        configs["model"]["tse_model"])(**configs["model_args"]["tse_model"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    sd = model.state_dict()
    keys = sorted(sd.keys())
    w_sha = sha(*[sd[k].float() for k in keys])
    w_sum = float(sum(sd[k].double().sum() for k in keys))
    n_par = sum(p.numel() for p in model.parameters())

    # 고정 입력 — 전역 시드와 분리된 generator 를 써서 tracker 유무와 무관하게 같음
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(16, 48000, generator=g).to(device)
    enr = torch.randn(16, 335, 80, generator=g).to(device)
    ref = torch.randn(16, 48000, generator=g).to(device)

    outs = []
    with torch.no_grad():
        for _ in range(3):             # 같은 프로세스 안에서 3회 반복
            o = model(x, enr)
            o = o[0] if isinstance(o, (list, tuple)) else o
            outs.append(o)
    same = all(torch.equal(outs[0], o) for o in outs[1:])
    out = outs[0]

    row = {"tracker": tracker,
           "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "gpu_name": torch.cuda.get_device_name(0).replace(",", " "),
           "rep": rep,
           "n_params": n_par, "weight_sha": w_sha,
           "weight_sum": f"{w_sum:.10f}",
           "out_sha": sha(out.float()), "out_sum": f"{float(out.double().sum()):.10f}",
           "sisdr_loss": f"{float(si_sdr(out.float(), ref)):.10f}",
           "fwd_repeat_same": same}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=LEFT + RIGHT)
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(" | ".join(f"{k}={v}" for k, v in row.items()))

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(*sys.argv[1:5])
