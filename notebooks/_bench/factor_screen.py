"""의심 요인을 하나씩 바꿔 가며 손실이 따라 바뀌는지 추적함.

    python notebooks/_bench/factor_screen.py --gpu 0 --arm base --rep 1

빠르게 보려고 **2 스텝만** 돎. 컴파일 안 씀.
데이터로더도 안 씀 — 고정 난수 배치라 run 사이에 입력이 같음.

측정 3개를 나눠 찍는 이유는 **어느 층에서 갈렸는지** 를 바로 보기 위함임:

    loss_step1      순전파만. 가중치 갱신 전
    grad_sum_step1  그 손실의 역전파 결과. optimizer.step() 하기 **전**
    loss_step2      갱신 뒤 두 번째 배치의 순전파

loss_step1 이 같은데 grad_sum_step1 이 다르면 → 역전파가 원인
loss_step1 부터 다르면                        → 순전파(알고리즘 선택)가 원인

CSV 왼쪽은 통제변인(요인), 오른쪽은 측정값임.
부동소수점은 repr 로 적어 소수점을 잃지 않게 함.
"""
import argparse
import csv
import fcntl
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wesep.models import get_model                      # noqa: E402
from wesep.utils.funcs import clip_gradients            # noqa: E402
from wesep.utils.losses import parse_loss               # noqa: E402
from wesep.utils.utils import set_seed                  # noqa: E402

V2 = Path(__file__).resolve().parents[2] / "examples/librimix/tse/v2"
OUT = Path(__file__).resolve().parents[1] / "_runs/factor_screen.csv"

# 왼쪽 = 바꿔 가며 시험하는 요인
FACTORS = ["arm", "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32",
           "matmul_allow_tf32", "use_det_algos", "cudnn_enabled", "amp",
           "enr_len", "tracker", "gpu", "rep", "seed"]
# 오른쪽 = 측정값
MEASURES = ["loss_step1", "grad_sum_step1", "loss_step2",
            "weight_sum_after", "sec", "gpu_name", "pid", "torch", "cudnn"]


def as_bool(s):
    return str(s).lower() in ("1", "true", "yes", "on")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="base", help="이 조건 묶음의 이름")
    p.add_argument("--gpu", default="0")
    p.add_argument("--rep", default="1")
    p.add_argument("--tracker", default="none",
                   choices=["none", "tensorboard", "both"])
    # 요인들 — 적지 않으면 wesep 원본과 같은 상태
    p.add_argument("--benchmark", default="true")
    p.add_argument("--deterministic", default="false")
    p.add_argument("--cudnn_tf32", default="true")
    p.add_argument("--matmul_tf32", default="false")
    p.add_argument("--det_algos", default="false")
    p.add_argument("--cudnn_enabled", default="true")
    p.add_argument("--amp", default="false")
    # 실제 학습은 등록 발화를 통째로 읽고 collate 가 배치 최소 길이로 자르므로
    # ResNet34 의 conv 입력 t 축이 스텝마다 달라짐(processor.py:477-479).
    # cudnn.benchmark 의 알고리즘 캐시 키에 shape 이 들어가므로
    # shape 이 바뀌면 스텝마다 재탐색이 일어남 — 그 조건을 재현하는 축임
    p.add_argument("--enr_len", default="335",
                   help="'335' 처럼 고정하거나 'vary' 로 배치마다 다르게")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=str(V2 / "confs/bsrnn.yaml"))
    p.add_argument("--exp_dir", default=None)
    a = p.parse_args()

    t0 = time.time()
    configs = yaml.safe_load(open(a.config))
    seed = a.seed if a.seed is not None else configs["seed"]

    # train.py:88 과 같은 자리. set_seed 안에서 cudnn.benchmark = True 가
    # 켜지므로(wesep/utils/utils.py:112) 요인 설정은 그 뒤에 와야 함
    set_seed(seed)
    torch.backends.cudnn.enabled = as_bool(a.cudnn_enabled)
    torch.backends.cudnn.benchmark = as_bool(a.benchmark)
    torch.backends.cudnn.deterministic = as_bool(a.deterministic)
    torch.backends.cudnn.allow_tf32 = as_bool(a.cudnn_tf32)
    torch.backends.cuda.matmul.allow_tf32 = as_bool(a.matmul_tf32)
    if as_bool(a.det_algos):
        # cuBLAS 는 CUBLAS_WORKSPACE_CONFIG 가 먼저 잡혀 있어야 함(runner 가 줌)
        torch.use_deterministic_algorithms(True, warn_only=True)

    exp_dir = a.exp_dir or str(Path(__file__).resolve().parents[1] /
                               f"_runs/fs/{a.arm}_g{a.gpu}_r{a.rep}")
    # tracker 블록 — train.py 와 **순서가 다름.** 일부러 그렇게 둔 것임
    #   train.py : set_seed(88) -> 모델 생성(219) -> DDP(243) -> tracker(322-342)
    #   이 probe : set_seed     -> tracker        -> 모델 생성
    # tracker 를 앞에 두면 tracker 가 전역 RNG 를 소비할 때 초기 가중치가 달라져
    # loss_step1 이 바로 갈림 — train.py 순서보다 **더 엄격한** 검사임
    writer = None
    if a.tracker != "none":
        os.makedirs(exp_dir, exist_ok=True)
        if a.tracker == "both":
            import wandb
            wandb.init(project="probe", mode="offline", dir=exp_dir)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(exp_dir, "tb"))

    device = torch.device("cuda")
    model = get_model(
        configs["model"]["tse_model"])(**configs["model_args"]["tse_model"])
    model.to(device)
    model.train()

    crit = parse_loss(configs["loss"])[0]
    if hasattr(crit, "to"):
        crit = crit.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs["scheduler_args"]["tse_model"]["initial_lr"],
        weight_decay=configs["optimizer_args"]["tse_model"]["weight_decay"])
    enable_amp = as_bool(a.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=enable_amp)

    # 고정 배치 2개 — 전역 RNG 와 분리. run 사이에 반드시 같아야 함
    bs = configs["dataloader_args"]["batch_size"]
    chunk = configs["dataset_args"]["chunk_len"]
    g = torch.Generator().manual_seed(1234)
    # 'vary' 면 배치마다 t 축을 다르게 줘서 cudnn.benchmark 재탐색을 일으킴
    enr_lens = ([335, 291] if a.enr_len == "vary"
                else [int(a.enr_len), int(a.enr_len)])
    batches = [(torch.randn(bs, chunk, generator=g),
                torch.randn(bs, enr_lens[i], 80, generator=g),
                torch.randn(bs, chunk, generator=g)) for i in range(2)]

    def step(mix, enr, tgt):
        mix, enr, tgt = (x.float().to(device) for x in (mix, enr, tgt))
        with torch.amp.autocast(device.type, enabled=enable_amp):
            out = model(mix, enr)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return crit(out[0], tgt).mean()

    # --- 스텝 1 : 순전파 -> 손실 -> 역전파 -> (기울기 측정) -> 갱신 ---
    loss1 = step(*batches[0])
    l1 = loss1.item()
    optimizer.zero_grad()
    scaler.scale(loss1).backward()
    scaler.unscale_(optimizer)
    # optimizer.step() 전에 기울기를 잼 — 역전파만의 결과임
    grad_sum = float(sum(p.grad.detach().double().sum()
                         for p in model.parameters() if p.grad is not None))
    clip_gradients(model, configs["clip_grad"])
    scaler.step(optimizer)
    scaler.update()

    # --- 스텝 2 : 갱신된 가중치로 순전파만 ---
    with torch.no_grad():
        l2 = step(*batches[1]).item()

    w_sum = float(sum(p.detach().double().sum() for p in model.parameters()))

    row = {
        "arm": a.arm,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "use_det_algos": torch.are_deterministic_algorithms_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "amp": enable_amp,
        "enr_len": a.enr_len,
        "tracker": a.tracker,
        "gpu": a.gpu,
        "rep": a.rep,
        "seed": seed,
        "loss_step1": repr(l1),
        "grad_sum_step1": repr(grad_sum),
        "loss_step2": repr(l2),
        "weight_sum_after": repr(w_sum),
        "sec": round(time.time() - t0, 2),
        "gpu_name": torch.cuda.get_device_name(0).replace(",", " "),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=FACTORS + MEASURES)
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)

    print(f"{a.arm:<16} g{a.gpu} r{a.rep}  "
          f"loss1={row['loss_step1']:<20} "
          f"grad={row['grad_sum_step1']:<22} "
          f"loss2={row['loss_step2']:<20} {row['sec']}s")

    if writer is not None:
        writer.close()
        if a.tracker == "both":
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
