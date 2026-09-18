"""학습 손실이 run 마다 흔들리는 원인을 데이터셋 없이 가려냄.

    python notebooks/_bench/train_step_probe.py \
        --gpu 0 --tracker none --benchmark true --rep 1 --tag solo

Libri2Mix 가 사라져 run.sh stage 3 을 못 돌리므로, 데이터로더만 빼고
나머지는 실제 학습 경로를 그대로 씀:

  모델   : confs/bsrnn.yaml 의 BSRNN (ResNet34 화자 인코더 포함)
  손실   : wesep.utils.losses 의 SISDR (auraloss.time.SISDRLoss)
  최적화 : Adam + clip_gradients + GradScaler(enabled=False)
           executor.py:143-149 와 같은 순서
  손실집계: sum(losses)/len(losses)  — executor.py:166 과 같은 정의

배치는 전역 RNG 와 분리된 torch.Generator(seed=1234) 로 **한 번 만들어 고정**함.
그래서 run 사이에 입력이 달라질 여지가 없음 — 남는 변인은 커널 선택뿐임.

통제변인은 왼쪽(tracker · cudnn_benchmark · gpu · rep · tag), 측정값은 오른쪽에 둠.
손실은 repr(float) 로 적어 소수점을 잃지 않게 함.
"""
import argparse
import csv
import fcntl
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wesep.models import get_model                      # noqa: E402
from wesep.utils.funcs import clip_gradients            # noqa: E402
from wesep.utils.losses import parse_loss               # noqa: E402
from wesep.utils.utils import set_seed                  # noqa: E402

V2 = Path(__file__).resolve().parents[2] / "examples/librimix/tse/v2"
OUT = Path(__file__).resolve().parents[1] / "_runs/train_step_probe.csv"

LEFT = ["tag", "tracker", "cudnn_benchmark", "cudnn_deterministic",
        "use_det_algos", "gpu", "rep", "gpu_name", "pid", "torch", "cudnn"]
RIGHT = ["ep1_train", "ep2_train", "ep3_train",
         "step1_loss", "all_steps", "weight_sum_after"]


def as_bool(s):
    return str(s).lower() in ("1", "true", "yes", "on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tracker", default="none",
                    choices=["none", "tensorboard", "both"])
    ap.add_argument("--benchmark", default="true")
    ap.add_argument("--deterministic", default="false",
                    help="wesep/utils/utils.py:115 에 주석 처리된 "
                         "torch.backends.cudnn.deterministic 을 켜 봄")
    ap.add_argument("--strict_det", default="false",
                    help="torch.use_deterministic_algorithms(True, warn_only=True). "
                         "cuDNN 밖의 atomicAdd 커널까지 결정적 구현으로 바꿈")
    ap.add_argument("--rep", default="1")
    ap.add_argument("--tag", default="solo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--config", default=str(V2 / "confs/bsrnn.yaml"))
    ap.add_argument("--exp_dir", default=None)
    a = ap.parse_args()

    configs = yaml.safe_load(open(a.config))

    # train.py:88 과 같은 자리. set_seed 안에서 cudnn.benchmark = True 가 켜지므로
    # (wesep/utils/utils.py:116) 그 뒤에 덮어써야 함
    set_seed(configs["seed"])
    torch.backends.cudnn.benchmark = as_bool(a.benchmark)
    torch.backends.cudnn.deterministic = as_bool(a.deterministic)
    if as_bool(a.strict_det):
        # cuBLAS 는 CUBLAS_WORKSPACE_CONFIG 가 먼저 잡혀 있어야 함 (runner 에서 줌)
        torch.use_deterministic_algorithms(True, warn_only=True)

    exp_dir = a.exp_dir or str(
        Path(__file__).resolve().parents[1] /
        f"_runs/probe/{a.tag}_g{a.gpu}_r{a.rep}_{a.tracker}_bm{a.benchmark}")
    os.makedirs(exp_dir, exist_ok=True)

    # tracker 블록 — train.py 와 **순서가 다름.** 일부러 그렇게 둔 것임
    #   train.py : set_seed(88) -> 모델 생성(219) -> DDP(243) -> tracker(322-342)
    #   이 probe : set_seed     -> tracker        -> 모델 생성
    # tracker 를 모델 생성보다 앞에 두면, tracker 가 전역 RNG 를 한 눈금이라도
    #소비할 경우 초기 가중치가 달라져 loss_step1 이 바로 갈림 — 즉 train.py 순서보다
    # **더 엄격한** 검사임. 실측에서 tracker 유무가 loss_step1 을 안 바꿨으므로
    # SummaryWriter·wandb.init 이 torch 전역 RNG 를 소비하지 않음이 확인된 것임
    writer = None
    if a.tracker != "none":
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

    criterion = parse_loss(configs["loss"])
    crit = criterion[0].to(device) if hasattr(criterion[0], "to") \
        else criterion[0]

    opt_args = configs["optimizer_args"]["tse_model"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs["scheduler_args"]["tse_model"]["initial_lr"],
        weight_decay=opt_args["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    # 고정 배치 — 전역 RNG 와 분리. run 사이에 반드시 같아야 함
    bs = configs["dataloader_args"]["batch_size"]
    chunk = configs["dataset_args"]["chunk_len"]
    g = torch.Generator().manual_seed(1234)
    n_batch = a.epochs * a.steps
    batches = [(torch.randn(bs, chunk, generator=g),
                torch.randn(bs, 335, 80, generator=g),
                torch.randn(bs, chunk, generator=g))
               for _ in range(n_batch)]

    ep_means, all_steps, step1 = [], [], None
    k = 0
    for ep in range(1, a.epochs + 1):
        losses = []
        for _ in range(a.steps):
            mix, enr, tgt = batches[k]
            k += 1
            mix = mix.float().to(device)
            enr = enr.float().to(device)
            tgt = tgt.float().to(device)

            outputs = model(mix, enr)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            loss = crit(outputs[0], tgt).mean()

            losses.append(loss.item())
            if step1 is None:
                step1 = loss.item()

            # executor.py:143-149 와 같은 순서
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model, configs["clip_grad"])
            scaler.step(optimizer)
            scaler.update()

        m = sum(losses) / len(losses)
        ep_means.append(m)
        all_steps.extend(losses)
        if writer is not None:
            writer.add_scalar("train/loss", m, ep)

    w_sum = float(sum(p.detach().double().sum()
                      for p in model.parameters()))

    row = {"tag": a.tag, "tracker": a.tracker,
           "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "use_det_algos": torch.are_deterministic_algorithms_enabled(),
           "gpu": a.gpu, "rep": a.rep,
           "gpu_name": torch.cuda.get_device_name(0).replace(",", " "),
           "pid": os.getpid(),
           "torch": torch.__version__,
           "cudnn": torch.backends.cudnn.version(),
           "step1_loss": repr(step1),
           "all_steps": " ".join(repr(x) for x in all_steps),
           "weight_sum_after": repr(w_sum)}
    for i in range(3):
        row[f"ep{i+1}_train"] = repr(ep_means[i]) if i < len(ep_means) else ""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=LEFT + RIGHT)
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)

    print(" | ".join(f"{k2}={row[k2]}" for k2 in
                     ["tag", "tracker", "cudnn_benchmark", "gpu", "rep",
                      "ep1_train", "ep2_train", "ep3_train"]))

    if writer is not None:
        writer.close()
        if a.tracker == "both":
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
