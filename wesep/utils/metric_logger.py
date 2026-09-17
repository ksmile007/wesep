"""학습 지표를 CSV 와 텐서보드에 **같은 값으로** 라이브 기록함.

이 파일은 wesep 원본에 없음 — 이 포크에서 더한 것임.

호출부는 **dict 하나**를 넘김. 그 dict 의 **key 가 CSV 열 이름**이 되고
**value 가 그 행의 값**이 됨. 기록할 지표를 늘리려면 key 를 하나 더 넣으면 되고,
이 파일을 고칠 필요가 없음.

    mlog = make_logger(exp_dir, writer, step_interval=50)   # 그냥 dict 임

    log_step(mlog, {"epoch": epoch, "step": i + 1, "global_step": cur_iter,
                    "loss": losses[-1], "running_mean": total_loss_avg,
                    "lr": optimizer.param_groups[0]["lr"]})

    log_epoch(mlog, {"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "lr": lr})

`wall_time` 은 이 파일이 자동으로 덧붙임 — 호출부가 신경 쓸 것이 아님.

CSV 와 텐서보드에 **같은 dict** 를 쓰므로 둘이 어긋날 여지가 없음.

`mlog` 는 클래스 인스턴스가 아니라 dict 이고, 파일은 쓸 때만 열었다 닫음.
그래서 닫아 줄 것이 없음 — 학습 끝에 close() 를 부를 의무가 생기지 않음.
기록이 step_interval 스텝에 한 번이라 open/close 비용은 학습 한 스텝보다 훨씬 쌈.

산출물은 exp_dir 아래 CSV 두 개임. 스텝 행과 에포크 행은 열이 달라
(스텝에는 val_loss 가 없고 에포크에는 step 이 없음) 한 파일에 섞지 않음.

    metrics_step.csv    epoch, step, global_step, loss, running_mean, lr, wall_time
    metrics_epoch.csv   epoch, train_loss, val_loss, lr, wall_time

**정밀도** — CSV 는 repr(float) 로 적음. tfevents 의 스칼라는 protobuf 의
simple_value 라 float32 이고, executor.py 의 train_loss 는 float32 값들의
파이썬 float64 평균이라 딱 떨어지지 않음. 실측으로 50.018273162841794 가
50.018272399902344 로 깎였음. 곡선을 보는 데는 무해하지만 소수점 끝자리까지
대조하는 작업에는 CSV 쪽을 써야 함.

**라이브 보장** — with 블록을 나올 때 파이썬 버퍼가 OS 로 넘어가므로
도는 중에 `tail -f` 로 보이고, 프로세스가 죽어도 그때까지가 남음.

**CSV 는 tracker 설정과 무관하게 항상 씀.** tracker=none 으로 돌린 run 도
나중에 비교 대상이 되기 때문임. tfevents 는 writer 가 있을 때만 씀.
"""
import csv
import datetime as dt
import numbers
import os

# CSV 열 이름 -> 텐서보드 태그.
# 여기 없는 key 는 아래 기본 규칙으로 태그를 만듦 — 열을 늘려도 이 표를 안 고쳐도 됨.
# 이 표가 있는 이유는 cbfce22 에서 이미 쓰던 태그 3개(train/loss · val/loss ·
# train/lr)를 그대로 유지하기 위함임. 태그가 바뀌면 기존 wandb run 과 이어지지 않음.
TB_TAG_STEP = {
    "loss": "train/loss_step",
    "running_mean": "train/loss_step_avg",
    "lr": "train/lr_step",
}
TB_TAG_EPOCH = {
    "train_loss": "train/loss",
    "val_loss": "val/loss",
    "lr": "train/lr",
}

# 지표가 아니라 x축·메모인 key — tfevents 에 스칼라로 올리지 않음
INDEX_KEYS = ("epoch", "step", "global_step", "wall_time")


def make_logger(exp_dir, writer=None, step_interval=50):
    """기록에 필요한 것만 담은 dict 를 만듦. rank 0 에서만 부를 것.

    exp_dir       : CSV 를 둘 폴더
    writer        : SummaryWriter 또는 None. None 이면 CSV 만 씀
    step_interval : 몇 스텝마다 기록할지. 0 이하면 스텝 기록을 안 함
    """
    os.makedirs(exp_dir, exist_ok=True)
    return {"dir": exp_dir, "writer": writer, "interval": int(step_interval)}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _append(mlog, name, row):
    """row 의 key 를 열 이름으로, value 를 행 값으로 해서 CSV 에 이어 붙임.

    파일이 없을 때만 헤더를 씀 — 이어학습이면 헤더가 두 번 들어가지 않음.
    실수는 repr 로 적어 소수점을 잃지 않게 함.
    """
    path = os.path.join(mlog["dir"], name)
    new = not os.path.exists(path)
    out = {k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(out)


def _to_tb(mlog, row, tag_map, suffix, x):
    """row 의 숫자 key 를 텐서보드 스칼라로 올림. CSV 와 같은 dict 를 씀."""
    writer = mlog["writer"]
    if writer is None:
        return
    for k, v in row.items():
        if k in INDEX_KEYS or not isinstance(v, numbers.Number):
            continue
        writer.add_scalar(tag_map.get(k, f"train/{k}{suffix}"), v, x)


def log_step(mlog, row):
    """에포크 안의 한 스텝. row 에 step·global_step 이 있어야 함.

    step 은 1부터 세고, interval 의 배수일 때만 기록함 —
    executor.py 의 `(i + 1) % log_interval == 0` 과 같은 방식임.
    간격 판정과 mlog is None 처리를 여기서 하므로 호출부는 매 스텝 그냥 부르면 됨.
    """
    if mlog is None or mlog["interval"] <= 0:
        return
    if row["step"] % mlog["interval"] != 0:
        return

    row = dict(row, wall_time=_now())
    _append(mlog, "metrics_step.csv", row)
    _to_tb(mlog, row, TB_TAG_STEP, "_step", row["global_step"])


def log_epoch(mlog, row):
    """에포크 하나가 끝났을 때. row 에 epoch 가 있어야 함."""
    if mlog is None:
        return

    row = dict(row, wall_time=_now())
    _append(mlog, "metrics_epoch.csv", row)
    _to_tb(mlog, row, TB_TAG_EPOCH, "", row["epoch"])
