"""학습 지표를 CSV·텐서보드·wandb 에 같은 값으로 기록함. 이 포크에서 더한 파일임.

    tracker = Tracker(configs, logger)
    tracker.log_step({"epoch": 1, "global_step": 50, "train/loss_step": ...})
    tracker.log_epoch({"epoch": 1, "global_step": 1737,
                       "train/loss_epoch": ..., "val/loss": ...})
    tracker.close()

**SD-FiLM 과 같은 Lightning 로거를 그대로 씀** — 직접 파일을 쓰지 않고
`log_metrics(dict, step)` 만 부름. 세 로거가 같은 dict 를 받으므로 값이 어긋날 수 없음.

    exp_dir/csv/version_N/metrics.csv   CSVLogger
    exp_dir/tb/version_N/               TensorBoardLogger
    exp_dir/wandb/                      WandbLogger (tracker=both 일 때만)

**호출부 dict 의 key 가 그대로 CSV 열이자 텐서보드 태그**가 됨 —
그래프에 뜰 이름이 executor.py 호출부에 글자 그대로 적혀 있음.
지표를 늘리려면 호출부에 key 를 더하면 되고 이 파일은 안 고쳐도 됨.

이름 규칙은 **SD-FiLM 과 같은 Lightning 방식** — `<stage>/<지표>` 로 stage 를 앞에 둠
(`val/text/pn/si_sdr_i` 처럼). 같은 지표를 스텝·에포크 두 단위로 남길 때는
Lightning 이 붙이는 것과 같이 `_step` · `_epoch` 을 **뒤에** 붙임.

    train/loss_step   학습 스텝 손실          train/loss_epoch   학습 에포크 평균
    train/lr_step     그때의 학습률           val/loss           검증 에포크 평균

텐서보드가 `/` 앞을 그룹으로 묶으므로 **`train` · `val` 로 접힘.**

**`version` 을 자동 증가로 둠.** wesep 은 latest_checkpoint.pt 로 자동 재개하는데,
version 을 고정하면 Lightning 이 재개할 때마다 기존 metrics.csv 를 **지움**(실측).
자동 증가면 재개마다 version_N 이 새로 생겨 이전 기록이 남음.

rank 0 이 아니거나 configs 가 None 이면 전부 no-op —
DDP 는 모든 rank 가 train()·cv() 를 도는데 다 기록하면 한 파일에 겹쳐 들어감.

CSV 는 tracker 설정과 무관하게 항상 씀. tracker=none 으로 돌린 run 도 비교 대상이고
CSV 는 외부 의존성이 없기 때문임.

**끝자리까지 대조할 일에는 CSV 를 쓸 것** — tfevents 스칼라는 float32 라
50.018273162841794 가 50.018272399902344 로 깎임(실측). CSV 는 원 정밀도로 남음.
"""
import os

# 셋 다 lightning.pytorch.loggers 에서 가져옴 — WandbLogger 가 fabric 에는 없고,
# pytorch 판 CSVLogger·TensorBoardLogger 는 fabric 판을 상속한 것이라
# Trainer 없이도 같게 동작함(실측). 인자 이름도 save_dir 로 셋이 같아짐
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

# 지표가 아니라 x축·메모인 key — 접두사를 안 붙임
INDEX_KEYS = ("epoch", "global_step")


class Tracker:
    """Lightning 로거 묶음. rank 0 에서만 실제로 씀."""
    TRACKER_LIST = ["none", "tensorboard", "both"]

    def __init__(self, configs=None, logger=None):
        self.loggers = []
        self.exp_dir = None
        self.tracker = "none"
        self.step_interval = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.enabled = (configs is not None and self.rank == 0)
        if not self.enabled:
            return

        self.exp_dir = configs["exp_dir"]
        self.step_interval = int(configs.get("tracker_step_interval", 50))
        self.tracker = configs.get("tracker", "none")

        self.loggers.append(CSVLogger(save_dir=self.exp_dir, name="csv"))
        if self.tracker == "none":
            pass
        elif self.tracker == "tensorboard":
            self.loggers.append(TensorBoardLogger(save_dir=self.exp_dir, name="tb"))
        elif self.tracker == "both":
            # 순서 제약은 없음 — 아래 WandbLogger 는 sync_tensorboard 를 안 넘기므로
            # wandb 가 SummaryWriter 를 가로채지 않음(lightning 2.6.0 소스 확인).
            # tfevents 는 TensorBoardLogger 가 따로 씀.
            # cbfce22 의 raw SummaryWriter 방식에서는 순서가 중요했고,
            # 9afd0ee 가 Lightning 로거로 바꾸며 그 근거가 없어졌음
            wandb_run_name = os.path.basename(self.exp_dir.rstrip("/"))
            self.loggers.append(
                WandbLogger(
                    project=configs.get("wandb_project", "wesep-tse"),
                    name=configs.get("wandb_run_name", wandb_run_name),
                    entity=configs.get("wandb_entity", None),
                    save_dir=self.exp_dir,
                    config=configs,
                )
            )
            self.loggers.append(TensorBoardLogger(save_dir=self.exp_dir, name="tb"))
        else:
            # run.sh --tracker 가 넘긴 값임. 오타면 조용히 다른 로거가 켜지므로 멈춤
            raise ValueError(f"tracker={self.tracker!r} 는 없는 값임. {self.TRACKER_LIST} 중 하나여야 함")

        # <<<<< 더한 것 - Lightning 로거는 .experiment 를 처음 건드릴 때 실제 자원을 만듦.
        #       그냥 두면 첫 log_metrics(= global_step 50)까지 미뤄져
        #       컴파일 113초 동안 wandb 에 run 이 안 보이고 exp_dir 도 비어 있었음(실측).
        #       여기서 한 번 깨우면 wandb.init()·tfevents·CSV 폴더가 다 지금 생김
        for lg in self.loggers:
            _ = lg.experiment

        names = " ".join(type(lg).__name__ for lg in self.loggers)
        logger.info(
            f"tracker={self.tracker} step_interval={self.step_interval} -> {self.exp_dir} ({names})"
        )

    def log_step(self, row):
        """global_step 이 step_interval 의 배수일 때만 씀. 단 global_step 0 은 건너뜀.

        호출부는 매 스텝 그냥 부르면 됨.
        에포크 안 번호가 아니라 전역 스텝으로 재는 것은 Lightning 과 같음 —
        epoch_iter 가 step_interval 의 배수가 아니어도 x축 간격이 균일해짐.

        **global_step 0 을 빼는 이유** — 그 점은 optimizer.step() 이 한 번도 불리기 전,
        즉 초기 가중치의 손실이라 값이 홀로 크게 튐(실측: 29.54 vs 그 뒤 -9.6~0.26).
        한 run 에 딱 한 점인데 그래프 y축의 76 % 를 차지해 나머지가 안 보였음.
        Lightning 은 optimizer.step() 뒤에 global_step 을 올려 이 점이 아예 안 생김.
        """
        if self.enabled:
            global_step = row["global_step"]
            if (self.step_interval and global_step > 0 and global_step % self.step_interval == 0):
                self._log(row)

    def log_epoch(self, row):
        """에포크 하나가 끝났을 때. x축은 그 에포크의 마지막 전역 스텝."""
        if self.enabled:
            self._log(row)

    def _log(self, row):
        """row 를 그대로 모든 로거에 넘김. 지표 이름은 호출부가 정함.

        Lightning 과 같이 **`global_step`(x축) 과 `epoch`(열) 둘을 필수**로 둠.
        row 에 없으면 KeyError 로 멈춤 - 빠지면 그래프의 x축이 조용히 어긋남.
        """
        metrics = {k: v for k, v in row.items() if k not in INDEX_KEYS}
        metrics["epoch"] = row["epoch"]
        global_step = row["global_step"]
        for lg in self.loggers:
            lg.log_metrics(metrics, step=global_step)
            lg.save()      # CSVLogger 는 save() 해야 디스크에 감 - tail -F 용

    def close(self):
        """CSV 를 비우고 wandb run 을 마감함."""
        for lg in self.loggers:
            lg.finalize("success")
