from typing import List, Optional

import torch

from wesep.utils.schedulers import BaseClass

# <<<<< 더한 것 - save_checkpoint 가 담고 load_checkpoint 가 돌려주는 학습 상태 키.
#       옛 체크포인트에는 없으므로 읽을 때 .get() 으로 접근함
EXTRA_KEYS = ("epoch", "global_step", "train_loss", "val_loss", "torch_version")


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator"):
    assert type in ["generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def load_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.amp.GradScaler],    # <<<<< 고친 것 - torch.cuda.amp.GradScaler 의 부모 클래스. 둘 다 받음
    path: str,
    only_model: bool = False,
    mode: str = "all",
):
    assert mode in ["all", "generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if mode == "generator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][0]],
            [states["optimizers"][0]],
            [states["schedulers"][0]],
        )
    elif mode == "discriminator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][1]],
            [states["optimizers"][1]],
            [states["schedulers"][1]],
        )
    else:
        model_state, optimizer_state, scheduler_state = (
            states["models"],
            states["optimizers"],
            states["schedulers"],
        )

    for model, state in zip(models, model_state):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state, strict=True)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
    if not only_model:
        for optimizer, state in zip(optimizers, optimizer_state):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_state):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is not None:
                scaler.load_state_dict(states["scaler"])
    # <<<<< 더한 것 - 학습 상태를 돌려줌. 원본은 아무것도 반환하지 않아
    #       재개 지점을 파일명 정규식으로 파내야 했음.
    #       이 키들이 없는 옛 체크포인트는 None 이 나오므로 호출부가 폴백하면 됨
    return {k: states.get(k) for k in EXTRA_KEYS}


def save_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.amp.GradScaler],    # <<<<< 고친 것 - torch.cuda.amp.GradScaler 의 부모 클래스. 둘 다 받음
    path: str,
    # <<<<< 더한 것 - 학습 상태. 전부 기본값이 있어 기존 호출부(train_gan.py)는 안 고쳐도 됨
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
):
    if isinstance(models[0], torch.nn.DataParallel):
        state_dict = [model.module.state_dict() for model in models]
    elif isinstance(models[0], torch.nn.parallel.DistributedDataParallel):
        state_dict = [model.module.state_dict() for model in models]
    else:
        state_dict = [model.state_dict() for model in models]
    torch.save(
        {
            "models":
            state_dict,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers":
            [s.state_dict() if s is not None else None for s in schedulers],
            "scaler":
            scaler.state_dict() if scaler is not None else None,
            # <<<<< 더한 것 - 이 판이 어느 시점의 것이고 성능이 얼마였는지를 함께 남김
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            # str() 로 감싸야 함 - TorchVersion 객체 그대로 담으면
            # torch.load 의 weights_only=True 기본값(2.6+)이 통째로 거부함(실측)
            "torch_version": str(torch.__version__),
        },
        path,
    )
