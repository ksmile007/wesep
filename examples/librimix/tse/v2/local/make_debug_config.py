#!/usr/bin/env python3
"""본 config 에 debug 덮어쓰기를 얹어 임시 config 를 만듦 — run.sh 의 --debug true 에서만 쓰임.

    python local/make_debug_config.py <본 config> <debug 덮어쓰기> <내보낼 경로>

내보낸 경로를 표준출력으로 찍으므로 run.sh 가 그대로 받아 쓴다.

`steps_per_epoch` 만 특별 취급함 — train.py 가 에포크당 스텝을 직접 받지 않고
sample_num_per_epoch // world_size // batch_size 로 계산하기 때문에 batch_size 를 곱해 넣는다.

본 config 는 yaml 로 직접 읽지 않고 load_config_with_base() 로 **먼저 합성**한다 (#83).
hydra 의 defaults 를 쓰는 config 은 자기 폴더에 형제 yaml 이 있어야 풀리는데,
내보낼 곳은 exp_dir 이라 그 형제가 없어 MissingConfigException 이 난다.
합성해서 내보내면 그 파일 하나로 완결되므로 어느 폴더에 둬도 읽힌다.
"""
import sys

import yaml

from wesep.utils.utils import load_config_with_base


def deep_update(base: dict, overlay: dict) -> dict:
    """overlay 의 키만 base 에 덮어씀. dict 는 재귀로 들어가 나머지 키를 지키지 않고 살림."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def main(base_path: str, overlay_path: str, out_path: str) -> None:
    base = load_config_with_base(base_path)   # defaults 를 합성해 평평하게 만든다
    overlay = yaml.safe_load(open(overlay_path)) or {}

    steps = overlay.pop("steps_per_epoch", None)
    val_steps = overlay.pop("val_steps_per_epoch", None)
    deep_update(base, overlay)
    batch_size = base["dataloader_args"]["batch_size"]
    if steps is not None:
        base.setdefault("dataset_args", {})["sample_num_per_epoch"] = steps * batch_size
    if val_steps is not None:
        base.setdefault("dataset_args", {})["val_sample_num_per_epoch"] = val_steps * batch_size

    with open(out_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
    print(out_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
