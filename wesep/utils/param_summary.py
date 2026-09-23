# <<<<< 더한 것 - 모델 파라미터를 묶음별로 세어 표로 찍음 (#95).
#       원본 wesep 은 train.py 에서 전체 합 한 줄만 찍었다. 조건화 모듈을 바꿔 가며
#       비교하는 실험에서는 "어디가 얼마나 늘었나" 가 보여야 하므로 묶음으로 가른다.
"""BSRNN 계열 모델의 파라미터를 묶음별로 센다.

    logger.info(f"\\n{pd.DataFrame(count_params(model)).to_string(index=False)}")
    tracker.log_hyperparams(param_metrics(count_params(model)))

묶음은 겹치지도 빠지지도 않는다 - next() 가 파라미터마다 한 묶음만 고르고 안 걸리면
separation 으로 가므로, 합은 구조상 항상 `total` 과 같다.

`cond` 는 **SD-FiLM 저장소의 `cond_module` 열과 같은 뜻**이다 - src/utils/param_flops.py 의
`SUMMARY_MODULE_CLASSES = {"cond_module": PosNegFiLM}` 이 isinstance 로 **바꿔 끼우는
조건화 모듈 그 자체만** 센다. 여기서도 `SpeakerFuseLayer` 인 것만 센다. FiLM 칸과 SD-FiLM 칸의
차이를 재는 것이 목적이라, 두 칸에 공통인 것(`spk_transform`)을 섞으면 그 차이가 희석된다.

`separation` 은 `BSRNN.separation` 속성이 아니라 **나머지 전부**다 - `mask`·`BN` 처럼
`separator` 밖에 있는 것도 들어간다. 조건화만 바꾼 비교에서는 이 값이 안 변해야 맞고,
실제로 config 5개 x 설정 3가지에서 21,402,376 으로 같다(실측).

FLOPs 는 세지 않는다. torch 의 FlopCounterMode 는 연산별 공식표를 보는데 그 표에 순환 연산이
없어 `nn.LSTM` 을 0 으로 센다(torch 2.7.1 · 2.11.0 실측). BSRNN 은 LSTM 12개가 주 연산이다.
thop 은 LSTM 을 세지만 `profile()` 이 state_dict 에 키 390개를 남겨 체크포인트를 깨뜨린다(실측).
"""
import torch.nn as nn

from wesep.modules.common.speaker import SpeakerFuseLayer

# 화자 인코더. spk_model_freeze: True 면 전부 requires_grad=False 가 된다.
SPK_MODEL_PREFIX = "spk_model"

# 화자 임베딩을 융합 전에 변환하는 모듈. use_spk_transform: False 면 nn.Identity 라 0 개이고,
# True 면 SpeakerTransform 이 82,432 개가 된다(실측). cond 에도 separation 에도 안 넣는다 -
# cond 에 넣으면 두 칸에 공통인 값이 섞이고, separation 에 넣으면 통제변인이 움직인다.
# 0 일 때는 줄을 만들지 않으므로 평소 출력은 네 줄이다.
SPK_TRANSFORM_PREFIX = "spk_transform"


def count_params(model: nn.Module) -> list:
    """묶음별 파라미터 수를 list[dict] 로 돌려준다. 그대로 pd.DataFrame() 에 넣을 수 있다.

    행 하나가 dict 하나다 - 이 저장소가 이미 쓰는 꼴과 같다(infer.py 의 utt_rows).

    :param model: DistributedDataParallel 로 감싸기 **전** 이어야 한다 - 감싸면 이름에
        `module.` 이 붙어 묶음 판정이 어긋나 spk_model 이 0 이 된다(실측).
    """
    # 조건화 모듈은 이름이 아니라 클래스로 찾는다. multi_fuse 값에 따라 층이 1개이거나 6개라
    # `separator.separation.0` 같은 고정 문자열은 5개를 놓친다. 이름은 리팩터로 조용히 바뀌지만
    # 클래스가 바뀌면 import 에서 바로 드러난다 - SD-FiLM 의 param_flops.py 와 같은 이유다.
    cond_prefixes = tuple(name for name, module in model.named_modules()
                          if isinstance(module, SpeakerFuseLayer))
    # 앞에서부터 먼저 걸리는 묶음으로 간다. 어디에도 안 걸리면 separation.
    rules = ((SPK_MODEL_PREFIX, (SPK_MODEL_PREFIX,)),
             (SPK_TRANSFORM_PREFIX, (SPK_TRANSFORM_PREFIX,)),
             ("cond", cond_prefixes))
    tallies = {name: [0, 0] for name in
               ("total", "separation", SPK_MODEL_PREFIX, "cond", SPK_TRANSFORM_PREFIX)}

    for param_name, param in model.named_parameters():
        # 점까지 보고 판정한다 - 안 그러면 `spk_model` 이 `spk_model_extra` 까지 잡는다.
        # 접두사는 전부 모듈 경로이고 파라미터 이름은 그 뒤에 속성 이름이 한 칸 더 붙으므로
        # 접두사와 완전히 같아지는 경우는 없다 - startswith 만 보면 된다(실측).
        group = next((name for name, prefixes in rules
                      if any(param_name.startswith(p + ".") for p in prefixes)),
                     "separation")
        for key in (group, "total"):
            tallies[key][0] += param.numel()
            tallies[key][1] += param.numel() if param.requires_grad else 0

    total = tallies["total"][0]
    shown = ["total", "separation", SPK_MODEL_PREFIX, "cond"]
    if tallies[SPK_TRANSFORM_PREFIX][0]:
        shown.append(SPK_TRANSFORM_PREFIX)
    rows = []
    for name in shown:
        count, trainable = tallies[name]
        rows.append({
            "group": name,
            "params": count,
            "trainable": trainable,
            "frozen": count - trainable,
            "share(%)": round(100.0 * count / total, 2) if total else 0.0
        })
    return rows


def param_metrics(rows: list) -> dict:
    """count_params() 의 결과를 Tracker.log_hyperparams() 에 넣을 납작한 dict 로 바꾼다.

    키 이름은 SD-FiLM 의 wandb hparams(`model/params/total` 등)와 같은 꼴이다 -
    텐서보드가 `/` 앞을 그룹으로 묶어 `model` 아래로 접힌다.
    파라미터 수는 학습 내내 안 변하므로 지표가 아니라 하이퍼파라미터로 보낸다.
    묶음별 trainable 은 0 이거나 params 와 같아 정보가 없으므로 전체만 따로 남긴다.
    """
    flat = {}
    for row in rows:
        flat[f"model/params/{row['group']}"] = row["params"]
        if row["group"] == "total":
            flat["model/params/trainable"] = row["trainable"]
            flat["model/params/non_trainable"] = row["frozen"]   # SD-FiLM logging_utils.py 와 같은 키
    return flat
