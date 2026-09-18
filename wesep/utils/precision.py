"""학습 정밀도를 Lightning 과 같은 이름으로 고르게 함. 이 포크에서 더한 파일임.

    amp_enabled, amp_dtype, scaler_enabled = parse_precision(configs)

config 의 `precision` 하나가 autocast 와 GradScaler 를 **둘 다** 결정함.

    precision        autocast        dtype       GradScaler
    32-true          꺼짐            -           꺼짐
    16-mixed         켜짐            float16     켜짐
    bf16-mixed       켜짐            bfloat16    **꺼짐**

**bf16 에 GradScaler 를 쓰지 않는 이유** — bf16 은 지수부가 8비트로 fp32 와 같아
작은 기울기가 0 으로 죽지 않음. fp16 은 지수부가 5비트라 언더플로가 나서 스케일링이
필요함. Lightning 도 같은 판단이고, bf16-mixed 에 scaler 를 넘기면 아예 막음 —
lightning/pytorch/plugins/precision/amp.py:55-56 의
`"bf16-mixed" does not use a scaler` MisconfigurationException.

**옛 키 `enable_amp` 은 precision 이 없을 때만 봄.** wesep 원본 config 11개가
그 키를 쓰므로, 그것들을 안 고쳐도 돌게 하려는 것임.
다만 examples/librimix/tse/v2/run.sh 는 --precision 을 **항상** 넘기므로,
그 폴백이 실제로 쓰이는 곳은 v1/run.sh 와 train.py 직접 호출뿐임.

    precision 있음         precision 대로 (enable_amp 은 무시)
    없음 + enable_amp false  32-true
    없음 + enable_amp true   16-mixed
"""
import torch


# precision -> (autocast,   autocast dtype, GradScaler)
# precision -> (enable_amp, amp_dtype,      scaler_enabled)
PRECISION_TABLE = {
    "32-true":      (False, torch.float32,  False),
    "bf16-mixed":   (True,  torch.bfloat16, False),
    "16-mixed":     (True,  torch.float16,  True),
}
PRECISION_LIST = list(PRECISION_TABLE.keys())


def parse_precision(configs):
    """configs 에서 precision 을 읽어 (amp_enabled, amp_dtype, scaler_enabled) 로 품.

    precision 이 없으면 옛 키 enable_amp 으로 떨어짐.
    """
    precision = configs.get("precision")
    if precision is None:
        if configs.get("enable_amp", False):
            precision = "16-mixed"
        else:
            precision = "32-true"

    if precision not in PRECISION_TABLE:
        # config yaml 이나 run.sh --precision 이 넘긴 값임.
        # 오타면 조용히 fp32 로 돌아 논문 수치가 달라지므로 멈춤
        raise ValueError(f"precision={precision!r} not in {PRECISION_LIST}")
    return PRECISION_TABLE[precision]
