"""BSRNN 컴파일 비용·스텝 속도 측정의 공통 코드.

노트북 두 개와 워커 스크립트가 **같은 코드**를 쓰도록 여기에 모음.

  · bsrnn_compile_cost_by_structure.ipynb — 구조를 바꿔 컴파일 비용의 원인을 찾음
  · bsrnn_compile_axes_yaml.ipynb         — yaml 설정 그대로 두고 돌리는 방식을 흔듦
  · _bench/worker.py                      — GPU 여러 장에 조건을 나눠 재는 워커

측정 결과는 **`_runs/results.csv` 한 개**에 모임 — 조건 하나가 한 줄.
워커가 여러 개여도 flock 으로 잠그고 append 하므로 섞이지 않음.
노트북은 그 CSV 를 **읽기만** 하므로 GPU 없이도 표가 그려짐.
"""
import csv
import fcntl
import gc
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Inductor 캐시를 매 실행 새 폴더로 — 컴파일 시간을 항상 콜드에서 재기 위함.
# torch 를 import 하기 전에 설정해야 함.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="inductor_nb_"))

import pandas as pd
import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
import yaml


# ════════════════════════════════════════════════════════ 경로 · 장치
def find_wesep_root(start: Path) -> Path:
    """`wesep/__init__.py` 를 담은 폴더를 위로 올라가며 찾음 — 그것이 wesep 클론 루트임."""
    for d in (start, *start.parents):
        if (d / "wesep" / "__init__.py").is_file():
            return d
    raise FileNotFoundError(f"{start} 위쪽에 wesep 클론 루트가 없음")


WESEP = find_wesep_root(Path(__file__).resolve())
if str(WESEP) not in sys.path:
    sys.path.insert(0, str(WESEP))

from wesep.models import get_model                      # noqa: E402
from wesep.utils.funcs import clip_gradients            # noqa: E402
from wesep.utils.losses import parse_loss               # noqa: E402

# 장치를 여기서 한 번만 정하고 아래 전부가 이것을 씀 — "cuda" 를 여기저기 적지 않으려는 것임.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

RUNS = Path(__file__).resolve().parent.parent / "_runs"   # wesep/notebooks/_runs


def sync():
    if IS_CUDA:
        torch.cuda.synchronize()


def empty_cache():
    if IS_CUDA:
        torch.cuda.empty_cache()


def reset_peak():
    if IS_CUDA:
        torch.cuda.reset_peak_memory_stats()


def peak_gib():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if IS_CUDA else float("nan")


def env_table():
    return pd.DataFrame([
        {"항목": "torch", "값": torch.__version__, "비고": f"cuda {torch.version.cuda}"},
        {"항목": "DEVICE", "값": str(DEVICE), "비고": "모든 측정이 이 장치에서 돎"},
        {"항목": "GPU", "값": torch.cuda.get_device_name(0) if IS_CUDA else "(없음)",
         "비고": "{:.1f} GiB".format(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
                 if IS_CUDA else "CPU 로 돎"},
        {"항목": "CUDA_VISIBLE_DEVICES", "값": os.environ.get("CUDA_VISIBLE_DEVICES", "(전부)"),
         "비고": "이 GPU 에 다른 프로세스가 있으면 측정이 오염됨"},
        {"항목": "wesep 루트", "값": str(WESEP), "비고": "sys.path 에 넣음"},
        {"항목": "결과 CSV", "값": str(RESULTS_CSV), "비고": "조건 하나가 한 줄"},
    ])


# ════════════════════════════════════════════════════════ yaml 설정
CONF_PATH = WESEP / "examples/librimix/tse/v2/confs/bsrnn_ecapa_FiLM.yaml"
ECAPA_CKPT = WESEP / "examples/librimix/tse/v2/wespeaker_models/voxceleb_ECAPA512/avg_model.pt"

conf = yaml.safe_load(CONF_PATH.read_text())

# yaml 의 model_args.tse_model 그대로. spk_model_init 만 절대경로로 바꿈
# (yaml 의 ./wespeaker_models/... 는 examples/librimix/tse/v2 기준 상대경로라 여기서는 못 찾음).
ARGS = dict(conf["model_args"]["tse_model"])
ARGS["spk_args"] = dict(ARGS["spk_args"])
ARGS["spk_model_init"] = str(ECAPA_CKPT)

YAML_B = conf["dataloader_args"]["batch_size"] * 2     # tse_collate_fn 이 화자 2명 몫으로 폄
YAML_T = conf["dataset_args"]["chunk_len"]
YAML_NREP = ARGS["num_repeat"]
YAML_FDIM = ARGS["feature_dim"]
F_MEL = conf["dataset_args"]["fbank_args"]["num_mel_bins"]
CLIP_GRAD = conf["clip_grad"]
LR = conf["scheduler_args"]["tse_model"]["initial_lr"]
WEIGHT_DECAY = conf["optimizer_args"]["tse_model"]["weight_decay"]
SEED = conf["seed"]


def conf_table():
    return pd.DataFrame([
        {"항목": "dataloader_args.batch_size", "값": conf["dataloader_args"]["batch_size"],
         "뜻": "혼합 개수. 모델이 보는 배치는 이것의 2배"},
        {"항목": "모델 배치 (실효)", "값": YAML_B, "뜻": "tse_collate_fn 이 편 뒤의 b"},
        {"항목": "dataset_args.chunk_len", "값": YAML_T,
         "뜻": "입력 길이 t (샘플) = {:.1f} 초".format(
             YAML_T / conf["dataset_args"]["resample_rate"])},
        {"항목": "model_args.num_repeat", "값": YAML_NREP,
         "뜻": "BSNet 반복 수. LSTM 개수는 이것의 2배"},
        {"항목": "model_args.feature_dim", "값": YAML_FDIM, "뜻": "밴드당 채널 c"},
        {"항목": "fbank num_mel_bins", "값": F_MEL, "뜻": "등록 발화 fbank 의 f"},
        {"항목": "clip_grad", "값": CLIP_GRAD, "뜻": "executor.py 가 매 스텝 부름"},
        {"항목": "ECAPA 가중치 존재", "값": ECAPA_CKPT.is_file(), "뜻": str(ECAPA_CKPT)},
    ])


# ════════════════════════════════════════════════════════ wesep 데이터로더 모사
#
# tse_collate_fn(dataset.py:206-264) 이 하는 일 중 shape 에 관계된 것만 옮김.
#
#   1. 샘플 하나마다 화자 수(2명)만큼 wav_mix 를 복제 -> 모델 배치 b = yaml batch_size x 2
#   2. wav_mix 는 chunk_len 으로 이미 잘려 있음        -> separator 입력은 항상 고정
#   3. spk_embeds(등록 발화 fbank)는 화자마다 길이가 다르고
#      mode="min" 이 배치 안 최소 길이로 잘라냄        -> 배치마다 T 가 달라짐
#
# 그래서 separator 는 정적 shape, spk_model(ECAPA)은 동적 shape 을 받음. 이것이 컴파일의 핵심 조건임.
#
# 주의 — 개별 발화 길이의 분포는 **근사이고 미검증**임. 실측한 것은 b=16 일 때
# min 의 분포(299~1005, 중앙값 396)뿐이라, 개별 길이를 [299, 1950] 균등으로 잡아
# 그 min 이 실측과 맞도록 역산했음.
ENROLL_MIN, ENROLL_MAX = 299, 1950


def _lcg(seed):
    """재현 가능한 의사난수. random 모듈을 쓰지 않아 몇 번을 돌려도 같은 값이 나옴."""
    x = seed
    while True:
        x = (1103515245 * x + 12345) % (1 << 31)
        yield x / (1 << 31)


def enroll_len_for_batch(b, step_idx):
    """배치 하나의 등록 길이 = 화자 b명의 개별 길이 중 최소값 (mode="min")."""
    g = _lcg(1000 + step_idx * 7919)
    return min(int(ENROLL_MIN + (ENROLL_MAX - ENROLL_MIN) * next(g)) for _ in range(b))


def make_wesep_batch(b, t, step_idx):
    """한 스텝 분량의 (wav_mix (b,t), spk_embeds (b,T_min,f), wav_targets (b,t))."""
    L = enroll_len_for_batch(b, step_idx)
    return (torch.randn(b, t, device=DEVICE),
            torch.randn(b, L, F_MEL, device=DEVICE),
            torch.randn(b, t, device=DEVICE))


def enroll_probe_table(batches=(1, 2, 4, 8, 16), n=60):
    """배치 크기별로 등록 길이가 어떻게 나오는지 — b 가 클수록 min 이 작아지는 것이 실제 경향임."""
    return pd.DataFrame([
        {"b": b,
         "등록 T 최소": min(enroll_len_for_batch(b, i) for i in range(n)),
         "등록 T 중앙값": int(statistics.median(enroll_len_for_batch(b, i) for i in range(n))),
         "등록 T 최대": max(enroll_len_for_batch(b, i) for i in range(n)),
         f"고유 개수({n}스텝)": len({enroll_len_for_batch(b, i) for i in range(n)})}
        for b in batches])


# ════════════════════════════════════════════════════════ 측정
N_WARMUP, N_STEPS = 10, 50      # 웜업 10 · 측정 50

criterion = parse_loss("SISDR")          # [auraloss.time.SISDRLoss()]

# precision 축. (autocast 를 켜나 · 어떤 dtype 으로 · GradScaler 를 쓰나)
PRECISIONS = {
    "fp32":       {"amp": False, "dtype": None,           "scaler": False},
    "fp16-mixed": {"amp": True,  "dtype": torch.float16,  "scaler": True},
    "bf16-mixed": {"amp": True,  "dtype": torch.bfloat16, "scaler": False},
}


def build_bsrnn(**over):
    """yaml 의 ARGS 에서 몇 개만 바꿔 BSRNN 을 새로 만듦."""
    args = dict(ARGS)
    args.update(over)
    torch.manual_seed(SEED)
    m = get_model("BSRNN")(**args).to(DEVICE)
    m.train()
    return m


def _scope_eager(m, dynamic):
    pass


def _scope_separator(m, dynamic):
    m.separator.compile(dynamic=dynamic)


def _scope_spk_only(m, dynamic):
    m.spk_model.compile(dynamic=dynamic)


def _scope_sep_spk(m, dynamic):
    m.separator.compile(dynamic=dynamic)     # 입력이 고정이라 dynamic 을 꺼도 됨
    m.spk_model.compile(dynamic=dynamic)     # 입력 길이가 매번 달라 dynamic 이 필요함


def _scope_model(m, dynamic):
    m.compile(dynamic=dynamic)               # 한꺼번에 1개로


SCOPES = {
    "eager":         _scope_eager,
    "separator":     _scope_separator,
    "spk_only":      _scope_spk_only,
    "separator&spk": _scope_sep_spk,
    "model":         _scope_model,
}


def train_step(model, optimizer, scaler, batch, prec):
    """executor.py:142-214 의 한 스텝을 그대로 옮긴 것.

    loss.item() 만 뺐음 — GPU 동기화를 강제해 스텝 시간 측정을 왜곡하기 때문임.
    """
    x, e, y = batch
    cfg = PRECISIONS[prec]
    with torch.amp.autocast(DEVICE.type, enabled=cfg["amp"], dtype=cfg["dtype"]):
        outputs = model(x, e)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        loss = 1.0 * criterion[0](outputs[0], y).mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_gradients(model, CLIP_GRAD)
    scaler.step(optimizer)
    scaler.update()


def _fail_row(status, ex):
    """측정이 안 된 조건의 행. 무엇 때문에 안 됐는지는 error 열에 한 줄로 남김."""
    msg = f"{type(ex).__name__}: {ex}".replace("\n", " ")
    return {"compile_s": float("nan"), "ms_step_median": float("nan"),
            "ms_step_mean": float("nan"), "peak_GiB": float("nan"),
            "graph_break": 0, "dynamo_frames": 0, "recompile_in_measure": 0,
            "status": status, "error": msg[:300]}


def dynamo_stats():
    """컴파일이 얼마나 쪼개졌는지. frames 와 graph break 는 서로 다른 값임."""
    c = dynamo.utils.counters
    return (int(c.get("frames", {}).get("total", 0)),
            int(sum(c.get("graph_break", {}).values())))


def bench(scope="eager", dynamic=False, b=None, t=None, prec="fp16-mixed",
          tf32_lstm=True, num_repeat=None, feature_dim=None, allow_rnn=False):
    """한 조건을 재고 행 하나를 돌려줌.

    통제변인(scope · dynamic · b · t · prec · tf32_lstm · num_repeat · feature_dim · allow_rnn)은 왼쪽,
    측정값(compile_s · ms_step_median · ms_step_mean · peak_GiB · graph_break · dynamo_frames
    · recompile_in_measure · status)은 오른쪽에 둠.
    """
    b = YAML_B if b is None else b
    t = YAML_T if t is None else t
    num_repeat = YAML_NREP if num_repeat is None else num_repeat
    feature_dim = YAML_FDIM if feature_dim is None else feature_dim

    dynamo.reset()
    dynamo.utils.counters.clear()
    dynamo.config.cache_size_limit = 32
    dynamo.config.allow_rnn = allow_rnn
    inductor_config.force_disable_caches = True        # 조건마다 콜드 컴파일을 보장
    torch.backends.cudnn.allow_tf32 = tf32_lstm        # LSTM 은 cuDNN 경로임
    empty_cache()
    reset_peak()

    row = {"scope": scope, "dynamic": "-" if scope == "eager" else dynamic,
           "b": b, "t": t, "prec": prec, "tf32_lstm": tf32_lstm,
           "num_repeat": num_repeat, "feature_dim": feature_dim,
           "lstm_n": num_repeat * 2, "allow_rnn": allow_rnn,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}
    try:
        model = build_bsrnn(num_repeat=num_repeat, feature_dim=feature_dim)
        SCOPES[scope](model, dynamic)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=PRECISIONS[prec]["scaler"])

        # 첫 스텝 = 컴파일이 실제로 일어나는 곳
        sync()
        t0 = time.perf_counter()
        train_step(model, optimizer, scaler, make_wesep_batch(b, t, 0), prec)
        sync()
        # eager 는 컴파일 단계가 없으므로 0.0 — "컴파일에 쓴 시간이 0초" 라는 뜻임
        row["compile_s"] = round(time.perf_counter() - t0, 1) if scope != "eager" else 0.0

        # 웜업 — 등록 길이가 매 스텝 달라지므로 여기서 재컴파일을 모두 끝냄
        for i in range(1, N_WARMUP + 1):
            train_step(model, optimizer, scaler, make_wesep_batch(b, t, i), prec)
        sync()
        warm_frames, _ = dynamo_stats()

        per_step = []
        for i in range(N_WARMUP + 1, N_WARMUP + 1 + N_STEPS):
            batch = make_wesep_batch(b, t, i)
            sync()
            s = time.perf_counter()
            train_step(model, optimizer, scaler, batch, prec)
            sync()
            per_step.append((time.perf_counter() - s) * 1000)

        frames, breaks = dynamo_stats()
        row.update({
            "compile_s": row["compile_s"],
            "ms_step_median": round(statistics.median(per_step), 1),
            "ms_step_mean": round(statistics.mean(per_step), 1),
            "peak_GiB": round(peak_gib(), 2),
            "graph_break": breaks,
            "dynamo_frames": frames,
            # 웜업 뒤에도 늘었다면 측정 구간에 재컴파일이 섞인 것임
            "recompile_in_measure": frames - warm_frames,
            "status": "ok", "error": "",
        })
        del model, optimizer, scaler
    except torch.OutOfMemoryError as ex:
        row.update(_fail_row("OOM", ex))
    except Exception as ex:
        # 컴파일 실패도 결과의 하나임 — 그 조건만 ERROR 로 남기고 다음 조건으로 넘어감.
        # 예: dynamic=True 는 GroupNorm 역전파 분해에서 SymInt divmod 로 죽음 (torch 2.7.1)
        row.update(_fail_row("ERROR", ex))
    finally:
        inductor_config.force_disable_caches = False
        dynamo.config.allow_rnn = False
        torch.backends.cudnn.allow_tf32 = True         # 기본값으로 되돌림
        gc.collect()
        empty_cache()
        reset_peak()
    return row


# ════════════════════════════════════════════════════════ 조건 목록
#
# 워커와 노트북이 **같은 목록**을 봐야 결과 파일이 맞아떨어지므로 여기서 한 번만 정의함.
SCOPE_DYNAMIC = ([("eager", False)] +
                 [(s, d) for s in ("separator", "separator&spk", "model") for d in (False, True)])
BATCHES = (1, 2, 4, 8, 16)
PRECS = ("fp32", "bf16-mixed", "fp16-mixed")
TF32 = (True, False)
DEPTHS = (1, 2, 4, 6, 8)
WIDTHS = (64, 128, 256)


def conditions(tag):
    """tag 하나에 딸린 조건 목록. 순서가 바뀌면 안 되므로 정렬된 형태로 만듦."""
    if tag == "structure-depth":
        return [dict(scope=s, dynamic=True, num_repeat=n)
                for n in DEPTHS for s in ("eager", "separator", "separator&spk")]
    if tag == "structure-width":
        return [dict(scope=s, dynamic=True, feature_dim=f)
                for f in WIDTHS for s in ("eager", "separator")]
    if tag == "structure-who":
        return [dict(scope=s, dynamic=True)
                for s in ("eager", "separator", "spk_only", "separator&spk", "model")]
    if tag == "structure-allow-rnn":
        return [dict(scope="separator", dynamic=True, num_repeat=n, allow_rnn=a)
                for a in (False, True) for n in (1, 2)]
    if tag == "axes-base":
        return [dict(scope=s, dynamic=d) for s, d in SCOPE_DYNAMIC]
    if tag == "axes-single":
        out = [dict(scope="separator&spk", dynamic=True, b=b) for b in BATCHES]
        out += [dict(scope="separator&spk", dynamic=True, prec=p) for p in PRECS]
        out += [dict(scope="separator&spk", dynamic=True, tf32_lstm=tf) for tf in TF32]
        return out
    if tag == "dynamic-none-scopes":
        # dynamic 생략(None)을 model 말고 다른 scope 에서도 확인 —
        # model 에서는 None 이 True 보다 빨랐는데(실측), 부분 컴파일에서도 그런지는 몰랐음.
        # b=1 은 BatchNorm 이 배치 1을 거부해 어떤 방식으로도 실패하므로 뺌.
        return [dict(scope=sc, dynamic=None, b=b, prec=pr, tf32_lstm=True)
                for sc in ("separator", "separator&spk")
                for b in (2, 4, 8, 16) for pr in PRECS]
    if tag == "dynamic-none-axes":
        # dynamic 기본값(None)만 배치 x precision 으로 펼침 —
        # eager / True / False 는 axes-full 이 이미 같은 조건에서 재 두었으므로 다시 재지 않음.
        return [dict(scope="model", dynamic=None, b=b, prec=p, tf32_lstm=True)
                for b in (2, 4, 8, 16) for p in PRECS]
    if tag == "dynamic-none":
        # .compile() 의 dynamic 기본값은 False 가 아니라 None 임 — 처음엔 정적으로 잡고
        # 크기가 바뀌면 동적으로 전환하는 제3의 동작이라 True/False 값에서 유추할 수 없음.
        # 비교가 되도록 eager 기준선까지 같은 GPU 에서 연달아 잼.
        return ([dict(scope="eager", dynamic=False)] +
                [dict(scope="model", dynamic=d) for d in (None, True, False)])
    if tag == "axes-full":
        return [dict(scope=s, dynamic=d, b=b, prec=p, tf32_lstm=tf)
                for s, d in SCOPE_DYNAMIC for b in BATCHES for p in PRECS for tf in TF32]
    raise KeyError(f"모르는 tag: {tag}")


ALL_TAGS = ("structure-depth", "structure-width", "structure-who", "structure-allow-rnn",
            "axes-base", "axes-single", "axes-full", "dynamic-none", "dynamic-none-axes", "dynamic-none-scopes")


# ════════════════════════════════════════════════════════ 저장 · 적재
# 표의 열 순서 — 왼쪽은 통제변인, 오른쪽은 측정값
LEFT = ["scope", "dynamic", "b", "t", "prec", "tf32_lstm", "num_repeat", "feature_dim",
        "lstm_n", "allow_rnn", "gpu"]
RIGHT = ["compile_s", "ms_step_median", "ms_step_mean", "peak_GiB", "graph_break",
         "dynamo_frames", "recompile_in_measure", "status", "error"]
def _cond_key(cond):
    """조건 dict -> 파일 이름. 키를 정렬해 만들므로 같은 조건이면 항상 같은 이름이 나옴."""
    full = {"scope": "eager", "dynamic": False, "b": YAML_B, "t": YAML_T,
            "prec": "fp16-mixed", "tf32_lstm": True, "num_repeat": YAML_NREP,
            "feature_dim": YAML_FDIM, "allow_rnn": False}
    full.update(cond)
    if full["scope"] == "eager":
        full["dynamic"] = "-"                      # eager 는 dynamic 이 뜻이 없음
    s = "|".join(f"{k}={full[k]}" for k in sorted(full))
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _group_key(cond):
    """같은 GPU 에서 재야 하는 묶음. scope·dynamic 만 다른 조건은 한 묶음임 —
    speedup 의 기준이 되는 eager 와 떨어지면 비교가 깨지기 때문임."""
    full = {"b": YAML_B, "t": YAML_T, "prec": "fp16-mixed", "tf32_lstm": True,
            "num_repeat": YAML_NREP, "feature_dim": YAML_FDIM, "allow_rnn": False}
    full.update({k: v for k, v in cond.items() if k in full})
    return tuple(full[k] for k in sorted(full))


# 결과는 CSV 한 개에 모음. 워커 여러 개가 동시에 append 하므로 flock 으로 잠그고 씀.
RESULTS_CSV = RUNS / "results.csv"
FIELDS = ["tag", "cond_key"] + LEFT + RIGHT


def _read_csv():
    """지금까지 쌓인 결과 전체. 파일이 없으면 빈 DataFrame."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=FIELDS)
    with open(RESULTS_CSV, newline="") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        df = pd.read_csv(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return df


_cache = {"mtime": None, "rows": {}}


def _rows_by_key():
    """(tag, cond_key) -> 행 dict. 파일이 바뀌었을 때만 다시 읽음."""
    if not RESULTS_CSV.exists():
        return {}
    mtime = RESULTS_CSV.stat().st_mtime
    if _cache["mtime"] != mtime:
        df = _read_csv()
        _cache["rows"] = {(r["tag"], r["cond_key"]): r.to_dict() for _, r in df.iterrows()}
        _cache["mtime"] = mtime
    return _cache["rows"]


def load(tag, cond):
    return _rows_by_key().get((tag, _cond_key(cond)))


def save(tag, cond, row):
    """CSV 한 줄 append. 여러 워커가 같이 써도 안전하도록 배타 잠금을 검."""
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = {"tag": tag, "cond_key": _cond_key(cond), **row}
    with open(RESULTS_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if f.tell() == 0:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in FIELDS})
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    _cache["mtime"] = None      # 다음 load 때 다시 읽도록


def run_or_load(tag, measure_missing=False, worker=None, n_workers=1, verbose=True):
    """조건 목록을 돌며 저장된 결과를 읽음. 없으면 (measure_missing 일 때) 잼.

    노트북은 measure_missing=False 로 부름 — GPU 없이 표만 그리려는 것임.
    워커는 measure_missing=True 와 worker/n_workers 를 줌.
    """
    conds = conditions(tag)
    # 배분은 **아직 안 잰 조건**만 놓고 함 — 이미 잰 것까지 세면 워커마다 몫이 크게 기울어짐
    # (다시 띄웠을 때 한 워커에 남은 일이 몰리는 것을 막으려는 것임).
    todo = [c for c in conds if load(tag, c) is None]
    groups = sorted({_group_key(c) for c in (todo or conds)})
    mine = {g for i, g in enumerate(groups) if worker is None or i % n_workers == worker}

    rows, missing = [], 0
    for cond in conds:
        row = load(tag, cond)
        if row is not None:
            rows.append(row)
            continue
        if not measure_missing or _group_key(cond) not in mine:
            missing += 1
            continue
        row = bench(**cond)
        save(tag, cond, row)
        rows.append(row)
        if verbose:
            print(" | ".join(f"{k}={row[k]}" for k in
                             ("scope", "dynamic", "b", "prec", "tf32_lstm", "num_repeat",
                              "compile_s", "ms_step_median", "status")), flush=True)
    if verbose and missing:
        print(f"[{tag}] 아직 측정 안 된 조건 {missing}개", flush=True)
    return rows


def as_table(rows, keep=()):
    """통제변인은 왼쪽, 측정값은 오른쪽. 값이 하나뿐인 통제변인 열은 접어서 감춤."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    left = [c for c in LEFT
            if c in df.columns and (c in keep or c == "scope" or df[c].nunique() > 1)]
    right = [c for c in RIGHT if c in df.columns]
    return df[left + right]


def progress_table():
    """태그별로 몇 개나 쟀는지 — 워커가 도는 중에도 노트북에서 확인 가능함."""
    out = []
    for tag in ALL_TAGS:
        conds = conditions(tag)
        done = sum(load(tag, c) is not None for c in conds)
        out.append({"tag": tag, "조건 수": len(conds), "측정됨": done,
                    "남음": len(conds) - done,
                    "진행": f"{done / len(conds) * 100:.0f} %"})
    return pd.DataFrame(out)


def all_results():
    """results.csv 전체를 그대로 — 왼쪽 통제변인, 오른쪽 측정값 순으로."""
    df = _read_csv()
    if df.empty:
        return df
    cols = ["tag"] + [c for c in LEFT if c in df.columns] + [c for c in RIGHT if c in df.columns]
    return df[cols]
