"""stat_* run 들의 train.log 를 읽어 CSV 한 파일로 모음.

    python notebooks/_bench/collect_jitter.py

왼쪽에 통제변인(gpu · rep · tracker · cudnn_benchmark), 오른쪽에 측정값을 둠.
"""
import csv
import datetime as dt
import re
from pathlib import Path

V2 = Path(__file__).resolve().parents[2] / "examples/librimix/tse/v2"
OUT = Path(__file__).resolve().parents[1] / "_runs/cudnn_benchmark_jitter.csv"

LEFT = ["tag", "gpu", "rep", "tracker", "cudnn_benchmark", "compile_model"]
RIGHT = ["ep1_train", "ep2_train", "ep3_train",
         "ep1_val", "ep2_val", "ep3_val",
         "sec_ep1", "sec_ep2", "sec_ep3"]


def parse(log: Path):
    t = log.read_text(encoding="utf-8", errors="ignore")
    tr = [float(x) for x in re.findall(r"Train info train_loss ([\d.eE+-]+)", t)]
    va = [float(x) for x in re.findall(r"Val info val_loss ([\d.eE+-]+)", t)]
    ts = [dt.datetime.strptime(m, "%Y-%m-%d %H:%M:%S,%f") for m in
          re.findall(r"\[ INFO : ([\d-]+ [\d:,]+) \].*Train info", t)]
    st = re.search(r"\[ INFO : ([\d-]+ [\d:,]+) \].*Training process", t)
    secs = []
    if st and ts:
        start = dt.datetime.strptime(st.group(1), "%Y-%m-%d %H:%M:%S,%f")
        secs = [round((ts[0] - start).total_seconds(), 2)]
        secs += [round((b - a).total_seconds(), 2) for a, b in zip(ts, ts[1:])]
    return tr, va, secs


def main():
    rows = []
    for d in sorted(V2.glob("exp/stat_*")):
        log = d / "train.log"
        if not log.exists():
            continue
        tr, va, secs = parse(log)
        if len(tr) < 3:            # 아직 안 끝난 run 은 건너뜀
            continue
        m = re.match(r"stat_g(\d)_r(\d)_(\w+)", d.name)
        row = {"tag": d.name, "gpu": m.group(1), "rep": m.group(2),
               "tracker": m.group(3), "cudnn_benchmark": True,
               "compile_model": False}
        for i in range(3):
            row[f"ep{i+1}_train"] = tr[i]
            row[f"ep{i+1}_val"] = va[i]
            row[f"sec_ep{i+1}"] = secs[i] if i < len(secs) else ""
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEFT + RIGHT)
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT} — {len(rows)} 행")


if __name__ == "__main__":
    main()
