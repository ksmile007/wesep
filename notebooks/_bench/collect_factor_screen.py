"""factor_screen.csv 를 읽어 "어느 요인을 바꾸면 무엇이 달라지는가" 를 표로 만듦.

    python notebooks/_bench/collect_factor_screen.py

두 가지를 따로 봄 — 섞으면 안 됨.

  흔들림(spread) : 같은 arm 의 8 run 안에서 값이 몇 갈래로 갈리는가.
                   1 이면 그 요인이 흔들림을 **잡은** 것임
  이동(shift)    : arm 의 중앙값이 base 와 다른가.
                   달라도 흔들림은 남을 수 있고, 그 반대도 있음
"""
import csv
from collections import defaultdict
from pathlib import Path
from math import isfinite
from statistics import median

CSV = Path(__file__).resolve().parents[1] / "_runs/factor_screen.csv"
MEASURES = ["loss_step1", "grad_sum_step1", "loss_step2"]


def load():
    with CSV.open() as f:
        return [r for r in csv.DictReader(f)
                if not r["arm"].startswith("smoke")]


def main():
    rows = load()
    if not rows:
        print("행 없음")
        return

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    base = by_arm.get("base", [])
    base_med = {m: median(float(r[m]) for r in base) for m in MEASURES} \
        if base else {}

    hdr = ["arm", "n", "유효"] + \
          [f"{m}_갈래" for m in MEASURES] + \
          [f"{m}_base대비" for m in MEASURES] + ["sec"]
    tbl = []
    for arm, rs in by_arm.items():
        row = {"arm": arm, "n": len(rs)}
        # 갈래=1 을 그대로 믿으면 안 됨 — 값이 nan/inf 여도 전부 같아져 1 이 됨.
        # amp_on 이 실제로 그랬음(GradScaler 가 inf 를 보고 스텝을 건너뜀)
        bad = any(not isfinite(float(r[m])) for r in rs for m in MEASURES)
        row["유효"] = "X(nan/inf)" if bad else "O"
        for m in MEASURES:
            row[f"{m}_갈래"] = len({r[m] for r in rs})
            if base_med:
                d = median(float(r[m]) for r in rs) - base_med[m]
                row[f"{m}_base대비"] = f"{d:+.6g}"
            else:
                row[f"{m}_base대비"] = ""
        row["sec"] = round(sum(float(r["sec"]) for r in rs) / len(rs), 1)
        tbl.append(row)

    order = ["base", "bm_off", "det_on", "cudnn_tf32_off", "matmul_tf32_on",
             "det_algos_on", "cudnn_off", "amp_on", "tracker_tb", "enr_vary",
             "bm_off__det_on", "bm_off__det_algos",
             "bm_off__det_on__det_algos", "cudnn_off__det_algos"]
    tbl.sort(key=lambda r: order.index(r["arm"])
             if r["arm"] in order else 99)

    w = {h: max(len(h), *(len(str(r[h])) for r in tbl)) for h in hdr}
    print(f"총 {len(rows)} run\n")
    print("  " + "  ".join(h.ljust(w[h]) for h in hdr))
    print("  " + "  ".join("-" * w[h] for h in hdr))
    for r in tbl:
        print("  " + "  ".join(str(r[h]).ljust(w[h]) for h in hdr))

    print("\n갈래=1 인 칸이 '그 요인이 흔들림을 잡았다' 는 뜻임 — 단 유효=O 일 때만.")
    print("유효=X 는 nan/inf 가 섞여 값이 전부 같아진 것이라 원인을 잡은 것이 아님.")
    print("base대비 는 중앙값의 차이 — 값이 이동했는지만 보는 것이고 흔들림과는 별개임.")


if __name__ == "__main__":
    main()
