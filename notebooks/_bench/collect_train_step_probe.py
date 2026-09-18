"""train_step_probe.csv 를 읽어 "어느 조건에서 값이 몇 갈래로 갈리는지" 를 표로 만듦.

    python notebooks/_bench/collect_train_step_probe.py

흔들림의 정의 — 같은 조건(tag · tracker · cudnn_benchmark) 안에서
ep1_train 문자열이 서로 다른 값의 **개수**. 1 이면 모든 run 이 일치한 것임.
부동소수점 비교라 repr 문자열을 그대로 비교함 (반올림으로 뭉개지 않게).
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "_runs/train_step_probe.csv"


def load():
    with CSV.open() as f:
        return [r for r in csv.DictReader(f)]


def group_table(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in sorted(g.items()):
        ep1 = sorted({r["ep1_train"] for r in rs})
        ep3 = sorted({r["ep3_train"] for r in rs})
        st1 = sorted({r["step1_loss"] for r in rs})
        wsum = sorted({r["weight_sum_after"] for r in rs})
        row = dict(zip(keys, k))
        row.update({
            "n_run": len(rs),
            "n_distinct_step1": len(st1),
            "n_distinct_ep1": len(ep1),
            "n_distinct_ep3": len(ep3),
            "n_distinct_weight_sum": len(wsum),
            "ep1_values": " | ".join(ep1),
        })
        out.append(row)
    return out


def main():
    rows = [r for r in load() if r["tag"] != "smoke"]
    if not rows:
        print("행 없음")
        return

    print(f"총 {len(rows)} run\n")

    for keys in (["tag", "cudnn_benchmark"],
                 ["tag", "cudnn_benchmark", "tracker"],
                 ["tag", "cudnn_benchmark", "gpu"]):
        print("=" * 100)
        print("묶음 기준:", " · ".join(keys))
        tbl = group_table(rows, keys)
        hdr = list(tbl[0].keys())
        widths = {h: max(len(h), *(len(str(r[h])) for r in tbl)) for h in hdr}
        widths["ep1_values"] = min(widths["ep1_values"], 74)
        print("  " + "  ".join(h.ljust(widths[h]) for h in hdr))
        for r in tbl:
            print("  " + "  ".join(
                str(r[h])[:widths[h]].ljust(widths[h]) for h in hdr))
        print()


if __name__ == "__main__":
    main()
