"""#93 프로파일 결과를 표로 모아 notebooks/_runs/profile_93/ 에 CSV 로 씀.

    python notebooks/_bench/profile_93_analyze.py

읽는 것 (examples/librimix/tse/v2/exp/ 아래, gitignore):
    profile_93/       run_profile.sh 결과      — 기본 할당기 · expandable_segments
    profile_93_clip/  run_profile_clip.sh 결과 — clip_grad_mode loop · nosync · foreach
                      (run_profile_clip.sh 와 clip_grad_mode 는 브랜치 profiler/93-clip_grad_mode 에만 있음)
각 run 폴더의 memory.csv(스텝마다 할당기 통계) · memory_stats_final.json · trace.json(활성 10스텝)
과 profile_93/nvsmi.csv(100 ms 간격 GPU% · memory.used) 를 씀. 판정은 SD-FiLM 저장소
docs/issues/wesep_gpu_mem_flush_and_idle.md.
"""
import bisect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WESEP = Path(__file__).resolve().parents[2]
EXP = WESEP / "examples/librimix/tse/v2/exp"
OUT = WESEP / "notebooks/_runs/profile_93"
REGIONS = ["1_to_device", "2_forward_loss", "3_loss_item", "4_backward", "5_clip", "6_opt_step"]

PROFILE = {n: EXP / "profile_93" / n for n in ("sdfilm-base", "sdfilm-exp", "film-base", "film-exp")}
GPU_OF = {"sdfilm-base": 0, "sdfilm-exp": 1, "film-base": 2, "film-exp": 3}
CLIP = {"sdfilm-loop": EXP / "profile_93_clip/sdfilm-loop", "sdfilm-nosync": EXP / "profile_93_clip/sdfilm-nosync",
        "sdfilm-foreach": EXP / "profile_93_clip/sdfilm-foreach", "film-loop": EXP / "profile_93/film-base",
        "film-foreach": EXP / "profile_93_clip/film-foreach"}
CLIP_PAIRS = [("sdfilm-nosync", "sdfilm-loop"), ("sdfilm-foreach", "sdfilm-loop"), ("film-foreach", "film-loop")]


def memory(d):
    m = pd.read_csv(d / "memory.csv")
    m["new_tspk"] = ~m["t_spk"].duplicated()          # 이 run 에서 처음 나온 enrollment 길이
    return m


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class Trace:
    """trace.json 의 활성 스텝 · 단계 구간 · GPU 커널 구간."""

    def __init__(self, d):
        ev = json.loads((d / "trace.json").read_text())["traceEvents"]
        self.X = [e for e in ev if e.get("ph") == "X"]
        self.gmem = sorted([e for e in ev if e.get("name") == "[memory]" and e["args"].get("Device Type") == 1],
                           key=lambda e: e["ts"])                     # GPU 메모리 이벤트만 (CPU 것은 섞으면 가짜 급락)
        self.gts = [e["ts"] for e in self.gmem]
        self.gpu = merge([(e["ts"], e["ts"] + e["dur"]) for e in self.X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")])
        self.gst = np.array([s for s, _ in self.gpu])
        self.steps = sorted([e for e in self.X if str(e.get("name", "")).startswith("ProfilerStep#")], key=lambda e: e["ts"])
        self.ann = [e for e in self.X if e.get("cat") == "user_annotation" and e.get("name") in REGIONS]

    def busy(self, a, b):
        i = max(np.searchsorted(self.gst, a) - 1, 0)
        t = 0.0
        while i < len(self.gpu) and self.gpu[i][0] < b:
            t += max(0.0, min(self.gpu[i][1], b) - max(self.gpu[i][0], a))
            i += 1
        return t

    def inside(self, e, spans):
        return any(s["ts"] <= e["ts"] <= s["ts"] + s["dur"] for s in spans)


def write(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"\n## {name}\n{df.to_string(index=False)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    # 1. nvtop 두 선의 재현 — nvidia-smi 100 ms 기록
    sm = pd.read_csv(EXP / "profile_93/nvsmi.csv", names=["timestamp", "index", "util", "mem"], skipinitialspace=True)
    sm["t"] = pd.to_datetime(sm["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    sm["util"] = sm["util"].str.replace(" %", "", regex=False).astype(float)
    sm["mem"] = sm["mem"].str.replace(" MiB", "", regex=False).astype(float)
    rows = []
    for run, g in GPU_OF.items():
        d = sm[sm["index"] == g].sort_values("t").reset_index(drop=True)
        busy = d.index[d["util"] > 0]
        w = d.loc[busy[0]:busy[-1]]                       # 커널이 처음 돈 때부터 마지막까지
        rows.append({"run": run, "gpu": g, "window_s": round((w["t"].iloc[-1] - w["t"].iloc[0]).total_seconds(), 1),
                     "samples": len(w), "gpu_util_mean": round(w["util"].mean(), 1),
                     "gpu_util_lt30_pct": round((w["util"] < 30).mean() * 100, 1),
                     "gpu_util_ge90_pct": round((w["util"] >= 90).mean() * 100, 1),
                     "mem_min_mib": int(w["mem"].min()), "mem_max_mib": int(w["mem"].max()),
                     "drops_gt_10gib": int((w["mem"].diff() < -10 * 1024).sum())})
    write(pd.DataFrame(rows), "nvsmi_summary.csv")

    # 2. 할당기 누적 통계 — 캐시를 비운 횟수와 할당 실패 재시도 횟수
    rows = []
    for run, d in PROFILE.items():
        st = json.loads((d / "memory_stats_final.json").read_text())
        m = memory(d)
        rows.append({"run": run, "steps": len(m),
                     "num_alloc_retries": st["num_alloc_retries"], "num_ooms": st["num_ooms"],
                     "num_sync_all_streams": st["num_sync_all_streams"],
                     "num_device_alloc": st["num_device_alloc"], "num_device_free": st["num_device_free"],
                     "reserved_peak_gib": round(st["reserved_bytes.all.peak"] / 2**30, 2),
                     "allocated_peak_gib": round(st["allocated_bytes.all.peak"] / 2**30, 2),
                     "t_spk_distinct": m["t_spk"].nunique(), "t_spk_range": f"{m['t_spk'].min()}~{m['t_spk'].max()}"})
    write(pd.DataFrame(rows), "allocator_summary.csv")

    # 3. 스텝 시간 — 처음 보는 t_spk 와 이미 본 t_spk (스텝 100 이후) · 구간별 새 길이 비율
    rows, frac = [], []
    for run, d in PROFILE.items():
        m = memory(d)
        for flag, g in m[m["step"] >= 100].groupby("new_tspk"):
            rows.append({"run": run, "t_spk": "new" if flag else "seen", "steps": len(g),
                         "step_ms_median": round(g["wall_s"].median() * 1000, 1),
                         "step_ms_mean": round(g["wall_s"].mean() * 1000, 1)})
        for a in (0, 100, 200):
            s = m[(m["step"] >= a) & (m["step"] < a + 100)]
            frac.append({"run": run, "steps": f"{a}-{a + 99}", "new_t_spk_pct": round(s["new_tspk"].mean() * 100, 1)})
    write(pd.DataFrame(rows), "step_time_by_tspk.csv")
    write(pd.DataFrame(frac), "new_tspk_fraction.csv")

    # 4. 활성 10스텝 — 스텝마다 새 길이 여부 · reserved 범위 · GPU 유휴 · 단계별 유휴
    act, reg, drop = [], [], []
    for run in ("sdfilm-base", "film-base"):
        d, m = PROFILE[run], memory(PROFILE[run])
        tr = Trace(d)
        ops = sorted([e for e in tr.X if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
        ots = [e["ts"] for e in ops]

        def encl(t):
            i = bisect.bisect_right(ots, t) - 1
            c = [ops[j] for j in range(i, max(i - 800, -1), -1) if ops[j]["ts"] <= t <= ops[j]["ts"] + ops[j]["dur"]]
            c.sort(key=lambda e: e["dur"])
            return c[0]["name"] if c else "(none)"
        for prev, cur in zip(tr.gmem, tr.gmem[1:]):
            if cur["args"]["Total Reserved"] - prev["args"]["Total Reserved"] < -2**30:
                drop.append({"run": run, "reserved_before_gib": round(prev["args"]["Total Reserved"] / 2**30, 2),
                             "reserved_after_gib": round(cur["args"]["Total Reserved"] / 2**30, 2),
                             "innermost_op": encl(cur["ts"])})
        kinds = {}
        for st in tr.steps:
            k = int(st["name"].split("#")[1])
            a, b = st["ts"], st["ts"] + st["dur"]
            seg = tr.gmem[bisect.bisect_left(tr.gts, a):bisect.bisect_right(tr.gts, b)]
            res = [e["args"]["Total Reserved"] / 2**30 for e in seg]
            new = bool(m.loc[m["step"] == k, "new_tspk"].iloc[0])
            kinds.setdefault(new, []).append(st)
            act.append({"run": run, "profiler_step": k, "t_spk": int(m.loc[m["step"] == k, "t_spk"].iloc[0]),
                        "new_t_spk": new, "step_ms": round(st["dur"] / 1000, 1),
                        "gpu_idle_pct": round((1 - tr.busy(a, b) / st["dur"]) * 100, 1),
                        "reserved_min_gib": round(min(res), 2), "reserved_max_gib": round(max(res), 2)})
        for new, ss in kinds.items():
            row = {"run": run, "t_spk": "new" if new else "seen", "steps": len(ss),
                   "step_ms": round(sum(s["dur"] for s in ss) / len(ss) / 1000, 1)}
            for r in REGIONS:
                es = [e for e in tr.ann if e["name"] == r and tr.inside(e, ss)]
                cpu = sum(e["dur"] for e in es)
                row[f"{r}_idle_ms"] = round((cpu - sum(tr.busy(e["ts"], e["ts"] + e["dur"]) for e in es)) / len(ss) / 1000, 1)
            reg.append(row)
    write(pd.DataFrame(act), "active_steps.csv")
    write(pd.DataFrame(reg), "idle_by_region.csv")
    write(pd.DataFrame(drop), "reserved_drops.csv")

    # 5. clip_grad_mode 학습 안 A/B — 같은 스텝끼리 짝지은 스텝 시간 차이 (스텝 120 이후)
    if all((d / "memory.csv").exists() for d in CLIP.values()):
        rows = []
        for a, b in CLIP_PAIRS:
            ma, mb = memory(CLIP[a]), memory(CLIP[b])
            same = len(ma) == len(mb) and bool((ma["t_spk"].values == mb["t_spk"].values).all())
            j = ma.merge(mb[["step", "wall_s"]], on="step", suffixes=("", "_base"))
            for flag, g in j[j["step"] >= 120].groupby("new_tspk"):
                dd = (g["wall_s"] - g["wall_s_base"]) * 1000
                rows.append({"target": a, "baseline": b, "same_t_spk_order": same, "t_spk": "new" if flag else "seen",
                             "steps": len(g), "baseline_ms_median": round(g["wall_s_base"].median() * 1000, 1),
                             "target_ms_median": round(g["wall_s"].median() * 1000, 1),
                             "paired_diff_ms_median": round(dd.median(), 1), "paired_diff_ms_mean": round(dd.mean(), 1),
                             "target_faster_pct": round((dd < 0).mean() * 100, 1)})
        write(pd.DataFrame(rows), "clip_ab_step_time.csv")
        rows = []
        for run, d in CLIP.items():
            tr = Trace(d)
            clip = [e for e in tr.ann if e["name"] == "5_clip"]
            n = len(clip)
            cpu = sum(e["dur"] for e in clip)
            rows.append({"run": run, "clip_cpu_ms": round(cpu / n / 1000, 1),
                         "clip_gpu_idle_ms": round((cpu - sum(tr.busy(e["ts"], e["ts"] + e["dur"]) for e in clip)) / n / 1000, 1),
                         "syncs_per_step": round(sum(1 for e in tr.X if e.get("name") == "cudaStreamSynchronize" and tr.inside(e, clip)) / n, 1),
                         "launches_per_step": round(sum(1 for e in tr.X if e.get("name") in ("cudaLaunchKernel", "cudaLaunchKernelExC")
                                                        and tr.inside(e, clip)) / n, 1)})
        write(pd.DataFrame(rows), "clip_ab_trace.csv")

    # 원자료 — 스텝별 할당기 통계와 nvidia-smi 기록 (trace.json 은 run 당 약 590 MB 라 옮기지 않음)
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    shutil.copy(EXP / "profile_93/nvsmi.csv", raw / "nvsmi.csv")
    for run, d in {**PROFILE, **{k: v for k, v in CLIP.items() if k != "film-loop"}}.items():
        if (d / "memory.csv").exists():
            shutil.copy(d / "memory.csv", raw / f"memory_{run}.csv")
    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
