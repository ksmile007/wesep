"""조건 하나만 재는 스크립트 — GPU 한 장에 조건 하나씩 붙일 때 씀.

    CUDA_VISIBLE_DEVICES=0 python run_one.py dynamic-none 0
    CUDA_VISIBLE_DEVICES=1 python run_one.py dynamic-none 1
    ...

worker.py 는 scope·dynamic 만 다른 조건을 **한 묶음으로 같은 GPU** 에 보냄(speedup 기준선을
떼어놓지 않으려는 것임). 그래서 조건이 몇 개뿐일 때는 GPU 여러 장에 못 퍼짐 — 이 스크립트가 그 자리를 메움.

**주의** — 조건마다 GPU 가 다르면 절대 ms 를 서로 비교할 수 없음. 급할 때만 쓸 것.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"사용법: {Path(__file__).name} <tag> <조건 인덱스>")
    tag, idx = sys.argv[1], int(sys.argv[2])

    conds = bc.conditions(tag)
    if not 0 <= idx < len(conds):
        raise SystemExit(f"{tag} 의 조건은 0~{len(conds) - 1} 뿐임 (받은 값 {idx})")

    cond = conds[idx]
    print(f"[{tag}#{idx}] {cond} | CUDA_VISIBLE_DEVICES="
          f"{bc.os.environ.get('CUDA_VISIBLE_DEVICES', '(전부)')}", flush=True)

    if bc.load(tag, cond) is not None:
        print("이미 측정됨 — 건너뜀", flush=True)
        return

    row = bc.bench(**cond)
    bc.save(tag, cond, row)
    print(" | ".join(f"{k}={row[k]}" for k in
                     ("scope", "dynamic", "b", "prec", "compile_s",
                      "ms_step_median", "ms_step_mean", "status")), flush=True)


if __name__ == "__main__":
    main()
