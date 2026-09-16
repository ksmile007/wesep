"""조건을 나눠 재는 워커. GPU 한 장당 하나씩 띄움.

    CUDA_VISIBLE_DEVICES=0 python worker.py --worker 0 --n-workers 3
    CUDA_VISIBLE_DEVICES=1 python worker.py --worker 1 --n-workers 3
    CUDA_VISIBLE_DEVICES=3 python worker.py --worker 2 --n-workers 3

같은 GPU 에서 재야 비교가 되는 조건(scope·dynamic 만 다른 것들)은 **한 묶음**으로 묶여
같은 워커에게 감 — bench_common._group_key 참조.

이미 측정된 조건은 건너뜀. 중간에 죽어도 다시 띄우면 남은 것부터 이어서 감.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True, help="0-based 워커 번호")
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--tags", nargs="*", default=list(bc.ALL_TAGS),
                    help=f"기본값: {' '.join(bc.ALL_TAGS)}")
    args = ap.parse_args()

    print(f"worker {args.worker}/{args.n_workers} | device {bc.DEVICE} | "
          f"CUDA_VISIBLE_DEVICES={bc.os.environ.get('CUDA_VISIBLE_DEVICES', '(전부)')}", flush=True)
    print(f"결과 폴더: {bc.RUNS}", flush=True)

    t_start = time.perf_counter()
    for tag in args.tags:
        n_before = sum(bc.load(tag, c) is not None for c in bc.conditions(tag))
        print(f"\n===== {tag} | 전체 {len(bc.conditions(tag))} 조건 "
              f"| 이미 측정됨 {n_before} =====", flush=True)
        bc.run_or_load(tag, measure_missing=True,
                       worker=args.worker, n_workers=args.n_workers, verbose=True)

    print(f"\n워커 {args.worker} 완료 | {(time.perf_counter() - t_start) / 60:.1f} 분", flush=True)


if __name__ == "__main__":
    main()
