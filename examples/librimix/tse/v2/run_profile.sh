#!/usr/bin/env bash
# #93 프로파일 — GPU 4장에서 학습 4개를 동시에 300 스텝만 돌려 GPU 유휴 · 메모리 출렁임을 기록함 (약 7분)
#   실행: bash run_profile.sh
#   결과: exp/profile_93/<이름>/ — trace.json · key_averages.txt · memory.csv · memory_stats_final.json
#         exp/profile_93/<이름>.log · exp/profile_93/nvsmi.csv
#   판정 기준: SD-FiLM 저장소 docs/issues/wesep_gpu_mem_flush_and_idle.md

set -u                  # 정의 안 된 변수를 쓰면 빈 값으로 넘어가지 않고 바로 멈춤

# 이 스크립트가 있는 폴더(v2)로 이동 — 어디서 실행해도 아래 상대경로(run.sh · confs/ · exp/)가 맞게 됨
#   $0            : 실행한 스크립트 경로 (예: examples/librimix/tse/v2/run_profile.sh)
#   dirname "$0"  : 그 경로의 폴더 부분 (예: examples/librimix/tse/v2)
#   "$( ... )"    : 괄호 안 명령의 출력을 그 자리에 넣음. 큰따옴표는 경로에 공백이 있어도 한 덩어리로 넘기려고
cd "$(dirname "$0")"

OUT=exp/profile_93
mkdir -p $OUT           # -p = 중간 폴더까지 만들고, 이미 있어도 오류 없음

# export 한 변수는 이 스크립트가 띄우는 프로세스(run.sh → torchrun → train.py)까지 전달됨
export PATH=/root/miniconda3/envs/wesep2/bin:$PATH   # run.sh 의 python 을 wesep2 로
export WESEP_PROFILE=1                               # executor.py 의 StepProfiler 를 켬
export WESEP_PROFILE_STEPS=300                       # 300 스텝에서 결과를 쓰고 종료

# 학습 하나를 끝까지 돌림 — run <GPU> <이름> <config> [할당기 설정]
#   $1 $2 $3 $4 = 호출할 때 준 1~4번째 인자
#   ${4:+글자} = "4번째 인자가 있으면 '글자'로 바꾸고, 없거나 비었으면 아무것도 안 넣음"
#       run 0 a c.yaml                          → env CUDA_VISIBLE_DEVICES=0 ...          (기본 할당기)
#       run 1 a c.yaml expandable_segments:True → env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 ...
#   env 변수=값 ... 명령 = 그 변수들을 이 명령에만 설정하고 실행 (다른 run 에는 새지 않음)
#   \ = 다음 줄로 이어짐 · > 파일 2>&1 = 화면 출력(1)과 오류 출력(2)을 모두 그 파일로
run() {
  env ${4:+PYTORCH_CUDA_ALLOC_CONF=$4} CUDA_VISIBLE_DEVICES=$1 WESEP_PROFILE_DIR=$OUT/$2 \
    bash run.sh --stage 3 --stop_stage 3 --config $3 --exp_dir $OUT/$2 \
                --precision 16-mixed --tracker tensorboard \
    > $OUT/$2.log 2>&1
}

# GPU% · 메모리 사용량을 100 ms 마다 기록 (nvtop 의 파란 선 · 노란 선과 같은 값)
#   끝의 & = 백그라운드로 띄우고 기다리지 않고 다음 줄로
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
           --format=csv,noheader -lms 100 > $OUT/nvsmi.csv &
# $! = 바로 앞에서 & 로 띄운 프로세스(= 위 nvidia-smi)의 PID. 아래 run 들이 $! 를 덮어쓰기 전에 SMI 에 저장해 둠
SMI=$!

# ( ) = 서브셸 — 괄호 안 명령을 묶어 따로 돎
#   & 로 4개를 동시에 띄우고, 괄호 안의 wait 는 괄호 안에서 띄운 이 4개만 기다림
#   (바깥의 nvidia-smi 는 스스로 안 끝나므로, 그냥 wait 하면 영원히 기다리게 됨)
(
  run 0 sdfilm-base confs/bsrnn_ecapa_sdfilm_spkeval.yaml                          &
  run 1 sdfilm-exp  confs/bsrnn_ecapa_sdfilm_spkeval.yaml expandable_segments:True &
  run 2 film-base   confs/bsrnn_ecapa_FiLM_spkeval.yaml                            &
  run 3 film-exp    confs/bsrnn_ecapa_FiLM_spkeval.yaml   expandable_segments:True &
  wait
)

kill $SMI                  # 저장해 둔 PID 로 nvidia-smi 기록기만 끔 (프로세스 종료 — nvsmi.csv 는 남음)
ls -l $OUT/*/memory.csv    # 결과가 생긴 run 만 보임 — 빠진 이름이 있으면 그 .log 를 볼 것
echo "done: $OUT"


# bash run_profile.sh 로 실행하면
#   exp/profile_93/nvsmi.csv         : GPU% · 메모리 사용량(nvtop 의 두 선)이 100 ms 마다 기록됨 — GPU 4장 모두
#   exp/profile_93/<이름>/memory.csv : 학습 스텝마다 PyTorch 할당기 통계(allocated · reserved · num_alloc_retries · t_spk 등)가 기록됨