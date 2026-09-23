#!/usr/bin/env bash
# #93 개선안 ② 속도 A/B — clip_grad_mode 만 바꿔 학습 4개를 동시에 300 스텝씩 돌림 (약 10분)
#   loop    : 원본 clip_gradients — 텐서마다 GPU 를 기다림 (기준)
#   nosync  : 같은 연산을 기다리지 않고 — 결과 비트 동일 (실측)
#   foreach : 노름을 한 번에 — 결과가 미세하게 다름 (실측)
#   실행: bash run_profile_clip.sh
#   결과: exp/profile_93_clip/<이름>/memory.csv (스텝마다 wall_s · t_spk) · trace.json · key_averages.txt
#   비교 기준 run 은 run_profile.sh 와 같은 조건 (GPU 4장 동시 · 16-mixed · 기본 할당기)

set -u                  # 정의 안 된 변수를 쓰면 빈 값으로 넘어가지 않고 바로 멈춤

# 이 스크립트가 있는 폴더(v2)로 이동 — 어디서 실행해도 아래 상대경로(run.sh · confs/ · exp/)가 맞게 됨
#   $0            : 실행한 스크립트 경로 (예: examples/librimix/tse/v2/run_profile_clip.sh)
#   dirname "$0"  : 그 경로의 폴더 부분 (예: examples/librimix/tse/v2)
#   "$( ... )"    : 괄호 안 명령의 출력을 그 자리에 넣음. 큰따옴표는 경로에 공백이 있어도 한 덩어리로 넘기려고
cd "$(dirname "$0")"
export PYTHONPATH=${PYTHONPATH:-}   # path.sh 가 $PYTHONPATH 를 읽는데, 비어 있으면 set -u 에 걸려 멈춤
. ./path.sh             # PYTHONPATH 에 wesep 루트를 넣음 — 아래 make_debug_config.py 가 wesep 을 import 함

OUT=exp/profile_93_clip
mkdir -p $OUT/confs     # -p = 중간 폴더까지 만들고, 이미 있어도 오류 없음

export PATH=/root/miniconda3/envs/wesep2/bin:$PATH   # run.sh 의 python 을 wesep2 로
export WESEP_PROFILE=1                               # executor.py 의 StepProfiler 를 켬
export WESEP_PROFILE_STEPS=300                       # 300 스텝에서 결과를 쓰고 종료

# config 만들기 — 본 config 에 clip_grad_mode 한 줄만 덮어쓴 자립형 yaml 을 $OUT/confs/ 에 씀
#   confs/ 에 파일을 늘리지 않으려고 run.sh --debug 가 쓰는 local/make_debug_config.py 를 그대로 씀
#   mk <본 config 이름> <clip_grad_mode> → 만든 yaml 의 경로를 출력
mk() {
  echo "clip_grad_mode: $2" > $OUT/confs/overlay_$2.yaml
  python local/make_debug_config.py confs/$1.yaml $OUT/confs/overlay_$2.yaml $OUT/confs/$1_$2.yaml
}
SDFILM_NOSYNC=$(mk bsrnn_ecapa_sdfilm_spkeval nosync)
SDFILM_FOREACH=$(mk bsrnn_ecapa_sdfilm_spkeval foreach)
FILM_FOREACH=$(mk bsrnn_ecapa_FiLM_spkeval foreach)

# 학습 하나를 끝까지 돌림 — run <GPU> <이름> <config>
#   $1 $2 $3 = 호출할 때 준 1~3번째 인자
#   env 변수=값 ... 명령 = 그 변수들을 이 명령에만 설정하고 실행 (다른 run 에는 새지 않음)
#   \ = 다음 줄로 이어짐 · > 파일 2>&1 = 화면 출력(1)과 오류 출력(2)을 모두 그 파일로
run() {
  env CUDA_VISIBLE_DEVICES=$1 WESEP_PROFILE_DIR=$OUT/$2 \
    bash run.sh --stage 3 --stop_stage 3 --config $3 --exp_dir $OUT/$2 \
                --precision 16-mixed --tracker tensorboard \
    > $OUT/$2.log 2>&1
}

# ( ) = 서브셸 — 괄호 안 명령을 묶어 따로 돎
#   & 로 4개를 동시에 띄우고, 괄호 안의 wait 는 괄호 안에서 띄운 이 4개만 기다림
#   film-loop 은 따로 돌리지 않음 — run_profile.sh 의 film-base(같은 조건)를 기준으로 씀
(
  run 0 sdfilm-loop    confs/bsrnn_ecapa_sdfilm_spkeval.yaml &   # 기준 (clip_grad_mode 키 없음 = loop)
  run 1 sdfilm-nosync  $SDFILM_NOSYNC                        &
  run 2 sdfilm-foreach $SDFILM_FOREACH                       &
  run 3 film-foreach   $FILM_FOREACH                         &
  wait
)

ls -l $OUT/*/memory.csv    # 결과가 생긴 run 만 보임 — 빠진 이름이 있으면 그 .log 를 볼 것
echo "done: $OUT"
