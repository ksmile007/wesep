#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)

. ./path.sh || exit 1

# General configuration
stage=-1
stop_stage=-1

# Data preparation related
data=data
fs=16k
min_max=min
noise_type="clean"
data_type="shard" # shard/raw
Libri2Mix_dir=/workspace/DB/Libri2Mix
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Training related
gpus="[0]"
use_gan_loss=false
config=confs/bsrnn.yaml
exp_dir=exp/BSRNN/no_spk_transform-multiply_fuse
if [ -z "${config}" ] && [ -f "${exp_dir}/config.yaml" ]; then
  config="${exp_dir}/config.yaml"
fi

# TSE model initialization related
checkpoint=

# Inferencing and scoring related
save_results=true
use_pesq=true
use_dnsmos=true
dnsmos_use_gpu=true

# Model average related
# 두 모드는 서로 다른 인자 하나씩만 씀 — 같이 적어도 겹치지 않음.
#   best  : avg_epochs 를 씀.  num_avg 는 무시 (아래 stage 4 가 개수를 세어 덮어씀)
#   final : num_avg 를 씀.     avg_epochs 는 무시
avg_mode=best
avg_epochs="138,141"       # best 일 때만. 여기 적은 epoch 만 평균함
num_avg=10                 # final 일 때만. 마지막 몇 개를 평균할지

# 학습 곡선 기록 — exp_dir/tb 에 tfevents 를 씀
#   none        : 안 함 (원본 동작)
#   tensorboard : 로컬 파일만
#   both        : 로컬 파일 + wandb 실시간 업로드
# both 는 wandb 가 SummaryWriter 를 가로채는 방식이라 tfevents 도 그대로 남음.
# wandb 로그인은 ~/.netrc 에 저장되므로 conda 환경과 무관함
tracker=both

# 지표 CSV — exp_dir 에 metrics_step.csv · metrics_epoch.csv 를 씀.
# tracker 설정과 **무관하게 항상** 기록함 (tracker=none 인 run 도 비교 대상이므로).
# 기록할 때마다 save() 를 부르므로 도는 중에 tail -F 로 볼 수 있음.
# 소문자 -f 가 아니라 **대문자 -F** 임 - metrics.csv 는 첫 기록 때 만들어지므로
# 그 전에는 파일이 없어 -f 가 바로 죽음(Lightning CSVLogger 가 헤더를 정하려면 지표가 필요함)
#   50 : 50 스텝마다 스텝 손실을 남김. 단 global_step 0 은 건너뜀 —
#        학습 전 손실이라 홀로 크게 튀어 그래프 y축을 다 잡아먹었음
#    0 : 스텝 기록을 끄고 에포크만 남김
# 아래 log_batch_interval 과는 별개임 — 그것은 train.log 에 표를 찍는 주기임
tracker_step_interval=50

# 학습 정밀도 — Lightning 과 같은 이름. 이 하나가 autocast 와 GradScaler 를 다 정함.
#   32-true     : fp32.      autocast 꺼짐, GradScaler 꺼짐
#   16-mixed    : fp16 혼합.  autocast 켜짐, GradScaler **켜짐**
#   bf16-mixed  : bf16 혼합.  autocast 켜짐, GradScaler 꺼짐 (지수부가 fp32 와 같아 불필요)
# 이 값이 본 config 의 옛 키 enable_amp 을 **항상 덮어씀.**
# 그래서 16-mixed 로 둠 - Table 2 의 bsrnn_ecapa_*.yaml 4개가 enable_amp: true 라
# 기존 run 과 같은 fp16 으로 유지됨. enable_amp: false 인 config 를 그 뜻대로 돌리려면
# 여기를 32-true 로 바꾸거나 --precision 32-true 를 줄 것
precision=16-mixed

# Debug 관련 — 짧게 돌려 "도는가 · GPU 메모리가 되는가" 만 볼 때.
# 아래 dev/ 경로들은 Libri2Mix 의 검증 분할이라 뜻이 다름. 헷갈리지 말 것
debug=false                    # true 면 아래 debug_config 를 본 config 위에 덮어씀
debug_config=confs/debug.yaml  # 덮어쓸 키만 담긴 파일
debug_test_shards=1            # debug 일 때 stage 5 평가에 쓸 shard tar 개수 (전체는 3개)

. tools/parse_options.sh || exit 1

# --debug true 면 본 config 에 debug 덮어쓰기를 얹은 임시 config 로 갈아타고, 실험 폴더도 분리함.
# 폴더를 나누는 이유는 아래 stage 3 이 exp_dir 의 latest_checkpoint.pt 를 자동으로 이어받기 때문임 —
# 같은 폴더를 쓰면 본 학습이 3 epoch 짜리 debug 가중치에서 시작해 버림.
if ${debug}; then
  exp_dir="${exp_dir}_debug"
  mkdir -p "${exp_dir}"
  config=$(python local/make_debug_config.py "${config}" "${debug_config}" "${exp_dir}/config_debug.yaml")
  avg_mode=final    # 3 epoch 만 돌아 checkpoint_138·141 이 없으므로 마지막 num_avg 개를 평균
  echo "Debug mode: config=${config}  exp_dir=${exp_dir}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 3
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in train-100 dev test; do
    #  for dset in train-360; do
    python tools/make_shard_list_premix.py --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
      ${data}/$dset/shards ${data}/$dset/shard.list
  done
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  if ${use_gan_loss}; then
    train_script=wesep/bin/train_gan.py
  else
    train_script=wesep/bin/train.py
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --tracker ${tracker} \
    --tracker_step_interval ${tracker_step_interval} \
    --precision ${precision} \
    --data_type "${data_type}" \
    --train_data ${data}/train-100/${data_type}.list \
    --train_utt2spk ${data}/train-100/single.utt2spk \
    --train_spk2utt ${data}/train-100/spk2enroll.json \
    --val_data ${data}/dev/${data_type}.list \
    --val_spk1_enroll ${data}/dev/spk1.enroll \
    --val_spk2_enroll ${data}/dev/spk2.enroll \
    --val_spk2utt ${data}/dev/single.wav.scp \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_best_model.pt
  avg_opts=(--mode "${avg_mode}")
  # <<<<< 고친 것 - best 모드면 평균할 개수를 avg_epochs 에서 세어 num_avg 를 덮어씀.
  #       average_model.py 는 best 일 때 avg_epochs 로 목록을 정하고 num 은 개수 검사에만 씀 —
  #       두 값이 어긋나면 assert 에서 죽으므로, 같은 정보를 두 번 적지 않게 함.
  if [ "${avg_mode}" = best ]; then
    avg_opts+=(--epochs "${avg_epochs}")
    num_avg=$(awk -F',' '{print NF}' <<<"${avg_epochs}")
  fi
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    "${avg_opts[@]}"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi


# shellcheck disable=SC2215
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start inferencing ..."
  test_data=${data}/test/${data_type}.list
  if ${debug}; then    # 평가도 줄임 — shard tar 앞 몇 개만
    head -n ${debug_test_shards} ${test_data} >${exp_dir}/test_debug.list
    test_data=${exp_dir}/test_debug.list
  fi
  python wesep/bin/infer.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${test_data} \
    --test_spk1_enroll ${data}/test/spk1.enroll \
    --test_spk2_enroll ${data}/test/spk2.enroll \
    --test_spk2utt ${data}/test/single.wav.scp \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  score_dset=${data}/test
  if ${debug}; then
    # 추론을 shard 일부만 돌렸으므로 정답 목록도 같은 키만 남김.
    # 안 그러면 score.py:120 의 assert inf_reader.keys() == ref_reader.keys() 에서 죽음
    score_dset=${exp_dir}/test_debug_dset
    mkdir -p ${score_dset}
    awk 'NR==FNR{k[$1];next} ($1 in k)' ${exp_dir}/audio/spk1.scp \
        ${data}/test/single.wav.scp >${score_dset}/single.wav.scp
    echo "Debug mode: score_dset=${score_dset} ($(wc -l <${score_dset}/single.wav.scp) 발화)"
  fi
  ./tools/score.sh --dset "${score_dset}" \
    --exp_dir "${exp_dir}" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
