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
num_avg=10
avg_mode=best              # best: 아래 avg_epochs 를 씀 / final: 마지막 num_avg 개를 씀
avg_epochs="138,141"       # avg_mode=best 일 때만 쓰임

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
  [ "${avg_mode}" = best ] && avg_opts+=(--epochs "${avg_epochs}")
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
