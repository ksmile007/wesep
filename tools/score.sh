#!/bin/bash

min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

# Set default values
dset=
exp_dir=
scoring_opts=

n_gpu=1
score_nj=16
ref_channel=0
use_pesq=false
use_dnsmos=false
dnsmos_use_gpu=true
fs=16k
scoring_protocol="STOI SDR SAR SIR SI_SNR"

# Parse command line options
. tools/parse_options.sh || exit 1

if [ ! ${fs} = 16k ] && ${use_dnsmos}; then
    echo "Warning: DNSMOS only supports 16k sampling rate."
    echo "--use_dnsmos will be set to false automatically."
    use_dnsmos=false
fi

# Set scoring options
scoring_opts=""
if ${use_dnsmos}; then
    # Set model path
    primary_model_path=DNSMOS/sig_bak_ovr.onnx
    p808_model_path=DNSMOS/model_v8.onnx

    if [ ! -f ${primary_model_path} ] || [ ! -f ${p808_model_path} ]; then
        echo "=========================================="
        echo "Warning: DNSMOS model files are not found."
        echo "Trying to download them from the official repository."
        echo "If this takes too long,"
        echo "please manually download the model files"
        echo "and put them in the DNSMOS directory."
        echo "=========================================="

        # creat directory for DNSMOS model files
        mkdir -p DNSMOS
        # download DNSMOS model files and save them to the directory
        wget -P DNSMOS https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
        wget -P DNSMOS https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx
        # check if the model files are downloaded successfully
        if [ ! -f ${primary_model_path} ] || [ ! -f ${p808_model_path} ]; then
            echo "Error: DNSMOS model files are not downloaded successfully."
            exit 1
        fi
    fi
    scoring_opts+="--dnsmos_mode local "
    scoring_opts+="--dnsmos_primary_model ${primary_model_path} "
    scoring_opts+="--dnsmos_p808_model ${p808_model_path} "
    if ${dnsmos_use_gpu}; then
        score_nj=$(min "${score_nj}" "${n_gpu}")
        scoring_opts+="--dnsmos_use_gpu ${dnsmos_use_gpu} "
    fi
fi

# Set directories and log directory
_dir="${exp_dir}/scoring"
_logdir="${_dir}/logdir"
mkdir -p "${_logdir}"

# 0. Check the inference file
inf_scp=${exp_dir}/audio/spk1.scp
if [ ! -s "${inf_scp}" ] || [ -z "$(cat "${inf_scp}")" ]; then
    echo "Error: ${inf_scp} does not exist or is empty!"
    exit 1
fi
# 1. Split the key file
key_file=${dset}/single.wav.scp
split_scps=""
_nj=$(min "${score_nj}" "$(wc <${key_file} -l)")
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
./tools/split_scp.pl "${key_file}" ${split_scps}

_ref_scp="--ref_scp ${dset}/single.wav.scp "
_inf_scp="--inf_scp ${exp_dir}/audio/spk1.scp "

# 2. Submit scoring jobs
echo "log: '${_logdir}/tse_scoring.*.log'"
if ${use_dnsmos} && ${dnsmos_use_gpu}; then
    cmd="./tools/run.pl --gpu ${n_gpu}"
else
    cmd="./tools/run.pl"
fi
# <<<<< 더한 것 - run.pl 이 각 작업의 출력을 로그 파일로 돌려서 화면에 진행이 안 보임.
#       로그에 쌓이는 "Scoring" 줄을 세어 현재/전체를 보여 줌 (score.py:124 가 발화마다 한 줄 남김)
_total=$(wc -l <"${key_file}")
(
    while true; do
        _done=$(cat "${_logdir}"/tse_scoring.*.log 2>/dev/null | grep -c Scoring)
        printf "\r  Scoring  %s / %s" "${_done}" "${_total}"
        sleep 1
    done
) &
_progress_pid=$!
trap 'kill ${_progress_pid} 2>/dev/null' EXIT

# shellcheck disable=SC2086
${cmd} JOB=1:"${_nj}" "${_logdir}"/tse_scoring.JOB.log \
    python -m wesep.bin.score \
    --key_file "${_logdir}"/keys.JOB.scp \
    --output_dir "${_logdir}"/output.JOB \
    ${_ref_scp} \
    ${_inf_scp} \
    --ref_channel ${ref_channel} \
    --use_pesq ${use_pesq} \
    --use_dnsmos ${use_dnsmos} \
    --dnsmos_gpu_device JOB \
    ${scoring_opts}

# Check if PESQ is used
if "${use_pesq}"; then
    if [ ${fs} = 16k ]; then
        scoring_protocol+=" PESQ_WB"
    else
        scoring_protocol+=" PESQ_NB"
    fi
fi

# Check if dnsmos is used
if "${use_dnsmos}"; then
    scoring_protocol+=" BAK SIG OVRL P808_MOS"
fi

kill ${_progress_pid} 2>/dev/null
trap - EXIT
printf "\r  Scoring  %s / %s\n" "${_total}" "${_total}"

# Merge and sort result files
for protocol in ${scoring_protocol} wav; do
    for i in $(seq "${_nj}"); do
        cat "${_logdir}/output.${i}/${protocol}_spk1"
    done | LC_ALL=C sort -k1 >"${_dir}/${protocol}_spk1"
done

# Calculate and save results
for protocol in ${scoring_protocol}; do
    # shellcheck disable=SC2046
    paste $(printf "%s/%s_spk1 " "${_dir}" "${protocol}") |
        awk 'BEGIN{sum=0}
            {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
            END{printf ("%.2f\n",sum/NR)}' >"${_dir}/result_${protocol,,}.txt"
done

# show the result
./tools/show_enh_score.sh "${_dir}/../.." > \
    "${_dir}/../../RESULTS.md"
