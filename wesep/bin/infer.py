from __future__ import print_function

import os
import time

import fire
import pandas as pd    # <<<<< 더한 것 - 발화별 결과를 csv 로 남기려고
import soundfile
import torch
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import read_label_file, read_vec_scp_file
from wesep.utils.param_summary import count_params   # <<<<< 더한 것 (#95)
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# <<<<< 더한 것 - logger.info 가 tqdm 막대와 같은 줄에 겹쳐 찍히는 것을 막음.
#       괄호가 있어야 함 — @contextmanager 객체를 만들어 데코레이터로 쓰는 것임
@logging_redirect_tqdm()
def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0
    # <<<<< 더한 것 - infer.log 의 Num= 줄은 {:.2f} 로 잘려 원값이 안 남으므로
    #       같은 값을 원 정밀도로 모아 csv 로 저장함
    utt_rows = []

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get(
        "save_wav", True)  # Control if save the extracted speech as .wav

    rank = 0
    set_seed(configs["seed"] + rank)
    # <<<<< 더한 것 - 평가만 cudnn.benchmark 를 끔 (#94). 발화마다 길이가 달라 커널을 매번 다시 골라
    #       재실행마다 결과가 흔들렸음. 학습은 set_seed() 의 True 를 그대로 씀
    torch.backends.cudnn.benchmark = False
    gpu = configs["gpus"]
    device = (torch.device(f"cuda:{gpu}")
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(
        configs["model"]["tse_model"])(**configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    # <<<<< 더한 것 - 산출물을 ckpt 별 폴더로 내림 (#96).
    #       원본은 audio/ · infer_utt_scores.csv · infer.log 를 exp_dir 바로 아래 뒀는데,
    #       그러면 같은 run 을 다른 ckpt 로 다시 추론할 때 앞 결과가 **통째로 덮어써져**
    #       ckpt 끼리 비교할 수가 없었음(934M 짜리 audio/ 까지 같이 날아감).
    #       stage 6 의 scoring/ 도 여기로 따라오도록 run.sh 가 이 경로를 넘김.
    #       기존 4 run 의 산출물은 옛 자리에 그대로 있으므로 집계 쪽이 둘 다 읽어야 함.
    out_dir = os.path.join(configs["exp_dir"], "infer",
                           os.path.splitext(os.path.basename(model_path))[0])
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(out_dir, "infer.log")
    logger.info(f"Results dir: {out_dir}")
    logger.info(f"Load checkpoint from {model_path}")
    # <<<<< 더한 것 - 학습 때와 같은 표를 추론 로그에도 남김 (#95). 원본 infer.py 는
    #       파라미터 수를 전혀 안 찍어, 어느 설정의 ckpt 인지 로그만 보고는 알 수 없었음
    logger.info(f"\n{pd.DataFrame(count_params(model)).to_string(index=False)}")
    save_audio_dir = os.path.join(out_dir, "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                print(f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                print(f"Error creating directory {save_audio_dir}: {e}")
        else:
            print(f"Directory {save_audio_dir} already exists.")
    else:
        print("Do NOT save the results in wav.")

    model = model.to(device)
    model.eval()

    test_spk_embeds = configs.get("test_spk_embeds", None)
    test_spk1_embed_scp = configs["test_spk1_enroll"]
    test_spk2_embed_scp = configs["test_spk2_enroll"]
    joint_training = configs["model_args"]["tse_model"].get(
        "joint_training", None)
    if not joint_training and test_spk_embeds:
        test_spk2embed_dict = read_vec_scp_file(test_spk_embeds)
    else:
        test_spk2embed_dict = read_label_file(configs["test_spk2utt"])

    test_spk1_embed = read_label_file(test_spk1_embed_scp)
    test_spk2_embed = read_label_file(test_spk2_embed_scp)

    lines = len(test_spk2embed_dict)

    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        test_spk2embed_dict,
        test_spk1_embed,
        test_spk2_embed,
        state="test",
        joint_training=joint_training,
        whole_utt=configs.get("whole_utt", True),
        repeat_dataset=configs.get("repeat_dataset", False),
    )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 collate_fn=tse_collate_fn_2spk)
    test_iter = lines // 2
    logger.info(f"test number: {test_iter}")

    with torch.no_grad():
        # <<<<< 고친 것 - 추론 진행 막대. test_iter 는 전체 기준이라
        #       --debug true 로 shard 를 잘라 쓰면 끝까지 안 차고 중간에 멈춤
        utts = tqdm.tqdm(test_dataloader, total=test_iter, desc="<infer>",
                         leave=False, dynamic_ncols=True,
                         disable=int(os.environ.get("RANK", 0)) != 0)
        for i, batch in enumerate(utts):
            features = batch["wav_mix"]
            targets = batch["wav_targets"]
            enroll = batch["spk_embeds"]
            spk = batch["spk"]
            key = batch["key"]

            features = features.float().to(device)  # (B,T,F)
            targets = targets.float().to(device)
            enroll = enroll.float().to(device)

            outputs = model(features, enroll)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            if sign_save_wav:
                file1 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[0]}-T{spk[0]}.wav",
                )
                soundfile.write(file1, outputs[0], sample_rate)
                file2 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[1]}-T{spk[1]}.wav",
                )
                soundfile.write(file2, outputs[1], sample_rate)

            ref = targets.cpu().numpy()
            ests = outputs
            mix = features.cpu().numpy()

            if ests[0].size != ref[0].size:
                end = min(ests[0].size, ref[0].size, mix[0].size)
                ests_1 = ests[0][:end]
                ref_1 = ref[0][:end]
                mix_1 = mix[0][:end]
                SISNR1, delta1 = cal_SISNRi(ests_1, ref_1, mix_1)
            else:
                SISNR1, delta1 = cal_SISNRi(ests[0], ref[0], mix[0])

            logger.info(
                f"Num={total_cnt + 1} | Utt={key[0]} | Target speaker={spk[0]} | "
                f"SI-SNR={SISNR1:.2f} | SI-SNRi={delta1:.2f}")
            # <<<<< 더한 것
            utt_rows.append({"key": key[0], "spk": spk[0],
                             "si_snr": SISNR1, "si_snri": delta1})
            total_SISNR += SISNR1
            total_SISNRi += delta1
            total_cnt += 1
            if delta1 > 1:
                accept_cnt += 1

            if ests[1].size != ref[1].size:
                end = min(ests[1].size, ref[1].size, mix[1].size)
                ests_2 = ests[1][:end]
                ref_2 = ref[1][:end]
                mix_2 = mix[1][:end]
                SISNR2, delta2 = cal_SISNRi(ests_2, ref_2, mix_2)
            else:
                SISNR2, delta2 = cal_SISNRi(ests[1], ref[1], mix[1])
            logger.info(
                f"Num={total_cnt + 1} | Utt={key[1]} | Target speaker={spk[1]} | "
                f"SI-SNR={SISNR2:.2f} | SI-SNRi={delta2:.2f}")
            # <<<<< 더한 것
            utt_rows.append({"key": key[1], "spk": spk[1],
                             "si_snr": SISNR2, "si_snri": delta2})
            total_SISNR += SISNR2
            total_SISNRi += delta2
            total_cnt += 1
            if delta2 > 1:
                accept_cnt += 1

            # if (i + 1) == test_iter:
            #     break
        end = time.time()
    # generate the scp file of the enhanced speech for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    # <<<<< 더한 것 - 발화별 SI-SNR·SI-SNRi 원값. stage 6 의 scoring/ 에는 SI-SNRi 가 없음
    utt_csv = os.path.join(out_dir, "infer_utt_scores.csv")
    pd.DataFrame(utt_rows).to_csv(utt_csv, index=False)
    logger.info(f"Per-utterance scores saved to {utt_csv}")

    logger.info(f"Time Elapsed: {end - start:.1f}s")
    logger.info(f"Average SI-SNR: {total_SISNR / total_cnt:.2f}")
    logger.info(f"Average SI-SNRi: {total_SISNRi / total_cnt:.2f}")
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: "
        f"{accept_cnt / total_cnt * 100:.2f}")


if __name__ == "__main__":
    fire.Fire(infer)
