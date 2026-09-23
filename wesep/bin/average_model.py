# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os
import os.path
import re

import torch


def get_args():
    parser = argparse.ArgumentParser(description="average model")
    # <<<<< 고친 것 - required 를 뗌. 아래 --dst_dir 과 둘 중 하나만 주면 됨 (#96)
    parser.add_argument("--dst_model", default=None, help="averaged model")
    # <<<<< 더한 것 - 이 폴더 안에 이름을 **평균한 내용으로 지어** 저장함 (#96).
    #       원본은 --dst_model 로 avg_best_model.pt 를 고정해 받았는데, 그러면
    #       한 run 에 평균 ckpt 를 하나밖에 못 두어 여러 조합을 비교할 수 없었음.
    #       이름을 파이썬이 짓는 이유 - final 모드는 여기서 glob·정렬해야 실제 epoch 목록을
    #       알 수 있어, shell 이 같은 계산을 또 하게 만들지 않으려는 것임
    parser.add_argument("--dst_dir", default=None,
                        help="폴더만 주면 평균한 epoch 으로 이름을 지어 저장함")
    parser.add_argument("--src_path",
                        required=True,
                        help="src model path for average")
    parser.add_argument("--num",
                        default=5,
                        type=int,
                        help="nums for averaged model")
    parser.add_argument(
        "--min_epoch",
        default=0,
        type=int,
        help="min epoch used for averaging model",
    )
    parser.add_argument(
        "--max_epoch",
        default=65536,  # Big enough
        type=int,
        help="max epoch used for averaging model",
    )
    parser.add_argument(
        "--mode",
        default="final",
        type=str,
        help="use last epochs for average or best epochs",
    )
    parser.add_argument(
        "--epochs",
        default="1,2,3,4,5",
        type=str,
        help="use last epochs for average or best epochs",
    )
    args = parser.parse_args()
    print(args)
    return args


def averaged_name(epochs):
    """평균한 epoch 목록으로 파일 이름을 짓는다.

        [150]                -> avg_ep150.pt
        [138, 141]           -> avg_ep138+141.pt
        [141 .. 150]         -> avg_n10_ep141~150.pt

    구분자 선택 근거(실측) - `+` 는 따옴표 없이 shell 에 넘겨도 CSV 한 칸에 넣어도 그대로이고,
    `&` 는 shell 이 백그라운드 연산자로 잘라 먹고, `,` 는 CSV 가 따옴표로 감싼다.
    범위에 `-` 가 아니라 `~` 를 쓰는 이유는 이 저장소에서 `-` 가 이미 변형 구분자이기 때문이다
    (`bsrnn_ecapa_FiLM-16` 의 `-16` 은 `16-mixed` 의 약자). 이름 중간의 `~` 는 shell 이
    전개하지 않고 `*~` 글롭·.gitignore 에도 안 걸린다 - 그 패턴은 **끝**이 `~` 인 것만 잡는다.

    정확한 목록은 파일 안의 `avg_epochs` 에 들어간다. 이름은 요약일 뿐이다 -
    이름은 옮기다 바뀔 수 있지만 파일 안의 키는 따라다닌다.
    """
    epochs = sorted(epochs)
    if len(epochs) <= 3:
        return f"avg_ep{'+'.join(str(e) for e in epochs)}.pt"
    return f"avg_n{len(epochs)}_ep{epochs[0]}~{epochs[-1]}.pt"


def main():
    args = get_args()
    assert (args.dst_model is None) != (args.dst_dir is None), \
        "--dst_model 과 --dst_dir 중 정확히 하나만 줄 것"
    if args.mode == "final":
        # <<<<< 고친 것 - 원본은 `*[!avg][!final][!latest].pt` 로 걸렀는데, 그 대괄호는
        #       "그 글자들 중 하나가 아닌 **한 글자**" 라는 뜻이라 확장자 앞 세 글자만 봄.
        #       avg_best_model.pt 는 `del` 의 `l` 이 [!latest] 에 걸려 우연히 빠졌을 뿐이고,
        #       #96 의 새 이름(avg_ep150.pt · avg_n10_ep141~150.pt)은 **그대로 통과**해
        #       아래 checkpoint_ 정규식에서 IndexError 가 났음(실측).
        #       평균 대상은 원본 ckpt 뿐이므로 그것만 정확히 집는다 -
        #       바로 아래 정규식이 기대하는 이름과도 같아짐
        path_list = glob.glob(f"{args.src_path}/checkpoint_*.pt")
        path_list = sorted(
            path_list,
            key=lambda p: int(re.findall(r"(?<=checkpoint_)\d*(?=.pt)", p)[0]),
        )
        path_list = path_list[-args.num:]
    else:
        epoch_indexes = list(args.epochs.split(","))
        path_list = [
            os.path.join(args.src_path, "checkpoint_" + x + ".pt")
            for x in epoch_indexes
        ]
    print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print(f"Processing {path}")
        states = torch.load(path, map_location=torch.device("cpu"))
        states = states["models"][0] if "models" in states else states
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    # <<<<< 고친 것 - 원본은 {"models": [avg]} 로 덮어써 epoch·val_loss 를 전부 버렸음 (#96).
    #       그래서 나중에 이 파일이 무엇인지 알려면 원본 ckpt 와 하나씩 대조해야 했음 -
    #       기존 4 run 의 avg_best_model.pt 가 실은 checkpoint_150.pt 와 비트 동일이라는 것도
    #       그렇게 알아냈음(실측). 무엇을 평균했는지 파일 안에 남김.
    #       키 이름은 checkpoint.py 의 EXTRA_KEYS 와 겹치지 않게 avg_ 접두사를 붙이고,
    #       `epoch` 만은 같은 자리를 써서 읽는 쪽이 원본 ckpt 와 같게 다룰 수 있게 함
    epochs = [int(re.findall(r"(?<=checkpoint_)\d+(?=\.pt)", p)[0]) for p in path_list]
    avg = {
        "models": [avg],
        "epoch": max(epochs),          # 대표 epoch. 원본 ckpt 의 같은 키와 같은 뜻
        "avg_mode": args.mode,
        "avg_epochs": sorted(epochs),  # 실제로 평균한 목록. 이름은 요약이고 이쪽이 정본
        "num_avg": num,
        "src_checkpoints": [os.path.basename(p) for p in path_list],
        "torch_version": str(torch.__version__),
    }
    dst = args.dst_model or os.path.join(args.dst_dir, averaged_name(epochs))
    print(f"Saving to {dst}")
    torch.save(avg, dst)
    # run.sh 가 $(...) 로 받아 쓰도록 마지막 줄에 경로만 찍음
    print(dst)


if __name__ == "__main__":
    main()
