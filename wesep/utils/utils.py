# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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
import difflib
import logging
import os
import random
import shutil
import sys
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def get_logger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ %(levelname)s : %(asctime)s ] - %(message)s",
    )
    # <<<<< 더한 것 - fsspec 이 파일을 열 때마다 "open file: ..." 을 DEBUG 로 찍음.
    #       Tracker 가 기록할 때마다 CSVLogger.save() 를 불러 그 줄이 쏟아졌음.
    #       train.log 가 아니라 **터미널(stderr)** 로 감 - 이 로거에는 핸들러가 없어
    #       root 의 StreamHandler 로만 전파되기 때문임
    #       위 basicConfig 의 DEBUG 는 원본 그대로 두고 이 로거만 올림
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logger = logging.getLogger("Pyobj, f")
    # <<<<< 더한 것 - 이 로거의 레벨을 직접 박음. 위 basicConfig 에만 기대면 infer.log 가 빈다 (#97).
    #       infer.py 는 @logging_redirect_tqdm() 로 감싼 함수 안에서 이 함수를 부르는데,
    #       그 컨텍스트가 본문보다 먼저 들어가 root 에 _TqdmLoggingHandler 를 꽂는다.
    #       basicConfig 는 root 에 핸들러가 이미 있으면 **아무것도 안 하므로**(force 기본 False)
    #       level=DEBUG 도 안 먹고 root 가 기본값 WARNING 으로 남는다. 이 로거는 NOTSET 이라
    #       유효 레벨이 root 를 따라 WARNING 이 되어 logger.info() 가 전부 걸러졌다 -
    #       기존 4 run 의 infer.log 가 전부 0 바이트인 원인임(실측).
    #       force=True 로 basicConfig 를 다시 부르면 tqdm 핸들러가 날아가 막대 겹침이 되돌아온다.
    #       root 를 건드리지 않고 이 로거만 올리는 것이 가장 좁은 고침임.
    logger.setLevel(logging.DEBUG)
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def setup_logger(rank, exp_dir, device_ids, MAX_NUM_LOG_FILES: int = 100):
    model_dir = os.path.join(exp_dir, "models")
    file_name = "train.log"
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        for i in range(MAX_NUM_LOG_FILES - 1, -1, -1):
            if i == 0:
                p = Path(os.path.join(exp_dir, file_name))
                pn = p.parent / (p.stem + ".1" + p.suffix)
            else:
                _p = Path(os.path.join(exp_dir, file_name))
                p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
                pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

            if p.exists():
                if i == MAX_NUM_LOG_FILES - 1:
                    p.unlink()
                else:
                    shutil.move(p, pn)
    dist.barrier(device_ids=[device_ids])  # let the rank 0 mkdir first
    return get_logger(exp_dir, file_name)


def load_config_with_base(config_file):
    """hydra 의 defaults 리스트까지 합성한 config 를 순수 dict 으로 돌려준다 (#83).

    SD-FiLM 저장소(src/train.py)는 @hydra.main 으로 진입점 자체를 hydra 에 넘기지만,
    여기서는 Compose API 만 쓴다 - wesep 의 진입점은 fire.Fire(train) 이고 run.sh 가
    `--config <경로>` 로 부르므로, @hydra.main 을 쓰면 CLI 문법이 key=value 로 바뀌고
    torchrun 의 rank 마다 outputs/ 를 만들며 cwd 까지 옮긴다. compose 는 그 셋을 건드리지
    않고 defaults 합성만 가져온다.

    defaults 에 적는 이름은 확장자 없는 파일 이름이고, 같은 폴더의 yaml 을 가리킨다.
    `- _self_` 를 마지막에 두어야 이 파일의 키가 부모를 덮어쓴다.
    resolve=True 로 ${...} 보간까지 풀어 돌려주므로 호출부는 dict 그대로 쓰면 된다.
    defaults 가 없는 기존 config 13벌은 yaml.load 결과와 완전히 같음을 실측했다.
    """
    path = os.path.abspath(config_file)          # initialize_config_dir 은 절대 경로만 받는다
    config_dir = os.path.dirname(path)           # 예: <...>/tse/v2/confs
    file_name = os.path.basename(path)           # 예: bsrnn_ecapa_sdfilm.yaml
    config_name, _ext = os.path.splitext(file_name)   # 예: bsrnn_ecapa_sdfilm
    # config_name 에 확장자를 붙이면 hydra 가 <이름>.yaml.yaml 을 찾아 실패한다.
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return OmegaConf.to_container(cfg, resolve=True)


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for conf
    """
    yaml_config = load_config_with_base(config_file)
    # values from conf file are all possible params
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    # passed kwargs will override yaml conf
    # for key in kwargs.keys():
    #    assert key in yaml_config, "Parameter {} invalid!\n".format(key)
    # add the path of config file to dict
    if "config" not in kwargs:
        kwargs["config"] = config_file
    return dict(yaml_config, **kwargs)


def validate_path(dir_name):
    """Create the directory if it doesn't exist
    :param dir_name
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ""):
        os.makedirs(dir_name)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def generate_enahnced_scp(directory: str, extension: str = "wav"):
    source_dir = Path(directory)
    spk_scp = source_dir.joinpath("spk1.scp")
    audio_list = []

    for file_path in source_dir.rglob(f"*.{extension}"):
        audio_list.append(file_path)

    with open(spk_scp, "w") as f:
        for audio in audio_list:
            path = str(audio.resolve())
            ori_filename = audio.stem
            spk1_id = ori_filename.split("-")[1]
            # spk2_id = ori_filename.split("_")[1].split("-")[0]
            curr_spk = ori_filename.split("T")[1]
            prefix = "s1" if curr_spk == spk1_id else "s2"
            f_dash_index = ori_filename.find("-")
            l_dash_index = ori_filename.rfind("-")
            filename = ori_filename[f_dash_index + 1:l_dash_index]
            final_filename = prefix + "/" + filename + ".wav"
            line = final_filename + " " + path
            f.write(line + "\n")


def get_commandline_args():
    # ported from
    # https://github.com/espnet/espnet/blob/master/espnet/utils/cli_utils.py
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [(arg.replace("'", "'\\''") if all(
        char not in arg
        for char in extra_chars) else "'" + arg.replace("'", "'\\''") + "'")
        for arg in sys.argv]

    return sys.executable + " " + " ".join(argv)


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/utils/config_argparse.py
class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not isinstance(d, dict):
                self.error("Config file has non dict value: {_args.config}")

            for key in d:
                for action in self._actions:
                    if key == action.dest:
                        break
                else:
                    self.error(
                        f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't
            #             check type. i.e. We can set any type value
            #             regardless of argument type.
            self.set_defaults(**d)
        return super().parse_known_args(args, namespace)


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library
                        (e.g. .'elu').
        library (module): Name of library/module where to search for
                          object handler with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer
                                e.g. (torch.nn.ELU)

    """

    all_torch_layers = list(dir(torch.nn))
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            f"Layer with name {l_name} not found in {str(library)}.\n "
            f"Closest matches: {close_matches}")
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            f"Multiple matchs for layer with name {l_name} not found in {str(library)}.\n "
            f"All matches: {close_matches}")
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler


# def spk2id(utt_spk_list):
#     _, spk_list = zip(*utt_spk_list)
#     spk_list = sorted(list(set(spk_list)))  # remove overlap and sort

#     spk2id_dict = {}
#     spk_list.sort()
#     for i, spk in enumerate(spk_list):
#         spk2id_dict[spk] = i
#     return spk2id_dict
