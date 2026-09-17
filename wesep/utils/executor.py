# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
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

import os
from contextlib import nullcontext

import tableprint as tp
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# if your python version < 3.7 use the below one
import torch

from wesep.utils.funcs import clip_gradients, compute_fbank, apply_cmvn
# <<<<< 더한 것 - 스텝 단위 CSV·텐서보드 기록
from wesep.utils.metric_logger import log_step
import random


class Executor:

    # <<<<< 고친 것 - mlog 를 생성자로 받음.
    #       metric_logger.make_logger() 가 만든 dict 또는 None.
    #       안 넘기면 None 이라 스텝 기록만 안 할 뿐 기존 동작 그대로임
    def __init__(self, mlog=None):
        self.step = 0
        self.mlog = mlog

    # <<<<< 더한 것 - logger.info 가 tqdm 막대와 같은 줄에 겹쳐 찍히는 것을 막음.
    #       이 함수 안의 로깅을 tqdm.write 로 흘려보내 막대를 지웠다 다시 그리게 함.
    #       괄호가 있어야 함 — @contextmanager 객체를 만들어 데코레이터로 쓰는 것임
    @logging_redirect_tqdm()
    def train(
            self,
            dataloader,
            models,
            epoch_iter,
            optimizers,
            criterion,
            schedulers,
            scaler,
            epoch,
            enable_amp,
            logger,
            clip_grad=5.0,
            log_batch_interval=100,
            device=torch.device("cuda"),
            se_loss_weight=1.0,
            multi_task=False,
            SSA_enroll_prob=0,
            fbank_args=None,
            sample_rate=16000,
            speaker_feat=True
    ):
        """Train one epoch"""
        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            # <<<<< 고친 것 - 스텝 진행 막대. 0번 GPU 에서만 그림.
            #       표(tableprint) 는 log_batch_interval 마다만 찍히므로 그 사이를 막대가 메움
            steps = tqdm.tqdm(dataloader, total=epoch_iter, desc=f"<Executor> TRAIN {epoch}",
                              leave=False, dynamic_ncols=True,
                              disable=int(os.environ.get("RANK", 0)) != 0)
            for i, batch in enumerate(steps):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                # embeddings when not joint training, enrollment wavforms
                # when joint training
                enroll = batch["spk_embeds"]
                # spk_lable is an empty list when not joint training
                # and multi-task
                spk_label = batch["spk_label"]

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                spk_label = spk_label.to(device)

                # <<<<< 고친 것 - torch.cuda.amp.* 가 FutureWarning 을 냄. torch.amp.* 로 옮김.
                #       두 API 는 같은 구현임 (torch.cuda.amp 쪽이 torch.amp 를 상속해 super() 를 부름).
                #       device_type 은 위에서 이미 정해 둔 device 를 그대로 씀 - cpu 로 돌려도 깨지지 않게
                with torch.amp.autocast(device.type, enabled=enable_amp):
                    if SSA_enroll_prob > 0:
                        if SSA_enroll_prob > random.random():
                            with torch.no_grad():
                                outputs = model(features, enroll)
                                est_speech = outputs[0]
                                self_fbank = est_speech
                                if fbank_args is not None and speaker_feat:
                                    self_fbank = compute_fbank(
                                        est_speech, **fbank_args,
                                        sample_rate=sample_rate)
                                    self_fbank = apply_cmvn(self_fbank)
                            outputs = model(features, self_fbank)
                        else:
                            outputs = model(features, enroll)
                    else:
                        outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    loss = 0
                    for ii in range(len(criterion)):
                        # se_loss_weight: ([position in outputs[0], [1]],
                        #                 [weights:[1.0], [0.5]])
                        for ji in range(len(se_loss_weight[0][ii])):
                            if (multi_task and criterion[ii].__class__.__name__
                                    == "CrossEntropyLoss"):
                                loss += se_loss_weight[1][ii][ji] * (
                                    criterion[ii](
                                        outputs[se_loss_weight[0][ii][ji]],
                                        spk_label,
                                    ).mean() / accum_grad)
                                continue
                            loss += se_loss_weight[1][ii][ji] * (criterion[ii](
                                outputs[se_loss_weight[0][ii][ji]],
                                targets).mean() / accum_grad)

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # <<<<< 더한 것 - tracker_step_interval 스텝마다 CSV·텐서보드에 기록.
                #       아래 dict 의 key 가 metrics_step.csv 의 열 이름이 됨 —
                #       지표를 늘리려면 여기에 key 를 하나 더 넣으면 됨.
                #       cur_iter 는 위에서 이미 계산해 둔 전역 스텝 번호임.
                #       간격 판정과 mlog=None 처리는 log_step 안에서 하므로
                #       여기서는 매 스텝 그냥 부르면 됨
                log_step(self.mlog, {
                    "epoch": epoch,
                    "step": i + 1,
                    "global_step": cur_iter,
                    "loss": losses[-1],
                    "running_mean": total_loss_avg,
                    "lr": optimizer.param_groups[0]["lr"],
                })

                # updata the model
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "TRAIN",
                                epoch,
                                i + 1,
                                total_loss_avg * accum_grad,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg, 0

    @logging_redirect_tqdm()
    def cv(
            self,
            dataloader,
            models,
            val_iter,
            criterion,
            epoch,
            enable_amp,
            logger,
            log_batch_interval=100,
            device=torch.device("cuda"),
    ):
        """Cross validation on"""
        model = models[0]

        model.eval()
        log_interval = log_batch_interval
        losses = []

        with torch.no_grad():
            # <<<<< 고친 것 - 검증도 같은 방식으로 막대를 둠
            steps = tqdm.tqdm(dataloader, total=val_iter, desc=f"<Executor> VAL   {epoch}",
                              leave=False, dynamic_ncols=True,
                              disable=int(os.environ.get("RANK", 0)) != 0)
            for i, batch in enumerate(steps):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                enroll = batch["spk_embeds"]

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)

                # <<<<< 고친 것 - torch.cuda.amp.* 가 FutureWarning 을 냄. torch.amp.* 로 옮김.
                #       두 API 는 같은 구현임 (torch.cuda.amp 쪽이 torch.amp 를 상속해 super() 를 부름).
                #       device_type 은 위에서 이미 정해 둔 device 를 그대로 씀 - cpu 로 돌려도 깨지지 않게
                with torch.amp.autocast(device.type, enabled=enable_amp):
                    outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    # By default, the first loss is used as the indicator
                    # of the validation set.
                    loss = criterion[0](outputs[0], targets).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == val_iter:
                    break
        return total_loss_avg, 0
