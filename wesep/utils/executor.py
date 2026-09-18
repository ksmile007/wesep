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
# <<<<< 더한 것 - 에포크 손실을 rank 끼리 합치는 데 씀
import torch.distributed as dist

from wesep.utils.funcs import clip_gradients, compute_fbank, apply_cmvn
# <<<<< 더한 것 - CSV·텐서보드·wandb 기록을 맡는 클래스
from wesep.utils.tracker import Tracker
import random


class Executor:

    # <<<<< 고친 것 - 기록은 Tracker 가 다 함.
    #       configs 를 안 넘기면 Tracker 가 스스로 꺼져 아무것도 기록하지 않음 -
    #       wesep 원본 동작 그대로임
    def __init__(self, configs=None, logger=None):
        self.step = 0
        self.tracker = Tracker(configs, logger)
        # <<<<< 더한 것 - cv() 는 전역 스텝을 모르는데 그래프의 x축으로 필요함.
        #       train() 이 여기에 두고 가므로 **train() 을 먼저 불러야 함** -
        #       안 부르면 None 이라 log_epoch 이 바로 KeyError 로 멈춤
        self._global_step = None

    # <<<<< 더한 것 - tfevents 를 닫고 wandb run 을 마감함
    def close(self):
        self.tracker.close()

    # <<<<< 더한 것 - 에포크 손실을 전체 데이터 기준 평균으로 만듦.
    #       rank 마다 자기 배치만 보므로 (가중합, 샘플 수) 를 한 번에 더해 나눔 —
    #       평균을 다시 평균내면 rank 별 샘플 수가 다를 때 틀림.
    #       집합통신이라 **모든 rank 가 불러야 함.** rank 가드 안에 두면 영구 대기함
    @staticmethod
    def _global_mean(total_loss, n_samples, device):
        if not (dist.is_available() and dist.is_initialized()):
            return total_loss / n_samples
        # float64 로 둠 - 기본 float32 면 끝자리가 1e-07 쯤 깎여
        # 단일 GPU 경로(파이썬 float)와 값이 갈림(실측)
        stat = torch.tensor([total_loss, float(n_samples)],
                            dtype=torch.float64, device=device)
        dist.all_reduce(stat, op=dist.ReduceOp.SUM)
        return (stat[0] / stat[1]).item()

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
            speaker_feat=True,
            # <<<<< 더한 것 - autocast 의 dtype. None 이면 torch 기본값(fp16).
            #       precision='bf16-mixed' 면 bfloat16 이 들어옴
            amp_dtype=None,
    ):
        """Train one epoch"""
        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []
        # <<<<< 더한 것 - 누적합을 들고 감. sum(losses) 를 매 스텝 다시 도는 것은
        #       스텝 수에 대해 O(n^2) 임 - 값은 비트 단위로 같음(실측).
        #       배치 평균이 아니라 **샘플 수로 가중**해 더함 - 마지막 배치가 작아도 맞음
        total_loss = 0.0
        n_samples = 0

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
                with torch.amp.autocast(device.type, enabled=enable_amp, dtype=amp_dtype):
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

                # targets 의 0번 축이 criterion 이 평균낸 샘플 수임
                # (tse_collate_fn 이 혼합 하나를 화자 수만큼 늘려 놓은 뒤의 값)
                n_batch = targets.shape[0]
                losses.append(loss.item())
                total_loss += losses[-1] * n_batch
                n_samples += n_batch
                total_loss_avg = total_loss / n_samples

                # <<<<< 더한 것 - 막대 오른쪽에 손실을 띄움. 표는 log_interval 마다만 찍혀
                #       그 사이가 깜깜했음. loss 는 그 배치, avg_loss 는 에포크 누적 평균임
                steps.set_postfix(
                    {"loss":        f"{losses[-1] * accum_grad:.4f}",
                     "avg_loss":    f"{total_loss_avg * accum_grad:.4f}"}
                )

                # <<<<< 더한 것 - tracker_step_interval 스텝마다 CSV·텐서보드에 기록.
                #       아래 dict 의 key 가 metrics.csv 의 열 이름이 됨 —
                #       지표를 늘리려면 여기에 key 를 하나 더 넣으면 됨.
                #       cur_iter 는 위에서 이미 계산해 둔 전역 스텝 번호임.
                #       간격 판정과 rank 판정은 Tracker 가 하므로
                #       여기서는 매 스텝 그냥 부르면 됨
                self.tracker.log_step({
                    "epoch": epoch,
                    "global_step": cur_iter,
                    "train/loss_step": losses[-1],
                    "train/loss_running_step": total_loss_avg,
                    "train/lr_step": optimizer.param_groups[0]["lr"],
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
            # <<<<< 더한 것 - 여기까지는 이 rank 가 본 배치만의 평균임
            total_loss_avg = self._global_mean(total_loss, n_samples, device)

            # <<<<< 더한 것 - 학습 에포크 지표를 값이 생긴 자리에서 기록함.
            #       cur_iter 는 cv() 도 x축으로 써야 하므로 self 에 남겨 둠
            self._global_step = cur_iter
            self.tracker.log_epoch({
                "epoch": epoch,
                "global_step": cur_iter,
                "train/loss_epoch": total_loss_avg,
                "train/lr_epoch": optimizer.param_groups[0]["lr"],
            })
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
            # <<<<< 더한 것 - train() 과 같음
            amp_dtype=None,
    ):
        """Cross validation on"""
        model = models[0]

        model.eval()
        log_interval = log_batch_interval
        losses = []
        total_loss = 0.0    # <<<<< 더한 것 - train() 과 같은 이유
        n_samples = 0

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
                with torch.amp.autocast(device.type, enabled=enable_amp, dtype=amp_dtype):
                    outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    # By default, the first loss is used as the indicator
                    # of the validation set.
                    loss = criterion[0](outputs[0], targets).mean()

                n_batch = targets.shape[0]
                losses.append(loss.item())
                total_loss += losses[-1] * n_batch
                n_samples += n_batch
                total_loss_avg = total_loss / n_samples

                # <<<<< 더한 것 - 학습 막대와 같은 표기. 검증은 중간 로그가 없어
                #       끝나야 값이 보였음
                steps.set_postfix(
                    {"loss":        f"{losses[-1]:.4f}",
                     "avg_loss":    f"{total_loss_avg:.4f}"}
                )

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == val_iter:
                    break

        # <<<<< 더한 것 - train() 과 같은 이유로 rank 끼리 합침.
        #       with torch.no_grad() 블록 밖이지만 모든 rank 가 지나는 자리임
        total_loss_avg = self._global_mean(total_loss, n_samples, device)

        # <<<<< 더한 것 - 검증 에포크 지표. 학습 쪽은 train() 이 따로 기록함.
        #       x축은 train() 이 남긴 전역 스텝을 씀 - cv() 는 그 값을 모름.
        #       rank 0 이 아니면 Tracker 가 꺼져 있어 바로 빠짐
        self.tracker.log_epoch({
            "epoch": epoch,
            "global_step": self._global_step,
            "val/loss": total_loss_avg,
        })
        return total_loss_avg, 0
