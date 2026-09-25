from __future__ import print_function

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from wespeaker.models.speaker_model import get_speaker_model

from wesep.modules.common.speaker import FUSE_TYPE_ALIASES   # <<<<< 더한 것 - 옛 표기 'sdfilm' 받기
from wesep.modules.common.speaker import PreEmphasis
from wesep.modules.common.speaker import SpeakerFuseLayer
from wesep.modules.common.speaker import SpeakerTransform


class ResRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * 2,
                              input_size)   # hidden_size = feature_dim * 2

    def forward(self, input):
        # input shape: batch, dim, seq

        # band_rnn 호출 시: B'=B*nband, C=N, L=T
        # band_comm 호출 시: B'=B*T, C=N, L=nband
        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())   # (B', L, 2*hidden)
        rnn_output = self.proj(rnn_output.contiguous().view(
            -1, rnn_output.shape[2])).view(input.shape[0], input.shape[2],
                                           input.shape[1])   # (B', L, C)

        return input + rnn_output.transpose(1, 2).contiguous()   # (B', C, L)


"""
TODO : attach the speaker embedding to each input
Input shape:(B,feature_dim + spk_emb_dim , T)
"""


class BSNet(nn.Module):

    def __init__(self, in_channel, nband=7, bidirectional=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim,
                               self.feature_dim * 2,
                               bidirectional=bidirectional)
        self.band_comm = ResRNN(self.feature_dim,
                                self.feature_dim * 2,
                                bidirectional=bidirectional)

    def forward(self, input, dummy: Optional[torch.Tensor] = None):
        # input shape: B, nband*N, T
        # 지역변수 N = nband*feature_dim. 아래 주석의 N 은 feature_dim
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim,
                       -1)).view(B, self.nband, -1, T)   # (B, nband, N, T)

        # band comm
        band_output = (band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband))   # (B*T, N, nband)
        output = (self.band_comm(band_output).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous())   # (B, nband, N, T)

        return output.view(B, N, T)   # (B, nband*N, T)


class FuseSeparation(nn.Module):

    def __init__(
        self,
        nband=7,
        num_repeat=6,
        feature_dim=128,
        spk_emb_dim=256,
        spk_fuse_type="concat",
        multi_fuse=True,
        spk_fuse_kwargs=None,   # <<<<< 더한 것 - SD-FiLM 하이퍼파라미터 통로 (#83)
    ):
        """

        :param nband : len(self.band_width)
        """
        super(FuseSeparation, self).__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList([])
        if self.multi_fuse:
            for _ in range(num_repeat):
                self.separation.append(
                    SpeakerFuseLayer(
                        embed_dim=spk_emb_dim,
                        feat_dim=feature_dim,
                        fuse_type=spk_fuse_type,
                        spk_fuse_kwargs=spk_fuse_kwargs,   # <<<<< 더한 것 (#83)
                    ))
                self.separation.append(BSNet(nband * feature_dim, nband))
        else:
            self.separation.append(
                SpeakerFuseLayer(
                    embed_dim=spk_emb_dim,
                    feat_dim=feature_dim,
                    fuse_type=spk_fuse_type,
                    spk_fuse_kwargs=spk_fuse_kwargs,   # <<<<< 더한 것 (#83)
                ))
            for _ in range(num_repeat):
                self.separation.append(BSNet(nband * feature_dim, nband))

    def forward(self, x, spk_embedding, nch: torch.Tensor = torch.tensor(1)):
        """
        x: [B, nband, feature_dim, T]
        out: [B, nband, feature_dim, T]
        """
        batch_size = x.shape[0]

        # multi_fuse=True 시, SpeakerFuseLayer 와 BSNet 이 번갈아 나옴
        # 그래서 x 가 (B, nband, N, T) 와 (B, nband*N, T) 를 오감
        if self.multi_fuse:
            for i, sep_func in enumerate(self.separation):
                x = sep_func(x, spk_embedding)
                if i % 2 == 0:
                    x = x.view(batch_size * nch, self.nband * self.feature_dim,
                               -1)   # (B, nband*N, T)
                else:
                    x = x.view(batch_size * nch, self.nband, self.feature_dim,
                               -1)   # (B, nband, N, T)
        else:
            x = self.separation[0](x, spk_embedding)                          # (B, nband, N, T)
            x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)   # (B, nband*N, T)
            for idx, sep in enumerate(self.separation):
                if idx > 0:
                    x = sep(x, spk_embedding)   # (B, nband*N, T)
            x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)   # (B, nband, N, T)
        return x


class BSRNN(nn.Module):
    # self, sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
    # use_bidirectional=True
    def __init__(
        self,
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        use_spk_transform=True,
        use_bidirectional=True,
        spk_fuse_type="concat",
        spk_fuse_kwargs=None,   # <<<<< 더한 것 - SD-FiLM 하이퍼파라미터 통로 (#83)
        multi_fuse=True,
        joint_training=True,
        multi_task=False,
        spksInTrain=251,
        spk_model=None,
        spk_model_init=None,
        spk_model_freeze=False,
        spk_model_eval=False,   # <<<<< 더한 것 - 동결 화자 인코더를 학습 중에도 eval 로 둘지 (#85)
        spk_args=None,
        spk_feat=False,
        feat_type="consistent",
    ):
        super(BSRNN, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = spk_emb_dim
        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.spk_model_eval = spk_model_eval   # <<<<< 더한 것 (#85)
        spk_fuse_type = FUSE_TYPE_ALIASES.get(spk_fuse_type, spk_fuse_type)   # <<<<< 더한 것 - 옛 표기를 정식 이름으로
        self.spk_fuse_type = spk_fuse_type     # <<<<< 더한 것 - forward 의 조건 모양 분기용 (#81)
        self.multi_task = multi_task

        # <<<<< 더한 것 - SD-FiLM 은 프레임 시퀀스 (B, 512, T_spk) 를 조건으로 받는데
        #       SpeakerTransform 은 embed_dim=256 으로 만들어져 512 채널을 못 받는다.
        #       Table 2 config 4벌이 use_spk_transform: False 라 실제로 걸리지 않지만,
        #       True 로 켜면 조용히 죽는 대신 여기서 이유를 말하고 멈춘다 (#83).
        if spk_fuse_type == "SDFiLM" and use_spk_transform:
            raise ValueError(
                "spk_fuse_type='SDFiLM' 은 use_spk_transform=True 와 같이 못 쓴다 - "
                "SpeakerTransform 이 풀링 벡터(embed_dim)용이라 프레임 채널 512 를 "
                "받지 못한다 (#83).")

        # 0-1k (100 hop), 1k-4k (250 hop),
        # 4k-8k (500 hop), 8k-16k (1k hop),
        # 16k-20k (2k hop), 20k-inf

        # 0-8k (1k hop), 8k-16k (2k hop), 16k
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))

        # add up to 8k
        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k] * 1

        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)

        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        if joint_training:
            self.spk_model = get_speaker_model(spk_model)(**spk_args)
            if spk_model_init:
                pretrained_model = torch.load(spk_model_init)
                state = self.spk_model.state_dict()
                for key in state.keys():
                    if key in pretrained_model.keys():
                        state[key] = pretrained_model[key]
                        # print(key)
                    else:
                        print("not %s loaded" % key)
                self.spk_model.load_state_dict(state)
            if spk_model_freeze:
                for param in self.spk_model.parameters():
                    param.requires_grad = False
            if not spk_feat:
                if feat_type == "consistent":
                    self.preEmphasis = PreEmphasis()
                    self.spk_encoder = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sr,
                        n_fft=win,
                        win_length=win,
                        hop_length=stride,
                        f_min=20,
                        window_fn=torch.hamming_window,
                        n_mels=spk_args["feat_dim"],
                    )
            else:
                self.preEmphasis = nn.Identity()
                self.spk_encoder = nn.Identity()

            if multi_task:
                self.pred_linear = nn.Linear(spk_emb_dim, spksInTrain)
            else:
                self.pred_linear = nn.Identity()

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                ))

        self.separator = FuseSeparation(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            spk_emb_dim=spk_emb_dim,
            spk_fuse_type=spk_fuse_type,
            spk_fuse_kwargs=spk_fuse_kwargs,   # <<<<< 더한 것 (#83)
            multi_fuse=multi_fuse,
        )

        # self.proj =  nn.Linear(hidden_size*2, input_size)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim,
                                 torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                ))

    # <<<<< 더한 것 - 화자 인코더를 학습 중에도 eval 로 둠 (#85).
    #       spk_model_freeze 와는 다른 축임 - 그쪽은 requires_grad(gradient)를 끄고
    #       이쪽은 BatchNorm·Dropout 의 모드를 끈다. requires_grad=False 만으로는
    #       BN 의 running_mean/var 가 안 멈춘다(파라미터가 아니라 버퍼라서) -
    #       학습 완료 ckpt 대조에서 BN 통계 58개가 전부 움직였음.
    #       두 플래그를 엮지 않는 이유 - 가중치는 학습하되 BN 만 고정하는 것도
    #       쓰이는 기법이고(Frozen BatchNorm), 엮으면 그 조합이 조용히 무시된다.
    #
    #       __init__ 이 아니라 여기인 이유 - nn.Module.train() 이 자식 전부를 재귀로
    #       train 모드로 되돌린다. executor.py 의 model.train() 이 매 에포크 부르므로
    #       __init__ 에서 한 번 꺼 두면 첫 에포크에 되살아남(실측).
    #       joint_training 은 존재 확인용임 - 그 분기 안에서만 self.spk_model 이 만들어짐.
    #       기본값이 False 라 기존 run 은 동작이 그대로임 - #72 Table 2 재현 경로 보존.
    #       근거는 SD-FiLM 저장소의 docs/issues/wesep_frozen_encoder_bn_drift.md
    def train(self, mode: bool = True):
        super().train(mode)
        if self.spk_model_eval and self.joint_training:
            self.spk_model.eval()
        return self

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, embeddings):
        # input shape: (B, C, T)

        # B: 배치, nband: 서브밴드(32), N: feature_dim, emb: spk_emb_dim
        # T: 프레임, t: 샘플, F: enc_dim(win//2+1), BW: band_width[i]
        # T_spk, t_spk: 등록 발화 쪽 길이
        wav_input = input            # (B, t)
        spk_emb_input = embeddings   # (B, T_spk, 80) 또는 (B, t_spk)
        batch_size, nsample = wav_input.shape
        nch = 1

        # frequency-domain separation
        spec = torch.stft(
            wav_input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            return_complex=True,
        )   # (B, F, T) 복소수

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)   # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx +
                                        self.band_width[i]].contiguous())   # nband 개 x (B, 2, BW, T)
            subband_mix_spec.append(spec[:, band_idx:band_idx +
                                         self.band_width[i]])   # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i, bn_func in enumerate(self.BN):
            subband_feature.append(
                bn_func(subband_spec[i].view(batch_size * nch,
                                             self.band_width[i] * 2, -1)))   # nband 개 x (B, N, T)
        subband_feature = torch.stack(subband_feature, 1)   # B, nband, N, T
        # print(subband_feature.size(), spk_emb_input.size())

        predict_speaker_lable = torch.tensor(0.0).to(
            spk_emb_input.device)   # dummy
        spk_frame_feat = None   # <<<<< 더한 것 - 화자 인코더의 프레임 시퀀스 (#81)
        if self.joint_training:
            if not self.spk_feat:
                if self.feat_type == "consistent":
                    with torch.no_grad():
                        spk_emb_input = self.preEmphasis(spk_emb_input)          # (B, t_spk)
                        spk_emb_input = self.spk_encoder(spk_emb_input) + 1e-8   # (B, 80, T_spk)
                        spk_emb_input = spk_emb_input.log()                      # (B, 80, T_spk)
                        spk_emb_input = spk_emb_input - torch.mean(
                            spk_emb_input, dim=-1, keepdim=True)   # (B, 80, T_spk)
                        spk_emb_input = spk_emb_input.permute(0, 2, 1)   # (B, T_spk, 80)

            tmp_spk_emb_input = self.spk_model(spk_emb_input)   # 튜플 - 프레임 (B, 512, T_spk) · 임베딩 (B, emb)
            if isinstance(tmp_spk_emb_input, tuple):
                # <<<<< 더한 것 - [0] 은 프레임 (B, 512, T_spk), [-1] 은 풀링 (B, emb).
                #       원본은 [-1] 만 썼고 [0] 은 버렸다 (#81).
                spk_frame_feat = tmp_spk_emb_input[0]   # (B, 512, T_spk)
                spk_emb_input = tmp_spk_emb_input[-1]   # (B, emb)
            else:
                spk_emb_input = tmp_spk_emb_input   # (B, emb)
            predict_speaker_lable = self.pred_linear(spk_emb_input)   # (B, spksInTrain). multi_task=False 면 Identity 라 (B, emb)

        # <<<<< 더한 것 - SD-FiLM 은 풀링 벡터가 아니라 프레임 시퀀스를 조건으로 받는다 (#81·#83).
        #       벡터 하나면 L_s=1 이라 softmax 가 원소 1개 위에서 돌아 항상 1 이 되고,
        #       SD-FiLM 이 FiLM 으로 퇴화한다. (B, 512, T_spk) 를 그대로 넘기고
        #       (B, T_spk, 512) 로의 전치는 SpeakerFuseLayer 가 einops 로 한다.
        #       pred_linear(multi_task) 는 위에서 풀링 벡터를 그대로 쓰므로 영향이 없다.
        if self.spk_fuse_type == "SDFiLM":
            if spk_frame_feat is None:
                raise ValueError(
                    "spk_fuse_type='SDFiLM' 은 프레임 시퀀스가 필요하다 - "
                    "joint_training=True 이고 튜플을 돌려주는 화자 인코더여야 한다 (#81).")
            spk_embedding = spk_frame_feat                        # (B, 512, T_spk)
        else:
            spk_embedding = self.spk_transform(spk_emb_input)         # (B, emb)
            spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)   # (B, 1, emb, 1)

        sep_output = self.separator(subband_feature, spk_embedding,
                                    torch.tensor(nch))   # (B, nband, N, T)

        sep_subband_spec = []
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.band_width[i], -1)   # (B, 2, 2, BW, T)
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1])   # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]   # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]   # B*nch, K, BW, T
            est_spec_real = (subband_mix_spec[i].real * this_mask_real -
                             subband_mix_spec[i].imag * this_mask_imag
                             )   # B*nch, BW, T
            est_spec_imag = (subband_mix_spec[i].real * this_mask_imag +
                             subband_mix_spec[i].imag * this_mask_real
                             )   # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real,
                                                  est_spec_imag))   # nband 개 x (B, BW, T) 복소수
        est_spec = torch.cat(sep_subband_spec, 1)   # B*nch, F, T
        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            length=nsample,
        )   # (B, t)

        output = output.view(batch_size, nch, -1)   # (B, nch, t)
        s = torch.squeeze(output, dim=1)            # (B, t)

        return s, predict_speaker_lable


if __name__ == "__main__":
    from thop import profile, clever_format

    model = BSRNN(
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        spk_fuse_type="additive",
    )

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))
    x = torch.randn(4, 32000)
    spk_embeddings = torch.randn(4, 256)
    output = model(x, spk_embeddings)
    print(output.shape)

    macs, params = profile(model, inputs=(x, spk_embeddings))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
