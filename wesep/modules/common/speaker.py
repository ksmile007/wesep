from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common import FiLM

# <<<<< 더한 것 - SD-FiLM 조건화 (#83). sdfilm 은 SD-FiLM 저장소의 src/models/cond_module 을
#       `pip install -e .` 한 환경에만 있으므로(#82), 없어도 이 파일은 import 되게 둔다.
#       fuse_type='sdfilm' 을 실제로 고를 때만 SpeakerFuseLayer.__init__ 이 막는다.
from einops import rearrange

try:
    from sdfilm import Condition, SeqDepFiLM
except ImportError:
    rearrange = None   # noqa: F811 - 미설치 환경용 대체값
    Condition = None
    SeqDepFiLM = None


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)                               # (B, 1, t)
        input = F.pad(input, (1, 0), "reflect")                  # (B, 1, t+1)
        return F.conv1d(input, self.flipped_filter).squeeze(1)   # (B, t)


class SpeakerTransform(nn.Module):

    def __init__(self, embed_dim=256, num_layers=3, hid_dim=128):
        """
        Transform the pretrained speaker embeddings, keep the dimension
        :param embed_dim:
        :param num_layers:
        :param hid_dim:
        :return:
        """
        super(SpeakerTransform, self).__init__()
        self.transforms = []
        self.transforms.append(nn.Conv1d(embed_dim, hid_dim, 1))
        for _ in range(num_layers - 2):
            self.transforms.append(nn.Conv1d(hid_dim, hid_dim, 1))
            self.transforms.append(nn.Tanh())
        self.transforms.append(nn.Conv1d(hid_dim, embed_dim, 1))
        self.transforms = nn.Sequential(*self.transforms)

    def forward(self, x):
        if len(x.size()) == 2:
            return self.transforms(x.unsqueeze(-1)).squeeze(-1)   # (B, emb)
        else:
            return self.transforms(x)   # (B, emb, T)


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, dummy: Optional[torch.Tensor] = None):
        return self.linear(x)   # (..., out_features)


class SpeakerFuseLayer(nn.Module):

    def __init__(self, embed_dim=256, feat_dim=512, fuse_type="concat",
                 spk_fuse_kwargs=None):   # <<<<< 더한 것 - SD-FiLM 하이퍼파라미터 통로 (#83)
        super(SpeakerFuseLayer, self).__init__()
        assert fuse_type in [
            "concat", "additive", "multiply", "FiLM", "sdfilm", "None"
        ]

        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        elif fuse_type == "additive":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "multiply":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "FiLM":
            self.fc = FiLM(feat_dim, embed_dim)
        elif fuse_type == "sdfilm":
            # <<<<< 더한 것 - SD-FiLM (#83)
            if SeqDepFiLM is None:
                raise ImportError(
                    "fuse_type='sdfilm' 은 sdfilm 패키지가 필요하다 - SD-FiLM 저장소의 "
                    "src/models/cond_module 에서 `pip install -e .` 로 설치한다 (#82).")
            spk_fuse_kwargs = dict(spk_fuse_kwargs or {})
            if "q_dim" in spk_fuse_kwargs:
                raise ValueError(
                    "q_dim 은 feat_dim 에서 유도하므로 spk_fuse_kwargs 에 적지 않는다 - "
                    "feat_dim={} 과 어긋나면 조용히 깨진다 (#83).".format(feat_dim))
            # q_dim 은 토큰의 채널 = feat_dim. 토큰 축이 밴드x시간이라 nband 도 T 도 아니다.
            self.fc = SeqDepFiLM(q_dim=feat_dim, **spk_fuse_kwargs)
        else:
            raise ValueError("Fuse type not defined.")

    def forward(self, x, embed):
        """

        :param x: batch x dimension x length
        :param embed: batch x dimension x 1
        :return:
        """
        # B: 배치, nband: 서브밴드, N: feat_dim, emb: embed_dim, T: 프레임
        # x.dim() == 4는 BSRNN 계열, x.dim() == 3은 DPCCN, TFGridNet 에서 넘어옴
        # cross_ 시, embed 가 이미 (B, nband, N=emb, T)
        if self.fuse_type == "concat":
            # For Conv
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))   # (B, emb, T)
                y = torch.cat([x, embed_t], 1)              # (B, N+emb, T)
                y = torch.transpose(y, 1, 2)                # (B, T, N+emb)
                x = torch.transpose(self.fc(y), 1, 2)       # (B, N, T)
            else:   # len(x.size()) == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))   # (B, nband, emb, T)
                y = torch.cat([x, embed_t], 2)                         # (B, nband, N+emb, T)
                y = torch.transpose(y, 2, 3)                           # (B, nband, T, N+emb)
                x = torch.transpose(self.fc(y), 2, 3).contiguous()     # (B, nband, N, T)
                # print(x.size())
        elif self.fuse_type == "additive":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))         # (B, emb, T)
                embed_t = torch.transpose(embed_t, 1, 2)          # (B, T, emb)
                x = x + torch.transpose(self.fc(embed_t), 1, 2)   # (B, N, T)
            else:   # len(x.size()) == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))   # (B, nband, emb, T)
                embed_t = torch.transpose(embed_t, 2, 3)               # (B, nband, T, emb)
                x = x + torch.transpose(self.fc(embed_t), 2, 3)        # (B, nband, N, T)
        elif self.fuse_type == "multiply":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))         # (B, emb, T)
                embed_t = torch.transpose(embed_t, 1, 2)          # (B, T, emb)
                x = x * torch.transpose(self.fc(embed_t), 1, 2)   # (B, N, T)
            else:   # len(x.size()) == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))   # (B, nband, emb, T)
                embed_t = torch.transpose(embed_t, 2, 3)               # (B, nband, T, emb)
                x = x * torch.transpose(self.fc(embed_t), 2, 3)        # (B, nband, N, T)
        elif self.fuse_type == "sdfilm":
            # <<<<< 더한 것 - SD-FiLM (#83). embed 만 모양이 다르다 - 풀링 벡터가 아니라
            #       화자 인코더의 프레임 시퀀스 (B, 512, T_spk) 가 그대로 들어온다.
            if len(x.size()) != 4:
                raise ValueError(
                    "fuse_type='sdfilm' 은 BSRNN 계열의 4차원 x 만 받는다 - "
                    "받은 차원 {} (#83).".format(len(x.size())))
            nband = x.size(1)
            # 토큰 축은 밴드x시간 - 패치 하나가 토큰 하나 (#83 정할 것 1 의 안 나).
            # 시간축(b t (nb n))으로 잡으면 조건화 파라미터가 26배가 된다.
            hidden_state = rearrange(x, "b nb n t -> b (nb t) n")   # (B, nband*T, N)
            cond_seq = rearrange(embed, "b c t -> b t c")           # (B, T_spk, 512)
            # attention_mask 는 안 넘긴다 - tse_collate_fn(mode='min') 이 배치 최솟값으로
            # 잘라내 패딩 0 이 없으므로 무시할 칸이 없다 (#83 실측,
            # notebooks/enrollment_length_and_silence.ipynb)
            cond = Condition(last_hidden_state=cond_seq)
            fused, _attn_weights = self.fc(hidden_state, cond)      # (B, nband*T, N)
            # .contiguous() 가 필요하다 - rearrange 가 돌려주는 것은 전치된 뷰이고,
            # FuseSeparation.forward 의 x.view(B, nband*N, T) 가 비연속 텐서를 거부한다
            # (실측 - RuntimeError: view size is not compatible ...).
            # 위 concat 분기도 같은 이유로 .contiguous() 를 붙여 두었다.
            x = rearrange(fused, "b (nb t) n -> b nb n t",
                          nb=nband).contiguous()   # (B, nband, N, T)
        else:
            embed = embed.squeeze(-1)   # (B, 1, emb)
            x = self.fc(embed, x)       # (B, nband, N, T) 또는 (B, N, T)
        return x


def test_speaker_fuse():
    st = SpeakerTransform(embed_dim=256, num_layers=3, hid_dim=128)
    sfl = SpeakerFuseLayer(fuse_type="multiply")

    embeds = torch.rand(4, 256)
    encoder_output = torch.rand(4, 512, 1000)

    print(embeds.size())
    embeds = st(embeds)
    print(embeds.size())
    output = sfl(encoder_output, embeds)
    print(output.size())


if __name__ == "__main__":
    test_speaker_fuse()
