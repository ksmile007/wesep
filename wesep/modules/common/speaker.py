from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common import FiLM


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

    def __init__(self, embed_dim=256, feat_dim=512, fuse_type="concat"):
        super(SpeakerFuseLayer, self).__init__()
        assert fuse_type in ["concat", "additive", "multiply", "FiLM", "None"]

        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        elif fuse_type == "additive":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "multiply":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "FiLM":
            self.fc = FiLM(feat_dim, embed_dim)
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
