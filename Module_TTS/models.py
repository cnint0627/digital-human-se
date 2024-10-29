import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from style_models import StyleEncoder, AdainResBlk1d, LinearNorm, ResBlk1d


# 汉字发音时长预测类
class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class AttentionModule(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.key_layer = nn.Linear(hidden_channels, hidden_channels)
        self.query_layer = nn.Linear(hidden_channels, hidden_channels)
        self.value_layer = nn.Linear(hidden_channels, hidden_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # Compute attention scores
        scores = torch.matmul(self.query_layer(query), self.key_layer(key).transpose(-2, -1)) / math.sqrt(
            query.size(-1))
        # Apply softmax to get attention weights
        weights = self.softmax(scores)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(weights, self.value_layer(value))
        return attention_output, weights


class TextEncoder(nn.Module):
    def __init__(
            self,
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.emb_bert = nn.Linear(256, hidden_channels)
        self.emb_style = nn.Linear(128, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.attention_module = AttentionModule(hidden_channels)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, bert, style=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if bert is not None:
            b = self.emb_bert(bert)
            x = x + b
        if style is not None:
            s = self.emb_style(style)  # [b, h]
            # Use attention mechanism to combine style with x
            s_expanded = s.unsqueeze(1).expand(-1, x.size(1), -1)  # [b, t, h]
            attention_output, weights = self.attention_module(x, s_expanded, s_expanded)
            x = x + attention_output  # Update x with the attention output

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()




class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        s_channels=128,  # 假设s的通道数为128，用于条件层
    ):
        super(Generator, self).__init__()
        self.upsample_initial_channel = upsample_initial_channel
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # # 添加AdaIN条件层
        # self.adain_blocks = nn.ModuleList([
        #     AdainResBlk1d(upsample_initial_channel // 2 // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
        #                   s_channels)
        #     for i in range(self.num_upsamples)
        # ])


    def forward(self, x, g=None, s=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)


        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)

            # if s is not None:
            # # 应用 AdaIN
            #     x = self.adain_blocks[i](x, s)


            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels


        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        style=None,
        use_sdp=False,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        # self.sp = StylePredictor()

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, bert, y, y_lengths, sid=None, style=None):
        """
            x: text_padded\n
            x_lengths: text_lengths\n
            bert: bert_padded\n
            spec: spec_padded\n
            y: wav_padded\n
            y_lengths: wav_lengths
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert, style)
        # if self.n_speakers > 0:
        #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        # else:
        #     g = None
        g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        # duration
        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging
        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # o = self.dec(z_slice, g=g)
        o = self.dec(z_slice, g=g, s=None)

        # infer
        z_r = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z_r = self.flow(z_r, y_mask, g=g, reverse=True)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, z_r, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, bert, sid=None, noise_scale=1, length_scale=1, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)


class SynthesizerEval(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=False,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def remove_weight_norm(self):
        print("Removing weight norm...")
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, x, x_lengths, bert=None, sid=None, noise_scale=1, length_scale=1, max_len=None, style=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert, style)

        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_pause(self, x, x_lengths, bert, pause_mask, pause_value, sid=None, noise_scale=1, length_scale=1, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        w_ceil = w_ceil * pause_mask + pause_value
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_stream(self, x, x_lengths, bert, sid=None, noise_scale=1, length_scale=1):
        print("-----------------------------------------------------------------------------")
        import datetime
        import numpy
        print(datetime.datetime.now())
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        len_z = z_p.size()[2]
        print('frame size is: ', len_z)
        if (len_z < 100):
            print('no nead steam')
            z = self.flow(z_p, y_mask, g=g, reverse=True)
            one_time_wav = self.dec(z, g=g)[0, 0].data.cpu().float().numpy()
            return one_time_wav

        # can not change these parameters
        hop_length = 256 # bert_vits.json
        hop_frame = 9
        hop_sample = hop_frame * hop_length
        stream_chunk = 100
        stream_index = 0
        stream_out_wav = []

        while (stream_index + stream_chunk < len_z):
            if (stream_index == 0): # start frame
                cut_s = stream_index
                cut_s_wav = 0
            else:
                cut_s = stream_index - hop_frame
                cut_s_wav = hop_sample

            if (stream_index + stream_chunk > len_z - hop_frame): # end frame
                cut_e = stream_index + stream_chunk
                cut_e_wav = 0
            else:
                cut_e = stream_index + stream_chunk + hop_frame
                cut_e_wav = -1 * hop_sample
            
            z_chunk = z_p[:, :, cut_s:cut_e]
            m_chunk = y_mask[:, :, cut_s:cut_e]
            z_chunk = self.flow(z_chunk, m_chunk, g=g, reverse=True)
            o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
            o_chunk = o_chunk[cut_s_wav:cut_e_wav]
            stream_out_wav.extend(o_chunk)
            stream_index = stream_index + stream_chunk
            print(datetime.datetime.now())

        if (stream_index < len_z):
            cut_s = stream_index - hop_frame
            cut_s_wav = hop_sample
            z_chunk = z_p[:, :, cut_s:]
            m_chunk = y_mask[:, :, cut_s:]
            z_chunk = self.flow(z_chunk, m_chunk, g=g, reverse=True)
            o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
            o_chunk = o_chunk[cut_s_wav:]
            stream_out_wav.extend(o_chunk)

        stream_out_wav = numpy.asarray(stream_out_wav)
        return stream_out_wav



# 假设的 AdainResBlk1d 和 LinearNorm 定义（此处未给出，需要您自己实现）
# class AdainResBlk1d(nn.Module):
#     # 实现细节...
# class LinearNorm(nn.Module):
#     # 实现细节...（可能是对 nn.Linear 的简单封装，带有权重归一化）

class StylePredictor(nn.Module):
    def __init__(self, d_hid=513, style_dim=128, dropout=0.1):
        super(StylePredictor, self).__init__()
        self.d_hid = d_hid
        self.style_dim = style_dim

        # 一个简单的层来将s扩展到与x相同的时间长度
        self.style_linear = nn.Linear(style_dim, d_hid)  # 将s的维度从style_dim转换为d_hid

        # LSTM层，注意d_hid被设置为512，这样d_hid // 2 = 256，与x的通道数的一半相同
        self.shared = nn.LSTM(d_hid * 2, d_hid // 2, 1, batch_first=True, bidirectional=True)

        # 输出层，将LSTM的输出（已经是256通道）映射回原始x的通道数（在这个例子中实际上是多余的，因为已经是256了）
        # 但为了保持灵活性，我们可以保留它，或者将其替换为一个身份映射
        self.output_proj = nn.Linear(d_hid // 2, 513)  # 或者使用 nn.Identity() 如果不需要额外的映射

    def forward(self, x, s, F0_true=None, N_true=None):
        # 处理风格向量s
        s_expanded = self.style_linear(s).unsqueeze(1).expand(-1, x.size(1), -1)  # [batch_size, T, d_hid]

        # 连接x和s_expanded
        x_combined = torch.cat([x, s_expanded], dim=-1)  # [batch_size, T, 2*d_hid]

        # 通过LSTM
        x_lstm, _ = self.shared(x_combined)

        # 由于LSTM是双向的，我们需要合并输出
        x_bidir = x_lstm.view(x_lstm.size(0), x_lstm.size(1), 2, -1).sum(2)  # [batch_size, T, d_hid // 2]

        # 使用线性层（如果需要的话）将LSTM的输出映射回原始x的通道数
        # 但在这个例子中，由于d_hid被设置为512，所以LSTM的输出已经是256通道了
        x_output = self.output_proj(x_bidir)  # [batch_size, T, 256]

        # 如果self.output_proj实际上是一个身份映射（nn.Identity()），则上面的行是多余的

        return x_output

        # 注意：在训练循环中，您需要调用 forward 方法（或直接在 F0Ntrain 中处理损失计算），
# 并传入真实的 F0 和 N 值来计算损失。
