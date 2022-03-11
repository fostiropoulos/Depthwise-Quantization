import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import denorm_batch
from loss_utils import coeff_sizes
from loss_utils import dmol_loss as mix_loss
from loss_utils import sample_from_dmol


class Quantize(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """

    @torch.cuda.amp.autocast(enabled=False)
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.requires_grad = True
        self.threshold = 1.0
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.is_init = False

    def _tile(self, x):
        import numpy as np

        d, ew = x.shape
        if d < self.n_embed:
            n_repeats = (self.n_embed + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embed(self, x):
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_embed].transpose(0, 1)
        embed = _k_rand
        self.embed.data.copy_(embed)

    @torch.no_grad()
    def _update_embed(self, x, closest_embed_oh):
        """
        embed_ind are the indexes of the closest embedding to x
        """
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_embed].transpose(0, 1)
        embed_onehot_sum = closest_embed_oh.sum(0)
        embed_sum = x.transpose(0, 1) @ closest_embed_oh
        self.cluster_size.data.mul_(self.decay).add_(
            embed_onehot_sum, alpha=1 - self.decay
        )
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        )
        cluster_size = cluster_size.unsqueeze(0)

        usage = (cluster_size >= self.threshold).float()
        embed = usage * self.embed_avg / cluster_size + (1 - usage) * _k_rand

        self.embed.data.copy_(embed)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        flatten = input.float().reshape(-1, self.dim)

        # if not self.is_init:
        #    self._init_embed(flatten)
        #    self.is_init = True

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)

        closest_embed_oh = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training and self.requires_grad:
            self._update_embed(flatten, closest_embed_oh)

        diff = torch.norm(quantize.detach() - input).pow(2) / np.prod(input.shape)
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class VQVAE(nn.Module):
    MODEL_ARGUMENTS = [
        "loss_name",
        "vq_type",
        "n_logistic_mix",
        "n_hier",
        "beta",
        "in_channel",
        "out_channel",
        "channel",
        "n_res_block",
        "n_res_channel",
        "n_coder_blocks",
        "embed_dim",
        "n_codebooks",
        "stride",
        "decay",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            if k not in self.MODEL_ARGUMENTS:
                raise Exception("Unrecognized Argument: %s %s" % (k, v))
            setattr(self, k, v)
        in_channel = self.in_channel
        channel = self.channel
        n_res_block = self.n_res_block
        n_res_channel = self.n_res_channel
        stride = self.stride
        embed_dim = self.embed_dim
        n_coder_blocks = self.n_coder_blocks
        n_codebooks = self.n_codebooks

        self.enc_blocks = nn.ModuleList()
        self.quantize_convs = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        # stride is originally 4
        enc_blocks = [
            Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        ]
        enc_blocks += [
            Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
            for i in range(n_coder_blocks - 1)
        ]
        self.enc_blocks.append(nn.Sequential(*enc_blocks))
        bot_codebook_size = self.n_hier[0]
        self.quantizers.append(
            torch.nn.ModuleList(
                [Quantize(embed_dim, bot_codebook_size) for _ in range(n_codebooks)]
            )
        )

        # channel*2 because we concatenate encodings and decodings

        self.quantize_convs.append(
            nn.Conv2d(channel * 2 if len(self.n_hier) > 1 else channel, embed_dim, 1)
        )

        dec_in_channels = embed_dim * self.n_codebooks
        up_convs = [
            Decoder(
                dec_in_channels,
                channel,
                channel,
                n_res_block,
                n_res_channel,
                stride=2 if n_coder_blocks > 1 else 1,
            )
        ]
        up_convs += [
            Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=2)
            for i in range(n_coder_blocks - 2)
        ]
        self.upsample.append(nn.Sequential(*up_convs))

        # bottom / last decoder is a no-op
        identity = nn.Identity()
        self.dec_blocks.append(identity)

        # loop bot to top. The -1 is because bot is defined above
        for i, codebook_size in enumerate(self.n_hier[1:]):
            enc_blocks = [
                Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
            ]
            self.enc_blocks.append(nn.Sequential(*enc_blocks))

            cur_hier = len(self.n_hier[1:]) - 1 - i
            # only for top we have channel as input to quantizer because we don't condition on prior codes
            conv2D_channels = channel if cur_hier == 0 else channel * 2

            self.quantize_convs.append(nn.Conv2d(conv2D_channels, embed_dim, 1))

            # for DQ we over-write this module
            self.quantizers.append(
                torch.nn.ModuleList(
                    [Quantize(embed_dim, codebook_size) for _ in range(n_codebooks)]
                )
            )

            # final upsample layer that is passed to decoder
            up_convs = [nn.Upsample(scale_factor=2 ** (i), mode="nearest")]
            up_convs += [
                Decoder(
                    dec_in_channels,
                    channel,
                    channel,
                    n_res_block,
                    n_res_channel,
                    stride=2,
                )
            ]
            up_convs += [
                Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=2)
                for i in range(n_coder_blocks - 1)
            ]
            self.upsample.append(nn.Sequential(*up_convs))

            # Used only for upsampling the conditoned codes
            dec_blocks = [
                Decoder(
                    dec_in_channels,
                    channel,
                    channel,
                    n_res_block,
                    n_res_channel,
                    stride=2,
                )
            ]
            self.dec_blocks.append(nn.Sequential(*dec_blocks))

        if self.loss_name == "ce":
            self.out_channel = self.in_channel * 256

            def loss_fn(x, y):
                y = denorm_batch(y)
                B, _, W, H = x.shape
                x = x.view(B, 256, -1, W, H)
                return nn.CrossEntropyLoss()(x, y)

            self.loss_fn = loss_fn
        elif self.loss_name == "mix":
            assert self.n_logistic_mix == 10
            self.out_channel = (
                2 * self.in_channel + 1 + coeff_sizes(self.in_channel)
            ) * self.n_logistic_mix

            def loss_fn(x, y):
                y = denorm_batch(y).true_divide(255).mul(2).add(-1)
                return torch.mean(mix_loss(x, y, nr_mix=self.n_logistic_mix))

            self.loss_fn = loss_fn
        else:
            self.out_channel = self.in_channel
            self.loss_fn = nn.MSELoss()

        # final decoder decoding from all hierarchies
        dec_blocks = [
            Decoder(
                channel * len(self.n_hier),
                channel,
                channel,
                n_res_block,
                n_res_channel,
                stride=stride,
            )
        ]
        dec_blocks += [
            Decoder(channel, channel, channel, n_res_block, n_res_channel, stride=1)
        ]
        dec_blocks += [
            nn.Conv2d(channel, self.out_channel, kernel_size=3, padding=1, stride=1)
        ]
        self.decoder = nn.Sequential(*dec_blocks)

    def getEncode(self, input):
        encodings = []

        enc = input

        # bot to top encoder blocks and encodings
        for block in self.enc_blocks:
            enc = block(enc)
            encodings.append(enc)

        # reverse them top to bot for the decoding process
        quantize_convs = list(reversed(self.quantize_convs))
        quantizers = list(reversed(self.quantizers))
        encodings = list(reversed(encodings))
        dec_blocks = list(reversed(self.dec_blocks))
        quants = []
        pre_quants = []  # used for analysis of mutual information
        ids = []
        # Quantizer Loss
        diffs = 0.0

        for i, enc in enumerate(encodings):
            if i == 0:
                # top doesn't have previous decodings to condition on
                pass
            else:
                enc = torch.cat([dec, enc], 1)
            quant, diff, idx, pre_quant = self.quantize(
                quantize_convs[i], quantizers[i], enc
            )
            quants.append(quant)
            pre_quants.append(pre_quant)
            ids.append(idx)
            diffs += diff

            dec = dec_blocks[i](quant)
        return quants, ids

    def getDecode(self, input):
        upsample_blocks = list(reversed(self.upsample))
        quantizers = list(reversed(self.quantizers))
        upsamples = []
        for i in range(len(upsample_blocks)):
            quant = self.embed(quantizers[i], input[i])
            upsampled = upsample_blocks[i](quant)
            upsamples.append(upsampled)

        dec = self.decoder(torch.cat(upsamples, 1))
        return dec

    def forward(self, input):

        encodings = []

        enc = input

        # bot to top encoder blocks and encodings
        for block in self.enc_blocks:
            enc = block(enc)
            encodings.append(enc)

        # reverse them top to bot for the decoding process
        quantize_convs = list(reversed(self.quantize_convs))
        quantizers = list(reversed(self.quantizers))
        encodings = list(reversed(encodings))
        dec_blocks = list(reversed(self.dec_blocks))
        upsample_blocks = list(reversed(self.upsample))

        quants = []
        pre_quants = []  # used for analysis of mutual information
        ids = []
        upsamples = []
        # Quantizer Loss
        diffs = 0.0

        for i, enc in enumerate(encodings):
            if i == 0:
                # top doesn't have previous decodings to condition on
                pass
            else:
                enc = torch.cat([dec, enc], 1)
            quant, diff, idx, pre_quant = self.quantize(
                quantize_convs[i], quantizers[i], enc
            )
            quants.append(quant)
            pre_quants.append(pre_quant)
            ids.append(idx)
            diffs += diff

            dec = dec_blocks[i](quant)
            upsampled = upsample_blocks[i](quant)

            upsamples.append(upsampled)

        dec = self.decoder(torch.cat(upsamples, 1))
        recon_loss = self.loss_fn(dec, input)
        latent_loss = diffs
        loss = (recon_loss + self.beta * latent_loss).mean()
        return dec, ids, (loss, recon_loss, latent_loss)

    def embed(self, quant_block, input):

        quants = []
        for i in range(self.n_codebooks):
            quant_i = quant_block[i].embed_code(input[:, i, :, :])
            quant_i = quant_i.permute(0, 3, 1, 2)
            quants.append(quant_i)
        return torch.cat(quants, 1)

    @torch.no_grad()
    def decode_code(self, codes, c_range=None, verbose=True):
        """ Given the discrete codes top -> bot, decode them into their image representation"""
        upsamples = []
        if c_range:
            start, end = c_range
        else:
            start, end = 0, len(codes)
        quantizers = self.quantizers
        upsample_blocks = self.upsample
        codes = codes[start:end]
        if verbose:
            print("Decoding Using: ", [c.shape for c in codes])
        codes = list(reversed(codes))
        for c, up, quantizer in zip(codes, upsample_blocks, quantizers):
            quants = self.embed(quantizer, c)
            upsamples.append(up(quants))

        for i in range(start):
            upsamples.append(torch.zeros_like(upsamples[0]))
        upsamples = upsamples[::-1]
        for i in range(len(self.n_hier) - end):
            upsamples.append(torch.zeros_like(upsamples[0]))

        _upsamples = torch.cat(upsamples, 1)
        decoded = self.decoder(_upsamples)
        if self.loss_name == "ce":
            B, _, W, H = decoded.shape
            decoded = decoded.view(B, 256, -1, W, H)
            decoded = decoded.argmax(1)
            decoded = torch.true_divide(decoded, 255)
        elif self.loss_name == "mse":
            decoded = denorm_batch(decoded)
            decoded = torch.true_divide(decoded, 255)
        elif self.loss_name == "mix":
            decoded = sample_from_dmol(decoded)
            decoded = (decoded + 1) / 2
        return decoded

    def quantize(self, conv_block, quant_block, input):

        quants = []
        diff = 0.0
        ids = []
        pre_quant = conv_block(input).permute(0, 2, 3, 1)
        for i in range(self.n_codebooks):
            quant_i, diff_i, idx = quant_block[i](pre_quant)
            quant_i = quant_i.permute(0, 3, 1, 2)
            diff_i = diff_i.unsqueeze(0)
            diff += diff_i
            quants.append(quant_i)
            ids.append(idx)

        ids = torch.stack(ids, 1)
        quants = torch.cat(quants, 1)

        return quants, diff, ids, pre_quant.permute(0, 3, 1, 2)


class DQAE(VQVAE):
    def __init__(self, *args, **kwargs):
        super(DQAE, self).__init__(*args, **kwargs)
        channel = self.channel
        embed_dim = self.embed_dim
        n_codebooks = self.n_codebooks
        del self.quantize_convs

        self.quantize_convs = torch.nn.ModuleList()

        # bot to top, excluding top because top doesn't accept a concat input
        for i, _ in enumerate(self.n_hier):

            cur_hier = len(self.n_hier) - 1 - i
            # only for top we have channel as input to quantizer because we don't condition on prior codes
            conv2D_channels = channel if cur_hier == 0 else channel * 2
            quantize_conv = torch.nn.ModuleList(
                [nn.Conv2d(conv2D_channels, embed_dim, 1) for _ in range(n_codebooks)]
            )

            self.quantize_convs.append(quantize_conv)

    def quantize(self, conv_block, quant_block, input):

        quants = []
        diff = 0.0
        ids = []
        pre_quants = []
        for i in range(self.n_codebooks):
            pre_quant = conv_block[i](input).permute(0, 2, 3, 1)
            quant_i, diff_i, idx = quant_block[i](pre_quant)
            quant_i = quant_i.permute(0, 3, 1, 2)
            diff_i = diff_i.unsqueeze(0)
            diff += diff_i
            quants.append(quant_i)
            ids.append(idx)
            pre_quants.append(pre_quant.permute(0, 3, 1, 2))

        ids = torch.stack(ids, 1)
        quants = torch.cat(quants, 1)

        return quants, diff, ids, torch.cat(pre_quants, 1)

