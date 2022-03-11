
"""
File Adapted from: https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/vae_helpers.py
"""
import torch
import torch.nn.functional as F
import numpy as np


def coeff_sizes(C):
    return ((C-1)*C)//2


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def dmol_loss(l, x, bit=8, nr_mix=10):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/vae_helpers.py#L52
    """ INPUT B,C,W,H -1 to 1 """
    x = torch.movedim(x, (0, 1, 2, 3), (0, 3, 1, 2))
    l = torch.movedim(l, (0, 1, 2, 3), (0, 3, 1, 2))
    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [s for s in x.shape]

    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs +
                      [-1])

    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.reshape(torch.tanh(
        l[:, :, :, :, 2*nr_mix:]), xs[:-1]+[-1, nr_mix])
    # here and below: getting the means and adjusting them based on preceding sub-pixels
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)

    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                       * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                       coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [
                      xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + 1/(2**bit-1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1/(2**bit-1))

    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value

    log_probs = torch.where(x < -0.999,
                            log_cdf_plus,
                            torch.where(x > 0.999,
                                        log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5,
                                                    torch.log(
                                                        const_max(cdf_delta, 1e-12)),
                                                    log_pdf_mid - np.log((2**bit-1)/2))))

    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_dmol(l, C=3, nr_mix=10):

    l = torch.movedim(l, (0, 1, 2, 3), (0, 3, 1, 2))
    ls = [s for s in l.shape]
    xs = ls[:-1] + [C]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [-1])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(
        1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max(
        (l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(
        l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    out = torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1,
                                                                     xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)

    return torch.movedim(out, (0, 1, 2, 3), (0, 2, 3, 1))
