import theano
import theano.tensor as T

from .tensor_util import softmax_nd


def sample_gumbel(shape, srng, eps=1e-9):
    rnd = srng.uniform(size=shape, low=0., high=1., dtype='float32')
    return -T.log(eps - T.log(eps + rnd))


def gumbel_softmax_sample(logits, temperature, srng):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, srng=srng)
    return softmax_nd(y / temperature)


def gumbel_argmax(logits, srng, axis=-1):
    g = sample_gumbel(shape=logits.shape, srng=srng)
    return T.argmax(logits + g, axis=axis)


def gumbel_softmax(logits, temperature, srng, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, srng=srng)
    if hard:
        y_hard = T.cast(T.eq(y, T.max(y, axis=-1, keepdims=True)), y.dtype)
        y = theano.gradient.zero_grad(y_hard - y) + y
    return y
