"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from tools.attr_dict import AttrDict
import inspect
import numpy as np
import socket
import torch
from torch import nn

from visdom import Visdom

import matplotlib
from PIL import Image
import imageio

matplotlib.use('agg')
hostname = socket.gethostname()

_errstr = "Mode is unknown or incompatible with input array shape."
viz = None
from torch.autograd import grad
from torch.autograd import Variable


def get_visdom_env(cfg):
    if len(cfg.visdom_env) == 0:
        visdom_env = cfg.exp_dir
    else:
        visdom_env = cfg.visdom_env
    return visdom_env


def get_visdom_connection(server='http://localhost', port=8097):
    global viz
    if viz is None:
        viz = Visdom(server=server, port=port)
    return viz


def choice_exclude(n_obj, exclude=[]):
    return np.random.choice([i for i in range(n_obj) if i not in exclude])


# Adapted from:
# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
# and https://gist.github.com/jojonki/be1e8af97dfa12c983446391c3640b68
def weight_init(m):
    if isinstance(m, nn.Conv2d) or \
       isinstance(m, nn.ConvTranspose2d):
        truncated_normal_(m.weight, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.affine:
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)


# Truncated normal init
def truncated_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def has_method(ob, m):
    obcls = ob.__class__
    return hasattr(obcls, m) and callable(getattr(obcls, m))


def get_net_input(batch):
    # move to gpu and cast to Var
    net_input = {}
    for k in batch:
        if has_method(batch[k], 'cuda') and torch.cuda.is_available():
            net_input[k] = batch[k].cuda()
        else:
            net_input[k] = batch[k]

    return net_input


def auto_init_args(obj, tgt=None, can_overwrite=False):
    # autoassign constructor arguments
    frame = inspect.currentframe().f_back  # the frame above
    params = frame.f_locals
    nparams = frame.f_code.co_argcount
    paramnames = frame.f_code.co_varnames[1:nparams]
    if tgt is not None:
        if not can_overwrite:
            assert not hasattr(obj, tgt)
        setattr(obj, tgt, AttrDict())
        tgt_attr = getattr(obj, tgt)
    else:
        tgt_attr = obj

    for name in paramnames:
        # print('autosetting %s -> %s' % (name,str(params[name])) )
        setattr(tgt_attr, name, params[name])

# Adapated from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
def gradient_penalty(x, f):
    # interpolation
    if torch.cuda.is_available():
        # GPU version
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape).cuda()
        beta = torch.rand(x.size()).cuda()

        y = x + 0.5 * x.std() * beta
        z = x + alpha * (y - x)

        # gradient penalty
        z = Variable(z, requires_grad=True).cuda()
        o = f(z)[1]
        g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    else:
        # CPU version
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape)
        beta = torch.rand(x.size())

        y = x + 0.5 * x.std() * beta
        z = x + alpha * (y - x)

        # gradient penalty
        z = Variable(z, requires_grad=True)
        o = f(z)[1]
        g = grad(o, z, grad_outputs=torch.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp

# Save utils from SVG code
def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    return toimage(tensor.numpy(),
                   high=255*tensor.max().numpy(),
                   channel_axis=0)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available
    if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if
    Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        images = [inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))
