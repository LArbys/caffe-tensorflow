import math
from collections import namedtuple

from .errors import KaffeError

TensorShape = namedtuple('TensorShape', ['batch_size', 'channels', 'height', 'width'])


def get_filter_output_shape(i_h, i_w, params, round_func):
    o_h = (i_h + 2 * params.pad_h - params.kernel_h) / float(params.stride_h) + 1
    o_w = (i_w + 2 * params.pad_w - params.kernel_w) / float(params.stride_w) + 1
    return (int(round_func(o_h)), int(round_func(o_w)))

def get_deconv_filter_output_shape(i_h, i_w, params, round_func):
    """ stolen from tensorflow.python.layers.util
        this basically only returns the 'SAME' shape scheme
    """

    o_h = i_h*params.stride_h
    #if params.pad_h == 0:
    #    o_h += max(params.kernel_h - params.stride_h, 0)
    #else:
    #    o_h -= (params.kernel_h + params.stride_h - 2)

    o_w = i_w*params.stride_w
    #if params.pad_w == 0:
    #    o_w += max(params.kernel_w - params.stride_w, 0)
    #else:
    #    o_w -= (params.kernel_w + params.stride_w - 2)
    
    return (int(round_func(o_h)), int(round_func(o_w)))


def get_strided_kernel_output_shape(node, round_func, deconv=False):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    if not deconv:
        o_h, o_w = get_filter_output_shape(input_shape.height, input_shape.width,
                                           node.layer.kernel_parameters, round_func)
    else:
        o_h, o_w = get_deconv_filter_output_shape(input_shape.height, input_shape.width,
                                                  node.layer.kernel_parameters, round_func)        
    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return TensorShape(input_shape.batch_size, c, o_h, o_w)


def shape_not_implemented(node):
    raise NotImplementedError


def shape_identity(node):
    assert len(node.parents) > 0
    return node.parents[0].output_shape


def shape_scalar(node):
    return TensorShape(1, 1, 1, 1)


def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        val = node.output_shape
        if len(val) < 4:
           return list(val) + [1] * (4 - len(val))
        return val
    try:
        # New-style input specification
        return map(int, node.parameters.shape[0].dim)
    except:
        # We most likely have a data layer on our hands. The problem is,
        # Caffe infers the dimensions of the data from the source (eg: LMDB).
        # We want to avoid reading datasets here. Fail for now.
        # This can be temporarily fixed by transforming the data layer to
        # Caffe's "input" layer (as is usually used in the "deploy" version).
        # TODO: Find a better solution for this.
        raise KaffeError('Cannot determine dimensions of data layer.\n'
                         'See comments in function shape_data for more info.')

def shape_reshape(node):
    dims = node.parameters.shape.dim
    return TensorShape(dims[0], dims[1], dims[2], dims[3])

def shape_mem_data(node):
    params = node.parameters
    return TensorShape(params.batch_size, params.channels, params.height, params.width)


def shape_concat(node):
    axis = node.layer.parameters.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)


def shape_convolution(node):
    return get_strided_kernel_output_shape(node, math.floor)

def shape_deconvolution(node):
    return get_strided_kernel_output_shape(node, math.floor, deconv=True)


def shape_pool(node):
    return get_strided_kernel_output_shape(node, math.ceil)


def shape_inner_product(node):
    input_shape = node.get_only_parent().output_shape
    return TensorShape(input_shape.batch_size, node.layer.parameters.num_output, 1, 1)
