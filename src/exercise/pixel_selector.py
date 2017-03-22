from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

module = tf.load_op_library('pixel_selector.so')

@ops.RegisterGradient("PixelSelector")
def _pixel_selector_grad(op, grad):
    """The gradients for 'pixel_selector'.
        
    Args:
        op: The 'pixel_selector' operation we want to differentiate.
        grad: Gradient with respect to the output of the 'pixel_selector' op.
    
    Returns:
        Gradients with respect to the coordinates of points of interest for 'pixel_selector'.
    """
    input = op.inputs[0]
    coord = op.inputs[1]
    strides = op.inputs[2]
    num_points = array_ops.shape(coord)[0]
    num_coord = array_ops.shape(coord)[1]
    shape = array_ops.shape(coord)
    coord_grad = array_ops.zeros_like(shape)
    back_grad = array_ops.reshape(grad,[-1])

    for i in range(0,num_points):
        for j in range(0, num_coord):
            coord_tmp = coord
            coord_tmp[i,j] = coord_tmp[i,j] + 1.0
            tmp1 = array_ops.reshape(module.pixel_selector(input,coord_tmp,strides),[-1])
            coord_tmp = coord
            coord_tmp[i,j] = coord_tmp[i,j] - 1.0
            tmp2 = array_ops.reshape(module.pixel_selector(input,coord_tmp,strides),[-1])
            tmp = math_ops.subtract(tmp1,tmp2)
            tmp = math_ops.divide(tmp,2)
            tmp = math_ops.multiply(tmp,back_grad)
            coord_grad[i,j] = math_ops.reduce_sum(tmp)

    return [None,coord_grad,None]

