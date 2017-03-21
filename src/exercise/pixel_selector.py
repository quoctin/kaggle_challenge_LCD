from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("PixelSelector")
def _pixel_selector_grad(op, grad):
    """The gradients for 'pixel_selector'.
        
    Args:
        TODO
    
    Returns:
        Gradients with respect to the coordinates of points of interest for 'pixel_selector'
    """
    # TODO 
