# Exercises to create new operation in TensorFlow

## Ex 1
Take in input a 3D tensor and copy it in the output
```python
import tensor flow as tf
copy_module = tf.load_op_library('/Path/to/copy_mat.so')
...
output = copy_module.copy_mat(input)
```

Inputs & outputs:
**input** and **output** are 3D tensors


## Ex 2
Take in input a 3D tensor and replicate it to create a 4D tensor in output
```python
import tensor flow as tf
replicate_module = tf.load_op_library('/Path/to/replicate_mat.so')
...
output = replicate_module.replicate_mat(input1,input2)
```

Inputs:
**input1** is a 3D tensor, **input2** is a constant defining the number of replicas (namely the fourth dimension of the output)


## Ex 3
Same as Ex 2 but accept multiple inputs. Therefore the input is 4D tensor
and the output is 5D
```python
import tensor flow as tf
replicate_module = tf.load_op_library('/Path/to/replicate_data.so')
...
output = replicate_module.replicate_data(input1,input2)
```

Inputs:
**input1** is a 4D tensor, **input2** is a constant defining the number of replicas (namely the fifth dimension of the output)


## Ex 4
Pixel selector
```python
import tensor flow as tf
replicate_module = tf.load_op_library('/Path/to/pixel_selector.so')
...
output = replicate_module.pixel_selector(input, coord, strides)
```

Inputs:
**input** is a 4D tensor of size `[batch,depth,width,height]`
**coord** is a 2D tensor of size `[num_points,3]` and it contains the coordinates of points of interest
**strides** is a 1D vector, namely `[1,stride_depth,stride_width,stride_height]`

Outputs:
**output** is a 5D tensor of size `[batch,depth,width,height,num_points]`

## Ex 5
Release of Pixel Selector (example for training using this operation)

## Ex 6
Release of Pixel Selector for GPU (still debugging) + modifications for compatibility with Tensorflow v.1.0.1
