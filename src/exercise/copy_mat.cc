#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("CopyMat")
    .Input("in: int16")
    .Output("out: int16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
                    c->set_output(0, c->input(0));
                    return Status::OK();
    }); // Set the shape property for the output tensor

class CopyMatOp : public OpKernel {
    public:
    explicit CopyMatOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        int num_dims = input_tensor.shape().dims();
        if (num_dims != 3)
            return;
        int depth = input_tensor.shape().dim_size(0);
        int width = input_tensor.shape().dim_size(1);
        int height = input_tensor.shape().dim_size(2);
        auto input = input_tensor.shaped<int16,3>({depth,width,height}); // Conversion to Eigen::Tensor
        
        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        
        auto output = output_tensor->shaped<int16,3>({depth,width,height}); // Conversion to Eigen::Tensor

        // Copy all elements
        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < height; k++)
                {
                    output(i,j,k) = input(i,j,k);
                }
            }
        }
        
    }
};

REGISTER_KERNEL_BUILDER(Name("CopyMat").Device(DEVICE_CPU), CopyMatOp);
