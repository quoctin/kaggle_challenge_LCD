#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

using namespace tensorflow;

REGISTER_OP("ReplicateData")
    .Input("in: int16")
    .Input("dim: int32")
    .Output("out: int16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
        ::tensorflow::shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(c->input(0),c->UnknownShapeOfRank(1),&output));
        c->set_output(0,output);
        return Status::OK();
    }) // Set the shape property for the output tensor
    .Doc(R"doc(
         Replicating the 4D input tensor in a 5D tensor.
         Input has the following format
            [batch_size, depth, width, height]
         Output has the following format
            [batch_size, depth, width, height, pixels]
    )doc");


class ReplicateDataOp : public OpKernel {
    public:
    explicit ReplicateDataOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& input_tensor1 = context->input(1);

        int batch = input_tensor.shape().dim_size(0);
        int depth = input_tensor.shape().dim_size(1);
        int width = input_tensor.shape().dim_size(2);
        int height = input_tensor.shape().dim_size(3);
        auto input = input_tensor.shaped<int16,4>({batch,depth,width,height}); // Conversion to Eigen::Tensor
        auto input1 = input_tensor1.scalar<int32>(); // Conversion to Eigen::Tensor
        
        int pixels = input1(0);
        
        // Create an output tensor
        Tensor* output_tensor = NULL;
        std::initializer_list<int64> dim_sizes = {int64(batch),int64(depth),int64(width),int64(height),int64(pixels)}; // check in file tensor_shape.h
        OP_REQUIRES_OK(context, context->allocate_output(0, ::tensorflow::TensorShape(dim_sizes),
                                                         &output_tensor));

        auto output = output_tensor->shaped<int16,5>({batch,depth,width,height,pixels}); // Conversion to Eigen::Tensor
        
        // Copy all elements and create replicas
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < depth; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    for (int l = 0; l < height; l++)
                    {
                        for (int m = 0; m < pixels; m++)
                        {
                            output(i,j,k,l,m) = input(i,j,k,l);
                        }
                    }
                }
            }
        }
        
    }
};

REGISTER_KERNEL_BUILDER(Name("ReplicateData").Device(DEVICE_CPU), ReplicateDataOp);
