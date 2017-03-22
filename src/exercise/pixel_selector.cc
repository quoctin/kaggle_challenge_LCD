#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("PixelSelector")
    .Input("in: float32")
    .Input("coord: float32")
    .Input("stride: int16")
    .Output("out: float32")
    /**.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
        ::tensorflow::shape_inference::ShapeHandle out;
        ::tensorflow::shape_inference::ShapeHandle output;
        ::tensorflow::shape_inference::DimensionHandle dim = c->Dim(c->input(1),0);
        TF_RETURN_IF_ERROR(c->Concatenate(c->input(0),c->UnknownShapeOfRank(1),&out));
        TF_RETURN_IF_ERROR(c->ReplaceDim(out,4,dim,&output));
        c->set_output(0,output);
        return Status::OK();
    }) // Set the shape property for the output tensor**/
    .Doc(R"doc(
         Replicating the 4D input tensor in a 5D tensor.
         
         Input 1 has the following format
            [batch_size, depth, width, height]
         
         Input 2 contains the coordinate of points and
         has size
            [num_points, 3]
         
         Input 3 contains the strides, namely
            [1, stride_depth, stride_width, stride_height]
         
         Output has the following format
            [batch_size, depth, width, height, pixels]
    )doc");


class PixelSelectorOp : public OpKernel {
    public:
    explicit PixelSelectorOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& input_tensor1 = context->input(1);
        const Tensor& input_tensor2 = context->input(2);

        int batch = input_tensor.shape().dim_size(0);
        int depth = input_tensor.shape().dim_size(1);
        int width = input_tensor.shape().dim_size(2);
        int height = input_tensor.shape().dim_size(3);
        auto input = input_tensor.shaped<float,4>({batch,depth,width,height}); // Conversion to Eigen::Tensor
        int pixels = input_tensor1.shape().dim_size(0);
        int num_coord = input_tensor1.shape().dim_size(1);
        auto input1 = input_tensor1.shaped<float,2>({pixels,num_coord}); // Conversion to Eigen::Tensor
        auto input2 = input_tensor2.flat<int16>(); // Conversion to Eigen::Tensor
        int stride_depth = input2(1);
        int stride_width = input2(2);
        int stride_height = input2(3);
        
        depth = (int) depth/stride_depth;
        width = (int) width/stride_width;
        height = (int) height/stride_height;
        
        // Create an output tensor
        Tensor* output_tensor = NULL;
        std::initializer_list<int64> dim_sizes = {int64(batch),int64(depth),int64(width),int64(height),int64(pixels)}; // check in file tensor_shape.h
        OP_REQUIRES_OK(context, context->allocate_output(0, ::tensorflow::TensorShape(dim_sizes),
                                                         &output_tensor));

        auto output = output_tensor->shaped<float,5>({batch,depth,width,height,pixels}); // Conversion to Eigen::Tensor
        
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
                            int tmp_j = j*stride_depth + ((int) input1(m,0));
                            int tmp_k = k*stride_width + ((int) input1(m,1));
                            int tmp_l = l*stride_height + ((int) input1(m,2));
                            //std::cout << "Before: " << tmp_j << " " << tmp_k << " " << tmp_l << std::endl;
                            if (tmp_j < 0)
                                tmp_j = 0;
                            if (tmp_j >= depth*stride_depth)
                                tmp_j = depth*stride_depth-1;
                            if (tmp_k < 0)
                                tmp_k = 0;
                            if (tmp_k >= width*stride_width)
                                tmp_k = width*stride_width-1;
                            if (tmp_l < 0)
                                tmp_l = 0;
                            if (tmp_l >= height*stride_height)
                                tmp_l = height*stride_height-1;
                            //std::cout << "After: " << tmp_j << " " << tmp_k << " " << tmp_l << std::endl;
                            output(i,j,k,l,m) = input(i,tmp_j,tmp_k,tmp_l);
                        }
                    }
                }
            }
        }
        
    }
};

REGISTER_KERNEL_BUILDER(Name("PixelSelector").Device(DEVICE_CPU), PixelSelectorOp);
