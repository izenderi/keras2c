#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void resnetv1(k2c_tensor* input_1_input, k2c_tensor* dense_output,float* conv2d_output_array,float* conv2d_padded_input_array,float* conv2d_kernel_array,float* conv2d_bias_array,float* batch_normalization_output_array,float* batch_normalization_mean_array,float* batch_normalization_stdev_array,float* batch_normalization_gamma_array,float* batch_normalization_beta_array,float* conv2d_1_output_array,float* conv2d_1_padded_input_array,float* conv2d_1_kernel_array,float* conv2d_1_bias_array,float* batch_normalization_1_output_array,float* batch_normalization_1_mean_array,float* batch_normalization_1_stdev_array,float* batch_normalization_1_gamma_array,float* batch_normalization_1_beta_array,float* conv2d_2_output_array,float* conv2d_2_padded_input_array,float* conv2d_2_kernel_array,float* conv2d_2_bias_array,float* batch_normalization_2_output_array,float* batch_normalization_2_mean_array,float* batch_normalization_2_stdev_array,float* batch_normalization_2_gamma_array,float* batch_normalization_2_beta_array,float* add_output_array,float* conv2d_3_output_array,float* conv2d_3_padded_input_array,float* conv2d_3_kernel_array,float* conv2d_3_bias_array,float* batch_normalization_3_output_array,float* batch_normalization_3_mean_array,float* batch_normalization_3_stdev_array,float* batch_normalization_3_gamma_array,float* batch_normalization_3_beta_array,float* conv2d_4_output_array,float* conv2d_4_padded_input_array,float* conv2d_4_kernel_array,float* conv2d_4_bias_array,float* conv2d_5_output_array,float* conv2d_5_padded_input_array,float* conv2d_5_kernel_array,float* conv2d_5_bias_array,float* batch_normalization_4_output_array,float* batch_normalization_4_mean_array,float* batch_normalization_4_stdev_array,float* batch_normalization_4_gamma_array,float* batch_normalization_4_beta_array,float* add_1_output_array,float* conv2d_6_output_array,float* conv2d_6_padded_input_array,float* conv2d_6_kernel_array,float* conv2d_6_bias_array,float* batch_normalization_5_output_array,float* batch_normalization_5_mean_array,float* batch_normalization_5_stdev_array,float* batch_normalization_5_gamma_array,float* batch_normalization_5_beta_array,float* conv2d_7_output_array,float* conv2d_7_padded_input_array,float* conv2d_7_kernel_array,float* conv2d_7_bias_array,float* conv2d_8_output_array,float* conv2d_8_padded_input_array,float* conv2d_8_kernel_array,float* conv2d_8_bias_array,float* batch_normalization_6_output_array,float* batch_normalization_6_mean_array,float* batch_normalization_6_stdev_array,float* batch_normalization_6_gamma_array,float* batch_normalization_6_beta_array,float* add_2_output_array,float* average_pooling2d_output_array,float* flatten_output_array,float* dense_kernel_array,float* dense_bias_array) { 

size_t conv2d_stride[2] = {1,1}; 
size_t conv2d_dilation[2] = {1,1}; 
k2c_tensor conv2d_output = {conv2d_output_array,3,16384,{32,32,16, 1, 1}}; 
k2c_tensor conv2d_padded_input = {conv2d_padded_input_array,3,3468,{34,34, 3, 1, 1}}; 
size_t conv2d_pad[4] = {1,1,1,1}; 
float conv2d_fill = 0.0f; 
k2c_tensor conv2d_kernel = {conv2d_kernel_array,4,432,{ 3, 3, 3,16, 1}}; 
k2c_tensor conv2d_bias = {conv2d_bias_array,1,16,{16, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_output = {batch_normalization_output_array,3,16384,{32,32,16, 1, 1}}; 
size_t batch_normalization_axis = 2; 
k2c_tensor batch_normalization_mean = {batch_normalization_mean_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_stdev = {batch_normalization_stdev_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_gamma = {batch_normalization_gamma_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_beta = {batch_normalization_beta_array,1,16,{16, 1, 1, 1, 1}}; 


size_t conv2d_1_stride[2] = {1,1}; 
size_t conv2d_1_dilation[2] = {1,1}; 
k2c_tensor conv2d_1_output = {conv2d_1_output_array,3,16384,{32,32,16, 1, 1}}; 
k2c_tensor conv2d_1_padded_input = {conv2d_1_padded_input_array,3,18496,{34,34,16, 1, 1}}; 
size_t conv2d_1_pad[4] = {1,1,1,1}; 
float conv2d_1_fill = 0.0f; 
k2c_tensor conv2d_1_kernel = {conv2d_1_kernel_array,4,2304,{ 3, 3,16,16, 1}}; 
k2c_tensor conv2d_1_bias = {conv2d_1_bias_array,1,16,{16, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_1_output = {batch_normalization_1_output_array,3,16384,{32,32,16, 1, 1}}; 
size_t batch_normalization_1_axis = 2; 
k2c_tensor batch_normalization_1_mean = {batch_normalization_1_mean_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_1_stdev = {batch_normalization_1_stdev_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_1_gamma = {batch_normalization_1_gamma_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_1_beta = {batch_normalization_1_beta_array,1,16,{16, 1, 1, 1, 1}}; 


size_t conv2d_2_stride[2] = {1,1}; 
size_t conv2d_2_dilation[2] = {1,1}; 
k2c_tensor conv2d_2_output = {conv2d_2_output_array,3,16384,{32,32,16, 1, 1}}; 
k2c_tensor conv2d_2_padded_input = {conv2d_2_padded_input_array,3,18496,{34,34,16, 1, 1}}; 
size_t conv2d_2_pad[4] = {1,1,1,1}; 
float conv2d_2_fill = 0.0f; 
k2c_tensor conv2d_2_kernel = {conv2d_2_kernel_array,4,2304,{ 3, 3,16,16, 1}}; 
k2c_tensor conv2d_2_bias = {conv2d_2_bias_array,1,16,{16, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_2_output = {batch_normalization_2_output_array,3,16384,{32,32,16, 1, 1}}; 
size_t batch_normalization_2_axis = 2; 
k2c_tensor batch_normalization_2_mean = {batch_normalization_2_mean_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_2_stdev = {batch_normalization_2_stdev_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_2_gamma = {batch_normalization_2_gamma_array,1,16,{16, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_2_beta = {batch_normalization_2_beta_array,1,16,{16, 1, 1, 1, 1}}; 


k2c_tensor add_output = {add_output_array,3,16384,{32,32,16, 1, 1}}; 
size_t add_num_tensors0 = 2; 


size_t conv2d_3_stride[2] = {2,2}; 
size_t conv2d_3_dilation[2] = {1,1}; 
k2c_tensor conv2d_3_output = {conv2d_3_output_array,3,8192,{16,16,32, 1, 1}}; 
k2c_tensor conv2d_3_padded_input = {conv2d_3_padded_input_array,3,18496,{34,34,16, 1, 1}}; 
size_t conv2d_3_pad[4] = {1,1,1,1}; 
float conv2d_3_fill = 0.0f; 
k2c_tensor conv2d_3_kernel = {conv2d_3_kernel_array,4,4608,{ 3, 3,16,32, 1}}; 
k2c_tensor conv2d_3_bias = {conv2d_3_bias_array,1,32,{32, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_3_output = {batch_normalization_3_output_array,3,8192,{16,16,32, 1, 1}}; 
size_t batch_normalization_3_axis = 2; 
k2c_tensor batch_normalization_3_mean = {batch_normalization_3_mean_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_3_stdev = {batch_normalization_3_stdev_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_3_gamma = {batch_normalization_3_gamma_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_3_beta = {batch_normalization_3_beta_array,1,32,{32, 1, 1, 1, 1}}; 


size_t conv2d_4_stride[2] = {1,1}; 
size_t conv2d_4_dilation[2] = {1,1}; 
k2c_tensor conv2d_4_output = {conv2d_4_output_array,3,8192,{16,16,32, 1, 1}}; 
k2c_tensor conv2d_4_padded_input = {conv2d_4_padded_input_array,3,10368,{18,18,32, 1, 1}}; 
size_t conv2d_4_pad[4] = {1,1,1,1}; 
float conv2d_4_fill = 0.0f; 
k2c_tensor conv2d_4_kernel = {conv2d_4_kernel_array,4,9216,{ 3, 3,32,32, 1}}; 
k2c_tensor conv2d_4_bias = {conv2d_4_bias_array,1,32,{32, 1, 1, 1, 1}}; 

 
size_t conv2d_5_stride[2] = {2,2}; 
size_t conv2d_5_dilation[2] = {1,1}; 
k2c_tensor conv2d_5_output = {conv2d_5_output_array,3,8192,{16,16,32, 1, 1}}; 
k2c_tensor conv2d_5_padded_input = {conv2d_5_padded_input_array,3,16384,{32,32,16, 1, 1}}; 
size_t conv2d_5_pad[4] = {0,0,0,0}; 
float conv2d_5_fill = 0.0f; 
k2c_tensor conv2d_5_kernel = {conv2d_5_kernel_array,4,512,{ 1, 1,16,32, 1}}; 
k2c_tensor conv2d_5_bias = {conv2d_5_bias_array,1,32,{32, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_4_output = {batch_normalization_4_output_array,3,8192,{16,16,32, 1, 1}}; 
size_t batch_normalization_4_axis = 2; 
k2c_tensor batch_normalization_4_mean = {batch_normalization_4_mean_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_4_stdev = {batch_normalization_4_stdev_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_4_gamma = {batch_normalization_4_gamma_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_4_beta = {batch_normalization_4_beta_array,1,32,{32, 1, 1, 1, 1}}; 


k2c_tensor add_1_output = {add_1_output_array,3,8192,{16,16,32, 1, 1}}; 
size_t add_1_num_tensors0 = 2; 


size_t conv2d_6_stride[2] = {2,2}; 
size_t conv2d_6_dilation[2] = {1,1}; 
k2c_tensor conv2d_6_output = {conv2d_6_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
k2c_tensor conv2d_6_padded_input = {conv2d_6_padded_input_array,3,10368,{18,18,32, 1, 1}}; 
size_t conv2d_6_pad[4] = {1,1,1,1}; 
float conv2d_6_fill = 0.0f; 
k2c_tensor conv2d_6_kernel = {conv2d_6_kernel_array,4,18432,{ 3, 3,32,64, 1}}; 
k2c_tensor conv2d_6_bias = {conv2d_6_bias_array,1,64,{64, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_5_output = {batch_normalization_5_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
size_t batch_normalization_5_axis = 2; 
k2c_tensor batch_normalization_5_mean = {batch_normalization_5_mean_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_5_stdev = {batch_normalization_5_stdev_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_5_gamma = {batch_normalization_5_gamma_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_5_beta = {batch_normalization_5_beta_array,1,64,{64, 1, 1, 1, 1}}; 


size_t conv2d_7_stride[2] = {1,1}; 
size_t conv2d_7_dilation[2] = {1,1}; 
k2c_tensor conv2d_7_output = {conv2d_7_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
k2c_tensor conv2d_7_padded_input = {conv2d_7_padded_input_array,3,6400,{10,10,64, 1, 1}}; 
size_t conv2d_7_pad[4] = {1,1,1,1}; 
float conv2d_7_fill = 0.0f; 
k2c_tensor conv2d_7_kernel = {conv2d_7_kernel_array,4,36864,{ 3, 3,64,64, 1}}; 
k2c_tensor conv2d_7_bias = {conv2d_7_bias_array,1,64,{64, 1, 1, 1, 1}}; 

 
size_t conv2d_8_stride[2] = {2,2}; 
size_t conv2d_8_dilation[2] = {1,1}; 
k2c_tensor conv2d_8_output = {conv2d_8_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
k2c_tensor conv2d_8_padded_input = {conv2d_8_padded_input_array,3,8192,{16,16,32, 1, 1}}; 
size_t conv2d_8_pad[4] = {0,0,0,0}; 
float conv2d_8_fill = 0.0f; 
k2c_tensor conv2d_8_kernel = {conv2d_8_kernel_array,4,2048,{ 1, 1,32,64, 1}}; 
k2c_tensor conv2d_8_bias = {conv2d_8_bias_array,1,64,{64, 1, 1, 1, 1}}; 

 
k2c_tensor batch_normalization_6_output = {batch_normalization_6_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
size_t batch_normalization_6_axis = 2; 
k2c_tensor batch_normalization_6_mean = {batch_normalization_6_mean_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_6_stdev = {batch_normalization_6_stdev_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_6_gamma = {batch_normalization_6_gamma_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor batch_normalization_6_beta = {batch_normalization_6_beta_array,1,64,{64, 1, 1, 1, 1}}; 


k2c_tensor add_2_output = {add_2_output_array,3,4096,{ 8, 8,64, 1, 1}}; 
size_t add_2_num_tensors0 = 2; 


size_t average_pooling2d_stride[2] = {8,8}; 
size_t average_pooling2d_pool_size[2] = {8,8}; 
k2c_tensor average_pooling2d_output = {average_pooling2d_output_array,3,64,{ 1, 1,64, 1, 1}}; 


k2c_tensor flatten_output = {flatten_output_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_tensor dense_kernel = {dense_kernel_array,2,640,{64,10, 1, 1, 1}}; 
k2c_tensor dense_bias = {dense_bias_array,1,10,{10, 1, 1, 1, 1}}; 
float dense_fwork[704] = {0}; 

 
k2c_pad2d(&conv2d_padded_input,input_1_input,conv2d_fill, 
	conv2d_pad); 
k2c_conv2d(&conv2d_output,&conv2d_padded_input,&conv2d_kernel, 
	&conv2d_bias,conv2d_stride,conv2d_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_output,&conv2d_output,&batch_normalization_mean,&batch_normalization_stdev,&batch_normalization_gamma,&batch_normalization_beta,batch_normalization_axis); 
k2c_relu(batch_normalization_output.array,batch_normalization_output.numel); 
k2c_tensor activation_output; 
activation_output.ndim = batch_normalization_output.ndim; // copy data into output struct 
activation_output.numel = batch_normalization_output.numel; 
memcpy(activation_output.shape,batch_normalization_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_output.array = &batch_normalization_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_1_padded_input,&activation_output,conv2d_1_fill, 
	conv2d_1_pad); 
k2c_conv2d(&conv2d_1_output,&conv2d_1_padded_input,&conv2d_1_kernel, 
	&conv2d_1_bias,conv2d_1_stride,conv2d_1_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_1_output,&conv2d_1_output,&batch_normalization_1_mean,&batch_normalization_1_stdev,&batch_normalization_1_gamma,&batch_normalization_1_beta,batch_normalization_1_axis); 
k2c_relu(batch_normalization_1_output.array,batch_normalization_1_output.numel); 
k2c_tensor activation_1_output; 
activation_1_output.ndim = batch_normalization_1_output.ndim; // copy data into output struct 
activation_1_output.numel = batch_normalization_1_output.numel; 
memcpy(activation_1_output.shape,batch_normalization_1_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_1_output.array = &batch_normalization_1_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_2_padded_input,&activation_1_output,conv2d_2_fill, 
	conv2d_2_pad); 
k2c_conv2d(&conv2d_2_output,&conv2d_2_padded_input,&conv2d_2_kernel, 
	&conv2d_2_bias,conv2d_2_stride,conv2d_2_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_2_output,&conv2d_2_output,&batch_normalization_2_mean,&batch_normalization_2_stdev,&batch_normalization_2_gamma,&batch_normalization_2_beta,batch_normalization_2_axis); 
k2c_add(&add_output,add_num_tensors0,&activation_output,&batch_normalization_2_output); 
k2c_relu(add_output.array,add_output.numel); 
k2c_tensor activation_2_output; 
activation_2_output.ndim = add_output.ndim; // copy data into output struct 
activation_2_output.numel = add_output.numel; 
memcpy(activation_2_output.shape,add_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_2_output.array = &add_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_3_padded_input,&activation_2_output,conv2d_3_fill, 
	conv2d_3_pad); 
k2c_conv2d(&conv2d_3_output,&conv2d_3_padded_input,&conv2d_3_kernel, 
	&conv2d_3_bias,conv2d_3_stride,conv2d_3_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_3_output,&conv2d_3_output,&batch_normalization_3_mean,&batch_normalization_3_stdev,&batch_normalization_3_gamma,&batch_normalization_3_beta,batch_normalization_3_axis); 
k2c_relu(batch_normalization_3_output.array,batch_normalization_3_output.numel); 
k2c_tensor activation_3_output; 
activation_3_output.ndim = batch_normalization_3_output.ndim; // copy data into output struct 
activation_3_output.numel = batch_normalization_3_output.numel; 
memcpy(activation_3_output.shape,batch_normalization_3_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_3_output.array = &batch_normalization_3_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_4_padded_input,&activation_3_output,conv2d_4_fill, 
	conv2d_4_pad); 
k2c_conv2d(&conv2d_4_output,&conv2d_4_padded_input,&conv2d_4_kernel, 
	&conv2d_4_bias,conv2d_4_stride,conv2d_4_dilation,k2c_linear); 
k2c_pad2d(&conv2d_5_padded_input,&activation_2_output,conv2d_5_fill, 
	conv2d_5_pad); 
k2c_conv2d(&conv2d_5_output,&conv2d_5_padded_input,&conv2d_5_kernel, 
	&conv2d_5_bias,conv2d_5_stride,conv2d_5_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_4_output,&conv2d_4_output,&batch_normalization_4_mean,&batch_normalization_4_stdev,&batch_normalization_4_gamma,&batch_normalization_4_beta,batch_normalization_4_axis); 
k2c_add(&add_1_output,add_1_num_tensors0,&conv2d_5_output,&batch_normalization_4_output); 
k2c_relu(add_1_output.array,add_1_output.numel); 
k2c_tensor activation_4_output; 
activation_4_output.ndim = add_1_output.ndim; // copy data into output struct 
activation_4_output.numel = add_1_output.numel; 
memcpy(activation_4_output.shape,add_1_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_4_output.array = &add_1_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_6_padded_input,&activation_4_output,conv2d_6_fill, 
	conv2d_6_pad); 
k2c_conv2d(&conv2d_6_output,&conv2d_6_padded_input,&conv2d_6_kernel, 
	&conv2d_6_bias,conv2d_6_stride,conv2d_6_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_5_output,&conv2d_6_output,&batch_normalization_5_mean,&batch_normalization_5_stdev,&batch_normalization_5_gamma,&batch_normalization_5_beta,batch_normalization_5_axis); 
k2c_relu(batch_normalization_5_output.array,batch_normalization_5_output.numel); 
k2c_tensor activation_5_output; 
activation_5_output.ndim = batch_normalization_5_output.ndim; // copy data into output struct 
activation_5_output.numel = batch_normalization_5_output.numel; 
memcpy(activation_5_output.shape,batch_normalization_5_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_5_output.array = &batch_normalization_5_output.array[0]; // rename for clarity 
k2c_pad2d(&conv2d_7_padded_input,&activation_5_output,conv2d_7_fill, 
	conv2d_7_pad); 
k2c_conv2d(&conv2d_7_output,&conv2d_7_padded_input,&conv2d_7_kernel, 
	&conv2d_7_bias,conv2d_7_stride,conv2d_7_dilation,k2c_linear); 
k2c_pad2d(&conv2d_8_padded_input,&activation_4_output,conv2d_8_fill, 
	conv2d_8_pad); 
k2c_conv2d(&conv2d_8_output,&conv2d_8_padded_input,&conv2d_8_kernel, 
	&conv2d_8_bias,conv2d_8_stride,conv2d_8_dilation,k2c_linear); 
k2c_batch_norm(&batch_normalization_6_output,&conv2d_7_output,&batch_normalization_6_mean,&batch_normalization_6_stdev,&batch_normalization_6_gamma,&batch_normalization_6_beta,batch_normalization_6_axis); 
k2c_add(&add_2_output,add_2_num_tensors0,&conv2d_8_output,&batch_normalization_6_output); 
k2c_relu(add_2_output.array,add_2_output.numel); 
k2c_tensor activation_6_output; 
activation_6_output.ndim = add_2_output.ndim; // copy data into output struct 
activation_6_output.numel = add_2_output.numel; 
memcpy(activation_6_output.shape,add_2_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
activation_6_output.array = &add_2_output.array[0]; // rename for clarity 
k2c_avgpool2d(&average_pooling2d_output,&activation_6_output,average_pooling2d_pool_size, 
	average_pooling2d_stride); 
k2c_flatten(&flatten_output,&average_pooling2d_output); 
k2c_dense(dense_output,&flatten_output,&dense_kernel, 
	&dense_bias,k2c_softmax,dense_fwork); 

 } 

void resnetv1_initialize(float** conv2d_output_array 
,float** conv2d_padded_input_array 
,float** conv2d_kernel_array 
,float** conv2d_bias_array 
,float** batch_normalization_output_array 
,float** batch_normalization_mean_array 
,float** batch_normalization_stdev_array 
,float** batch_normalization_gamma_array 
,float** batch_normalization_beta_array 
,float** conv2d_1_output_array 
,float** conv2d_1_padded_input_array 
,float** conv2d_1_kernel_array 
,float** conv2d_1_bias_array 
,float** batch_normalization_1_output_array 
,float** batch_normalization_1_mean_array 
,float** batch_normalization_1_stdev_array 
,float** batch_normalization_1_gamma_array 
,float** batch_normalization_1_beta_array 
,float** conv2d_2_output_array 
,float** conv2d_2_padded_input_array 
,float** conv2d_2_kernel_array 
,float** conv2d_2_bias_array 
,float** batch_normalization_2_output_array 
,float** batch_normalization_2_mean_array 
,float** batch_normalization_2_stdev_array 
,float** batch_normalization_2_gamma_array 
,float** batch_normalization_2_beta_array 
,float** add_output_array 
,float** conv2d_3_output_array 
,float** conv2d_3_padded_input_array 
,float** conv2d_3_kernel_array 
,float** conv2d_3_bias_array 
,float** batch_normalization_3_output_array 
,float** batch_normalization_3_mean_array 
,float** batch_normalization_3_stdev_array 
,float** batch_normalization_3_gamma_array 
,float** batch_normalization_3_beta_array 
,float** conv2d_4_output_array 
,float** conv2d_4_padded_input_array 
,float** conv2d_4_kernel_array 
,float** conv2d_4_bias_array 
,float** conv2d_5_output_array 
,float** conv2d_5_padded_input_array 
,float** conv2d_5_kernel_array 
,float** conv2d_5_bias_array 
,float** batch_normalization_4_output_array 
,float** batch_normalization_4_mean_array 
,float** batch_normalization_4_stdev_array 
,float** batch_normalization_4_gamma_array 
,float** batch_normalization_4_beta_array 
,float** add_1_output_array 
,float** conv2d_6_output_array 
,float** conv2d_6_padded_input_array 
,float** conv2d_6_kernel_array 
,float** conv2d_6_bias_array 
,float** batch_normalization_5_output_array 
,float** batch_normalization_5_mean_array 
,float** batch_normalization_5_stdev_array 
,float** batch_normalization_5_gamma_array 
,float** batch_normalization_5_beta_array 
,float** conv2d_7_output_array 
,float** conv2d_7_padded_input_array 
,float** conv2d_7_kernel_array 
,float** conv2d_7_bias_array 
,float** conv2d_8_output_array 
,float** conv2d_8_padded_input_array 
,float** conv2d_8_kernel_array 
,float** conv2d_8_bias_array 
,float** batch_normalization_6_output_array 
,float** batch_normalization_6_mean_array 
,float** batch_normalization_6_stdev_array 
,float** batch_normalization_6_gamma_array 
,float** batch_normalization_6_beta_array 
,float** add_2_output_array 
,float** average_pooling2d_output_array 
,float** flatten_output_array 
,float** dense_kernel_array 
,float** dense_bias_array 
) { 

*conv2d_output_array = k2c_read_array("resnetv1conv2d_output_array.csv",16384); 
*conv2d_padded_input_array = k2c_read_array("resnetv1conv2d_padded_input_array.csv",3468); 
*conv2d_kernel_array = k2c_read_array("resnetv1conv2d_kernel_array.csv",432); 
*conv2d_bias_array = k2c_read_array("resnetv1conv2d_bias_array.csv",16); 
*batch_normalization_output_array = k2c_read_array("resnetv1batch_normalization_output_array.csv",16384); 
*batch_normalization_mean_array = k2c_read_array("resnetv1batch_normalization_mean_array.csv",16); 
*batch_normalization_stdev_array = k2c_read_array("resnetv1batch_normalization_stdev_array.csv",16); 
*batch_normalization_gamma_array = k2c_read_array("resnetv1batch_normalization_gamma_array.csv",16); 
*batch_normalization_beta_array = k2c_read_array("resnetv1batch_normalization_beta_array.csv",16); 
*conv2d_1_output_array = k2c_read_array("resnetv1conv2d_1_output_array.csv",16384); 
*conv2d_1_padded_input_array = k2c_read_array("resnetv1conv2d_1_padded_input_array.csv",18496); 
*conv2d_1_kernel_array = k2c_read_array("resnetv1conv2d_1_kernel_array.csv",2304); 
*conv2d_1_bias_array = k2c_read_array("resnetv1conv2d_1_bias_array.csv",16); 
*batch_normalization_1_output_array = k2c_read_array("resnetv1batch_normalization_1_output_array.csv",16384); 
*batch_normalization_1_mean_array = k2c_read_array("resnetv1batch_normalization_1_mean_array.csv",16); 
*batch_normalization_1_stdev_array = k2c_read_array("resnetv1batch_normalization_1_stdev_array.csv",16); 
*batch_normalization_1_gamma_array = k2c_read_array("resnetv1batch_normalization_1_gamma_array.csv",16); 
*batch_normalization_1_beta_array = k2c_read_array("resnetv1batch_normalization_1_beta_array.csv",16); 
*conv2d_2_output_array = k2c_read_array("resnetv1conv2d_2_output_array.csv",16384); 
*conv2d_2_padded_input_array = k2c_read_array("resnetv1conv2d_2_padded_input_array.csv",18496); 
*conv2d_2_kernel_array = k2c_read_array("resnetv1conv2d_2_kernel_array.csv",2304); 
*conv2d_2_bias_array = k2c_read_array("resnetv1conv2d_2_bias_array.csv",16); 
*batch_normalization_2_output_array = k2c_read_array("resnetv1batch_normalization_2_output_array.csv",16384); 
*batch_normalization_2_mean_array = k2c_read_array("resnetv1batch_normalization_2_mean_array.csv",16); 
*batch_normalization_2_stdev_array = k2c_read_array("resnetv1batch_normalization_2_stdev_array.csv",16); 
*batch_normalization_2_gamma_array = k2c_read_array("resnetv1batch_normalization_2_gamma_array.csv",16); 
*batch_normalization_2_beta_array = k2c_read_array("resnetv1batch_normalization_2_beta_array.csv",16); 
*add_output_array = k2c_read_array("resnetv1add_output_array.csv",16384); 
*conv2d_3_output_array = k2c_read_array("resnetv1conv2d_3_output_array.csv",8192); 
*conv2d_3_padded_input_array = k2c_read_array("resnetv1conv2d_3_padded_input_array.csv",18496); 
*conv2d_3_kernel_array = k2c_read_array("resnetv1conv2d_3_kernel_array.csv",4608); 
*conv2d_3_bias_array = k2c_read_array("resnetv1conv2d_3_bias_array.csv",32); 
*batch_normalization_3_output_array = k2c_read_array("resnetv1batch_normalization_3_output_array.csv",8192); 
*batch_normalization_3_mean_array = k2c_read_array("resnetv1batch_normalization_3_mean_array.csv",32); 
*batch_normalization_3_stdev_array = k2c_read_array("resnetv1batch_normalization_3_stdev_array.csv",32); 
*batch_normalization_3_gamma_array = k2c_read_array("resnetv1batch_normalization_3_gamma_array.csv",32); 
*batch_normalization_3_beta_array = k2c_read_array("resnetv1batch_normalization_3_beta_array.csv",32); 
*conv2d_4_output_array = k2c_read_array("resnetv1conv2d_4_output_array.csv",8192); 
*conv2d_4_padded_input_array = k2c_read_array("resnetv1conv2d_4_padded_input_array.csv",10368); 
*conv2d_4_kernel_array = k2c_read_array("resnetv1conv2d_4_kernel_array.csv",9216); 
*conv2d_4_bias_array = k2c_read_array("resnetv1conv2d_4_bias_array.csv",32); 
*conv2d_5_output_array = k2c_read_array("resnetv1conv2d_5_output_array.csv",8192); 
*conv2d_5_padded_input_array = k2c_read_array("resnetv1conv2d_5_padded_input_array.csv",16384); 
*conv2d_5_kernel_array = k2c_read_array("resnetv1conv2d_5_kernel_array.csv",512); 
*conv2d_5_bias_array = k2c_read_array("resnetv1conv2d_5_bias_array.csv",32); 
*batch_normalization_4_output_array = k2c_read_array("resnetv1batch_normalization_4_output_array.csv",8192); 
*batch_normalization_4_mean_array = k2c_read_array("resnetv1batch_normalization_4_mean_array.csv",32); 
*batch_normalization_4_stdev_array = k2c_read_array("resnetv1batch_normalization_4_stdev_array.csv",32); 
*batch_normalization_4_gamma_array = k2c_read_array("resnetv1batch_normalization_4_gamma_array.csv",32); 
*batch_normalization_4_beta_array = k2c_read_array("resnetv1batch_normalization_4_beta_array.csv",32); 
*add_1_output_array = k2c_read_array("resnetv1add_1_output_array.csv",8192); 
*conv2d_6_output_array = k2c_read_array("resnetv1conv2d_6_output_array.csv",4096); 
*conv2d_6_padded_input_array = k2c_read_array("resnetv1conv2d_6_padded_input_array.csv",10368); 
*conv2d_6_kernel_array = k2c_read_array("resnetv1conv2d_6_kernel_array.csv",18432); 
*conv2d_6_bias_array = k2c_read_array("resnetv1conv2d_6_bias_array.csv",64); 
*batch_normalization_5_output_array = k2c_read_array("resnetv1batch_normalization_5_output_array.csv",4096); 
*batch_normalization_5_mean_array = k2c_read_array("resnetv1batch_normalization_5_mean_array.csv",64); 
*batch_normalization_5_stdev_array = k2c_read_array("resnetv1batch_normalization_5_stdev_array.csv",64); 
*batch_normalization_5_gamma_array = k2c_read_array("resnetv1batch_normalization_5_gamma_array.csv",64); 
*batch_normalization_5_beta_array = k2c_read_array("resnetv1batch_normalization_5_beta_array.csv",64); 
*conv2d_7_output_array = k2c_read_array("resnetv1conv2d_7_output_array.csv",4096); 
*conv2d_7_padded_input_array = k2c_read_array("resnetv1conv2d_7_padded_input_array.csv",6400); 
*conv2d_7_kernel_array = k2c_read_array("resnetv1conv2d_7_kernel_array.csv",36864); 
*conv2d_7_bias_array = k2c_read_array("resnetv1conv2d_7_bias_array.csv",64); 
*conv2d_8_output_array = k2c_read_array("resnetv1conv2d_8_output_array.csv",4096); 
*conv2d_8_padded_input_array = k2c_read_array("resnetv1conv2d_8_padded_input_array.csv",8192); 
*conv2d_8_kernel_array = k2c_read_array("resnetv1conv2d_8_kernel_array.csv",2048); 
*conv2d_8_bias_array = k2c_read_array("resnetv1conv2d_8_bias_array.csv",64); 
*batch_normalization_6_output_array = k2c_read_array("resnetv1batch_normalization_6_output_array.csv",4096); 
*batch_normalization_6_mean_array = k2c_read_array("resnetv1batch_normalization_6_mean_array.csv",64); 
*batch_normalization_6_stdev_array = k2c_read_array("resnetv1batch_normalization_6_stdev_array.csv",64); 
*batch_normalization_6_gamma_array = k2c_read_array("resnetv1batch_normalization_6_gamma_array.csv",64); 
*batch_normalization_6_beta_array = k2c_read_array("resnetv1batch_normalization_6_beta_array.csv",64); 
*add_2_output_array = k2c_read_array("resnetv1add_2_output_array.csv",4096); 
*average_pooling2d_output_array = k2c_read_array("resnetv1average_pooling2d_output_array.csv",64); 
*flatten_output_array = k2c_read_array("resnetv1flatten_output_array.csv",64); 
*dense_kernel_array = k2c_read_array("resnetv1dense_kernel_array.csv",640); 
*dense_bias_array = k2c_read_array("resnetv1dense_bias_array.csv",10); 
} 

void resnetv1_terminate(float* conv2d_output_array,float* conv2d_padded_input_array,float* conv2d_kernel_array,float* conv2d_bias_array,float* batch_normalization_output_array,float* batch_normalization_mean_array,float* batch_normalization_stdev_array,float* batch_normalization_gamma_array,float* batch_normalization_beta_array,float* conv2d_1_output_array,float* conv2d_1_padded_input_array,float* conv2d_1_kernel_array,float* conv2d_1_bias_array,float* batch_normalization_1_output_array,float* batch_normalization_1_mean_array,float* batch_normalization_1_stdev_array,float* batch_normalization_1_gamma_array,float* batch_normalization_1_beta_array,float* conv2d_2_output_array,float* conv2d_2_padded_input_array,float* conv2d_2_kernel_array,float* conv2d_2_bias_array,float* batch_normalization_2_output_array,float* batch_normalization_2_mean_array,float* batch_normalization_2_stdev_array,float* batch_normalization_2_gamma_array,float* batch_normalization_2_beta_array,float* add_output_array,float* conv2d_3_output_array,float* conv2d_3_padded_input_array,float* conv2d_3_kernel_array,float* conv2d_3_bias_array,float* batch_normalization_3_output_array,float* batch_normalization_3_mean_array,float* batch_normalization_3_stdev_array,float* batch_normalization_3_gamma_array,float* batch_normalization_3_beta_array,float* conv2d_4_output_array,float* conv2d_4_padded_input_array,float* conv2d_4_kernel_array,float* conv2d_4_bias_array,float* conv2d_5_output_array,float* conv2d_5_padded_input_array,float* conv2d_5_kernel_array,float* conv2d_5_bias_array,float* batch_normalization_4_output_array,float* batch_normalization_4_mean_array,float* batch_normalization_4_stdev_array,float* batch_normalization_4_gamma_array,float* batch_normalization_4_beta_array,float* add_1_output_array,float* conv2d_6_output_array,float* conv2d_6_padded_input_array,float* conv2d_6_kernel_array,float* conv2d_6_bias_array,float* batch_normalization_5_output_array,float* batch_normalization_5_mean_array,float* batch_normalization_5_stdev_array,float* batch_normalization_5_gamma_array,float* batch_normalization_5_beta_array,float* conv2d_7_output_array,float* conv2d_7_padded_input_array,float* conv2d_7_kernel_array,float* conv2d_7_bias_array,float* conv2d_8_output_array,float* conv2d_8_padded_input_array,float* conv2d_8_kernel_array,float* conv2d_8_bias_array,float* batch_normalization_6_output_array,float* batch_normalization_6_mean_array,float* batch_normalization_6_stdev_array,float* batch_normalization_6_gamma_array,float* batch_normalization_6_beta_array,float* add_2_output_array,float* average_pooling2d_output_array,float* flatten_output_array,float* dense_kernel_array,float* dense_bias_array) { 

free(conv2d_output_array); 
free(conv2d_padded_input_array); 
free(conv2d_kernel_array); 
free(conv2d_bias_array); 
free(batch_normalization_output_array); 
free(batch_normalization_mean_array); 
free(batch_normalization_stdev_array); 
free(batch_normalization_gamma_array); 
free(batch_normalization_beta_array); 
free(conv2d_1_output_array); 
free(conv2d_1_padded_input_array); 
free(conv2d_1_kernel_array); 
free(conv2d_1_bias_array); 
free(batch_normalization_1_output_array); 
free(batch_normalization_1_mean_array); 
free(batch_normalization_1_stdev_array); 
free(batch_normalization_1_gamma_array); 
free(batch_normalization_1_beta_array); 
free(conv2d_2_output_array); 
free(conv2d_2_padded_input_array); 
free(conv2d_2_kernel_array); 
free(conv2d_2_bias_array); 
free(batch_normalization_2_output_array); 
free(batch_normalization_2_mean_array); 
free(batch_normalization_2_stdev_array); 
free(batch_normalization_2_gamma_array); 
free(batch_normalization_2_beta_array); 
free(add_output_array); 
free(conv2d_3_output_array); 
free(conv2d_3_padded_input_array); 
free(conv2d_3_kernel_array); 
free(conv2d_3_bias_array); 
free(batch_normalization_3_output_array); 
free(batch_normalization_3_mean_array); 
free(batch_normalization_3_stdev_array); 
free(batch_normalization_3_gamma_array); 
free(batch_normalization_3_beta_array); 
free(conv2d_4_output_array); 
free(conv2d_4_padded_input_array); 
free(conv2d_4_kernel_array); 
free(conv2d_4_bias_array); 
free(conv2d_5_output_array); 
free(conv2d_5_padded_input_array); 
free(conv2d_5_kernel_array); 
free(conv2d_5_bias_array); 
free(batch_normalization_4_output_array); 
free(batch_normalization_4_mean_array); 
free(batch_normalization_4_stdev_array); 
free(batch_normalization_4_gamma_array); 
free(batch_normalization_4_beta_array); 
free(add_1_output_array); 
free(conv2d_6_output_array); 
free(conv2d_6_padded_input_array); 
free(conv2d_6_kernel_array); 
free(conv2d_6_bias_array); 
free(batch_normalization_5_output_array); 
free(batch_normalization_5_mean_array); 
free(batch_normalization_5_stdev_array); 
free(batch_normalization_5_gamma_array); 
free(batch_normalization_5_beta_array); 
free(conv2d_7_output_array); 
free(conv2d_7_padded_input_array); 
free(conv2d_7_kernel_array); 
free(conv2d_7_bias_array); 
free(conv2d_8_output_array); 
free(conv2d_8_padded_input_array); 
free(conv2d_8_kernel_array); 
free(conv2d_8_bias_array); 
free(batch_normalization_6_output_array); 
free(batch_normalization_6_mean_array); 
free(batch_normalization_6_stdev_array); 
free(batch_normalization_6_gamma_array); 
free(batch_normalization_6_beta_array); 
free(add_2_output_array); 
free(average_pooling2d_output_array); 
free(flatten_output_array); 
free(dense_kernel_array); 
free(dense_bias_array); 
} 

