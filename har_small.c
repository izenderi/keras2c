#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void har_small(k2c_tensor* INPUT_input, k2c_tensor* FC_END_output) { 

size_t CONV_0_stride = 1; 
size_t CONV_0_dilation = 1; 
float CONV_0_output_array[512] = {0}; 
k2c_tensor CONV_0_output = {&CONV_0_output_array[0],2,512,{128,  4,  1,  1,  1}}; 
float CONV_0_kernel_array[36] = {
+3.32184255e-01f,+1.02320015e+00f,+4.95528728e-01f,-1.24501251e-01f,-5.94973445e-01f,
-7.26287484e-01f,-9.71313596e-01f,-5.21473527e-01f,-1.99183300e-01f,-4.71760482e-01f,
-8.03835928e-01f,+2.50367168e-02f,+9.85830247e-01f,-5.76307476e-01f,+3.26860279e-01f,
+9.83930603e-02f,+4.79148597e-01f,+6.79398060e-01f,-2.59354383e-01f,-1.71564057e-01f,
+7.88530469e-01f,-3.58845055e-01f,-9.30867493e-01f,+1.77421182e-01f,+1.72497064e-01f,
+2.44231194e-01f,-4.25468266e-01f,+8.23483229e-01f,+2.92073518e-01f,-3.42371427e-02f,
+7.31694996e-01f,-4.91465271e-01f,+9.20105428e-02f,-2.19809532e-01f,+3.61504108e-01f,
-7.55626082e-01f,}; 
k2c_tensor CONV_0_kernel = {&CONV_0_kernel_array[0],3,36,{1,9,4,1,1}}; 
float CONV_0_bias_array[4] = {0}; 
k2c_tensor CONV_0_bias = {&CONV_0_bias_array[0],1,4,{4,1,1,1,1}}; 

 
size_t CONV_1_stride = 1; 
size_t CONV_1_dilation = 1; 
float CONV_1_output_array[512] = {0}; 
k2c_tensor CONV_1_output = {&CONV_1_output_array[0],2,512,{128,  4,  1,  1,  1}}; 
float CONV_1_kernel_array[16] = {
+8.60052466e-01f,-4.02327716e-01f,+1.03401542e+00f,+1.26748011e-01f,+9.17711437e-01f,
+1.55373648e-01f,+7.30540514e-01f,-7.19729483e-01f,-1.13713048e-01f,-4.78750020e-01f,
+6.57621503e-01f,+7.50133514e-01f,+2.43188217e-01f,+5.50704002e-01f,-3.17405879e-01f,
-2.53297061e-01f,}; 
k2c_tensor CONV_1_kernel = {&CONV_1_kernel_array[0],3,16,{1,4,4,1,1}}; 
float CONV_1_bias_array[4] = {0}; 
k2c_tensor CONV_1_bias = {&CONV_1_bias_array[0],1,4,{4,1,1,1,1}}; 

 
size_t CONV_2_stride = 1; 
size_t CONV_2_dilation = 1; 
float CONV_2_output_array[512] = {0}; 
k2c_tensor CONV_2_output = {&CONV_2_output_array[0],2,512,{128,  4,  1,  1,  1}}; 
float CONV_2_kernel_array[16] = {
-5.36688328e-01f,-3.17420125e-01f,+6.62003279e-01f,+1.01657733e-01f,-2.32858673e-01f,
-3.91293079e-01f,-7.95899689e-01f,+5.06250858e-01f,+4.33759719e-01f,-7.04226673e-01f,
+5.26465058e-01f,-3.57475042e-01f,+1.17682993e+00f,-8.18097413e-01f,-8.66861880e-01f,
-2.31357589e-01f,}; 
k2c_tensor CONV_2_kernel = {&CONV_2_kernel_array[0],3,16,{1,4,4,1,1}}; 
float CONV_2_bias_array[4] = {0}; 
k2c_tensor CONV_2_bias = {&CONV_2_bias_array[0],1,4,{4,1,1,1,1}}; 

 
float GAVGPOOL_0_output_array[4] = {0}; 
k2c_tensor GAVGPOOL_0_output = {&GAVGPOOL_0_output_array[0],1,4,{4,1,1,1,1}}; 


float FC_END_kernel_array[24] = {
-1.06309688e+00f,+1.38160422e-01f,-1.15532625e+00f,-4.37850431e-02f,-5.11843264e-01f,
+1.02998102e+00f,-4.77331758e-01f,+4.68010664e-01f,-4.77438122e-01f,+5.59557199e-01f,
-3.05411458e-01f,-3.90904546e-02f,+6.06051862e-01f,+3.02866220e-01f,+6.75805628e-01f,
-5.09378850e-01f,-1.11162329e+00f,-6.03559315e-01f,+2.97577828e-01f,-9.57325757e-01f,
-9.77224648e-01f,-1.92064852e-01f,+5.99655628e-01f,-1.72391180e-02f,}; 
k2c_tensor FC_END_kernel = {&FC_END_kernel_array[0],2,24,{4,6,1,1,1}}; 
float FC_END_bias_array[6] = {0}; 
k2c_tensor FC_END_bias = {&FC_END_bias_array[0],1,6,{6,1,1,1,1}}; 
float FC_END_fwork[28] = {0}; 

 
k2c_conv1d(&CONV_0_output,INPUT_input,&CONV_0_kernel, 
	&CONV_0_bias,CONV_0_stride,CONV_0_dilation,k2c_relu); 
k2c_conv1d(&CONV_1_output,&CONV_0_output,&CONV_1_kernel, 
	&CONV_1_bias,CONV_1_stride,CONV_1_dilation,k2c_relu); 
k2c_conv1d(&CONV_2_output,&CONV_1_output,&CONV_2_kernel, 
	&CONV_2_bias,CONV_2_stride,CONV_2_dilation,k2c_relu); 
k2c_global_avg_pooling(&GAVGPOOL_0_output,&CONV_2_output); 
k2c_dense(FC_END_output,&GAVGPOOL_0_output,&FC_END_kernel, 
	&FC_END_bias,k2c_softmax,FC_END_fwork); 

 } 

void har_small_initialize() { 

} 

void har_small_terminate() { 

} 

