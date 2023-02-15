#include <math.h> 
 #include <string.h> 
#include "../include/k2c_include.h" 
#include "../include/k2c_tensor_include.h" 

 


void iNAS_csv(k2c_tensor* INPUT_input, k2c_tensor* FC_END_output,float* CONV_0_output_array,float* CONV_0_kernel_array,float* CONV_0_bias_array,float* CONV_1_output_array,float* CONV_1_kernel_array,float* CONV_1_bias_array,float* CONV_2_output_array,float* CONV_2_kernel_array,float* CONV_2_bias_array,float* CONV_3_output_array,float* CONV_3_kernel_array,float* CONV_3_bias_array,float* CONV_4_output_array,float* CONV_4_kernel_array,float* CONV_4_bias_array,float* GAVGPOOL_0_output_array,float* FC_END_kernel_array,float* FC_END_bias_array) { 

size_t CONV_0_stride[2] = {1,1}; 
size_t CONV_0_dilation[2] = {1,1}; 
k2c_tensor CONV_0_output = {CONV_0_output_array,3,8192,{32,32, 8, 1, 1}}; 
k2c_tensor CONV_0_kernel = {CONV_0_kernel_array,4,24,{1,1,3,8,1}}; 
k2c_tensor CONV_0_bias = {CONV_0_bias_array,1,8,{8,1,1,1,1}}; 

 
size_t CONV_1_stride[2] = {1,1}; 
size_t CONV_1_dilation[2] = {1,1}; 
k2c_tensor CONV_1_output = {CONV_1_output_array,3,7200,{30,30, 8, 1, 1}}; 
k2c_tensor CONV_1_kernel = {CONV_1_kernel_array,4,576,{3,3,8,8,1}}; 
k2c_tensor CONV_1_bias = {CONV_1_bias_array,1,8,{8,1,1,1,1}}; 

 
size_t CONV_2_stride[2] = {1,1}; 
size_t CONV_2_dilation[2] = {1,1}; 
k2c_tensor CONV_2_output = {CONV_2_output_array,3,2704,{26,26, 4, 1, 1}}; 
k2c_tensor CONV_2_kernel = {CONV_2_kernel_array,4,800,{5,5,8,4,1}}; 
k2c_tensor CONV_2_bias = {CONV_2_bias_array,1,4,{4,1,1,1,1}}; 

 
size_t CONV_3_stride[2] = {1,1}; 
size_t CONV_3_dilation[2] = {1,1}; 
k2c_tensor CONV_3_output = {CONV_3_output_array,3,9216,{24,24,16, 1, 1}}; 
k2c_tensor CONV_3_kernel = {CONV_3_kernel_array,4,576,{ 3, 3, 4,16, 1}}; 
k2c_tensor CONV_3_bias = {CONV_3_bias_array,1,16,{16, 1, 1, 1, 1}}; 

 
size_t CONV_4_stride[2] = {1,1}; 
size_t CONV_4_dilation[2] = {1,1}; 
k2c_tensor CONV_4_output = {CONV_4_output_array,3,1296,{18,18, 4, 1, 1}}; 
k2c_tensor CONV_4_kernel = {CONV_4_kernel_array,4,3136,{ 7, 7,16, 4, 1}}; 
k2c_tensor CONV_4_bias = {CONV_4_bias_array,1,4,{4,1,1,1,1}}; 

 
k2c_tensor GAVGPOOL_0_output = {GAVGPOOL_0_output_array,1,4,{4,1,1,1,1}}; 


k2c_tensor FC_END_kernel = {FC_END_kernel_array,2,40,{ 4,10, 1, 1, 1}}; 
k2c_tensor FC_END_bias = {FC_END_bias_array,1,10,{10, 1, 1, 1, 1}}; 
float FC_END_fwork[44] = {0}; 

 
k2c_conv2d(&CONV_0_output,INPUT_input,&CONV_0_kernel, 
	&CONV_0_bias,CONV_0_stride,CONV_0_dilation,k2c_relu); 
k2c_conv2d(&CONV_1_output,&CONV_0_output,&CONV_1_kernel, 
	&CONV_1_bias,CONV_1_stride,CONV_1_dilation,k2c_relu); 
k2c_conv2d(&CONV_2_output,&CONV_1_output,&CONV_2_kernel, 
	&CONV_2_bias,CONV_2_stride,CONV_2_dilation,k2c_relu); 
k2c_conv2d(&CONV_3_output,&CONV_2_output,&CONV_3_kernel, 
	&CONV_3_bias,CONV_3_stride,CONV_3_dilation,k2c_relu); 
k2c_conv2d(&CONV_4_output,&CONV_3_output,&CONV_4_kernel, 
	&CONV_4_bias,CONV_4_stride,CONV_4_dilation,k2c_relu); 
k2c_global_avg_pooling(&GAVGPOOL_0_output,&CONV_4_output); 
k2c_dense(FC_END_output,&GAVGPOOL_0_output,&FC_END_kernel, 
	&FC_END_bias,k2c_softmax,FC_END_fwork); 

 } 

void iNAS_csv_initialize(float** CONV_0_output_array 
,float** CONV_0_kernel_array 
,float** CONV_0_bias_array 
,float** CONV_1_output_array 
,float** CONV_1_kernel_array 
,float** CONV_1_bias_array 
,float** CONV_2_output_array 
,float** CONV_2_kernel_array 
,float** CONV_2_bias_array 
,float** CONV_3_output_array 
,float** CONV_3_kernel_array 
,float** CONV_3_bias_array 
,float** CONV_4_output_array 
,float** CONV_4_kernel_array 
,float** CONV_4_bias_array 
,float** GAVGPOOL_0_output_array 
,float** FC_END_kernel_array 
,float** FC_END_bias_array 
) { 

*CONV_0_output_array = k2c_read_array("iNAS_csvCONV_0_output_array.csv",8192); 
*CONV_0_kernel_array = k2c_read_array("iNAS_csvCONV_0_kernel_array.csv",24); 
*CONV_0_bias_array = k2c_read_array("iNAS_csvCONV_0_bias_array.csv",8); 
*CONV_1_output_array = k2c_read_array("iNAS_csvCONV_1_output_array.csv",7200); 
*CONV_1_kernel_array = k2c_read_array("iNAS_csvCONV_1_kernel_array.csv",576); 
*CONV_1_bias_array = k2c_read_array("iNAS_csvCONV_1_bias_array.csv",8); 
*CONV_2_output_array = k2c_read_array("iNAS_csvCONV_2_output_array.csv",2704); 
*CONV_2_kernel_array = k2c_read_array("iNAS_csvCONV_2_kernel_array.csv",800); 
*CONV_2_bias_array = k2c_read_array("iNAS_csvCONV_2_bias_array.csv",4); 
*CONV_3_output_array = k2c_read_array("iNAS_csvCONV_3_output_array.csv",9216); 
*CONV_3_kernel_array = k2c_read_array("iNAS_csvCONV_3_kernel_array.csv",576); 
*CONV_3_bias_array = k2c_read_array("iNAS_csvCONV_3_bias_array.csv",16); 
*CONV_4_output_array = k2c_read_array("iNAS_csvCONV_4_output_array.csv",1296); 
*CONV_4_kernel_array = k2c_read_array("iNAS_csvCONV_4_kernel_array.csv",3136); 
*CONV_4_bias_array = k2c_read_array("iNAS_csvCONV_4_bias_array.csv",4); 
*GAVGPOOL_0_output_array = k2c_read_array("iNAS_csvGAVGPOOL_0_output_array.csv",4); 
*FC_END_kernel_array = k2c_read_array("iNAS_csvFC_END_kernel_array.csv",40); 
*FC_END_bias_array = k2c_read_array("iNAS_csvFC_END_bias_array.csv",10); 
} 

void iNAS_csv_terminate(float* CONV_0_output_array,float* CONV_0_kernel_array,float* CONV_0_bias_array,float* CONV_1_output_array,float* CONV_1_kernel_array,float* CONV_1_bias_array,float* CONV_2_output_array,float* CONV_2_kernel_array,float* CONV_2_bias_array,float* CONV_3_output_array,float* CONV_3_kernel_array,float* CONV_3_bias_array,float* CONV_4_output_array,float* CONV_4_kernel_array,float* CONV_4_bias_array,float* GAVGPOOL_0_output_array,float* FC_END_kernel_array,float* FC_END_bias_array) { 

free(CONV_0_output_array); 
free(CONV_0_kernel_array); 
free(CONV_0_bias_array); 
free(CONV_1_output_array); 
free(CONV_1_kernel_array); 
free(CONV_1_bias_array); 
free(CONV_2_output_array); 
free(CONV_2_kernel_array); 
free(CONV_2_bias_array); 
free(CONV_3_output_array); 
free(CONV_3_kernel_array); 
free(CONV_3_bias_array); 
free(CONV_4_output_array); 
free(CONV_4_kernel_array); 
free(CONV_4_bias_array); 
free(GAVGPOOL_0_output_array); 
free(FC_END_kernel_array); 
free(FC_END_bias_array); 
} 

