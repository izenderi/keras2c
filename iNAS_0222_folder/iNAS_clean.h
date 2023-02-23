#pragma once 
#include "../include/k2c_tensor_include.h" 
void iNAS_clean(k2c_tensor* INPUT_input, k2c_tensor* FC_END_output,
    float* CONV_0_output_array,float* CONV_0_kernel_array,
    float* CONV_1_output_array,float* CONV_1_kernel_array,
    float* CONV_2_output_array,float* CONV_2_kernel_array,
    float* CONV_3_output_array,float* CONV_3_kernel_array,
    float* CONV_4_output_array,float* CONV_4_kernel_array,
    float* GAVGPOOL_0_output_array,float* FC_END_kernel_array
); 
void iNAS_clean_initialize(float** CONV_0_output_array 
    ,float** CONV_0_kernel_array 
    // ,float** CONV_0_bias_array 
    ,float** CONV_1_output_array 
    ,float** CONV_1_kernel_array 
    // ,float** CONV_1_bias_array 
    ,float** CONV_2_output_array 
    ,float** CONV_2_kernel_array 
    // ,float** CONV_2_bias_array 
    ,float** CONV_3_output_array 
    ,float** CONV_3_kernel_array 
    // ,float** CONV_3_bias_array 
    ,float** CONV_4_output_array 
    ,float** CONV_4_kernel_array 
    // ,float** CONV_4_bias_array 
    ,float** GAVGPOOL_0_output_array 
    ,float** FC_END_kernel_array 
    // ,float** FC_END_bias_array 
); 
void iNAS_clean_terminate(
    float* CONV_0_output_array,float* CONV_0_kernel_array,
    float* CONV_1_output_array,float* CONV_1_kernel_array,
    float* CONV_2_output_array,float* CONV_2_kernel_array,
    float* CONV_3_output_array,float* CONV_3_kernel_array,
    float* CONV_4_output_array,float* CONV_4_kernel_array,
    float* GAVGPOOL_0_output_array,float* FC_END_kernel_array); 
