#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "../include/k2c_include.h" 
#include "iNAS_csv.h" 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);
struct timeval GetTimeStamp(); 
int max_out(float* array);

#define ITERATIONS 9000

int main()
{
    // float errors[10];
    // size_t num_tests = 10; 
    // size_t num_outputs = 1;
    float accuracy = 0.0;
    float result = 0.0;
    float* CONV_0_output_array; 
    float* CONV_0_kernel_array; 
    float* CONV_0_bias_array; 
    float* CONV_1_output_array; 
    float* CONV_1_kernel_array; 
    float* CONV_1_bias_array; 
    float* CONV_2_output_array; 
    float* CONV_2_kernel_array; 
    float* CONV_2_bias_array; 
    float* CONV_3_output_array; 
    float* CONV_3_kernel_array; 
    float* CONV_3_bias_array; 
    float* CONV_4_output_array; 
    float* CONV_4_kernel_array; 
    float* CONV_4_bias_array; 
    float* GAVGPOOL_0_output_array; 
    float* FC_END_kernel_array; 
    float* FC_END_bias_array; 
    iNAS_csv_initialize(&CONV_0_output_array,&CONV_0_kernel_array,&CONV_0_bias_array,&CONV_1_output_array,&CONV_1_kernel_array,&CONV_1_bias_array,&CONV_2_output_array,&CONV_2_kernel_array,&CONV_2_bias_array,&CONV_3_output_array,&CONV_3_kernel_array,&CONV_3_bias_array,&CONV_4_output_array,&CONV_4_kernel_array,&CONV_4_bias_array,&GAVGPOOL_0_output_array,&FC_END_kernel_array,&FC_END_bias_array); 

    // =========================================================
    unsigned char buffer[3073];
    int label[10000];

    float input_cifar[3072];
    FILE *ptr;

    float c_FC_END_test_cifar_array[10] = {0};  
    k2c_tensor c_FC_END_test_cifar = {&c_FC_END_test_cifar_array[0],1,10,{10, 1, 1, 1, 1}}; 

    ptr = fopen("/home/johnson/dataset/cifar-10-batches-bin/data_batch_1.bin","rb");  // r for read, b for binary

    for(int i=0; i<ITERATIONS; i++){

        fread(buffer,sizeof(buffer),1,ptr); // read 3073 bytes to our buffer
        
        label[i] = (int)buffer[0];
        // printf("label: %d\n", label[i]);

        for(int j = 1; j<3073; j++){
            input_cifar[j-1] = (float)buffer[j];
        }

        // for(int k = 0; k<3072; k++)
        //     printf("%f ", input_cifar[k]);
        // printf("\n");
        // printf("\n");

        k2c_tensor input_cifar_all = {&input_cifar[0],3,3072,{32,32, 3, 1, 1}};

        iNAS_csv(&input_cifar_all,&c_FC_END_test_cifar,
        CONV_0_output_array,CONV_0_kernel_array,
        CONV_0_bias_array,CONV_1_output_array,
        CONV_1_kernel_array,CONV_1_bias_array,
        CONV_2_output_array,CONV_2_kernel_array,
        CONV_2_bias_array,CONV_3_output_array,
        CONV_3_kernel_array,CONV_3_bias_array,
        CONV_4_output_array,CONV_4_kernel_array,CONV_4_bias_array,
        GAVGPOOL_0_output_array,FC_END_kernel_array,FC_END_bias_array); 

        // output the result
        float x =0;
        float y = 0;
        // for(size_t i=0; i<c_FC_END_test_cifar.numel; i++){
        //     printf("%e ", c_FC_END_test_cifar.array[i]);
        // }
        // printf("\n");
        // printf("result: %d\n", max_out(c_FC_END_test_cifar.array));

        if(max_out(c_FC_END_test_cifar.array)==label[i]){
            result += 1.0 ;
        }

        accuracy = result/i;
    }

    // =========================================================

    iNAS_csv_terminate(CONV_0_output_array,CONV_0_kernel_array,CONV_0_bias_array,CONV_1_output_array,CONV_1_kernel_array,CONV_1_bias_array,CONV_2_output_array,CONV_2_kernel_array,CONV_2_bias_array,CONV_3_output_array,CONV_3_kernel_array,CONV_3_bias_array,CONV_4_output_array,CONV_4_kernel_array,CONV_4_bias_array,GAVGPOOL_0_output_array,FC_END_kernel_array,FC_END_bias_array); 
    // if (maxerror > 1e-05) {return 1;}

    printf("result: %f\n", accuracy);

    return 0;
} 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ 

    float x = 0; 

    float y = 0; 

    for(size_t i=0; i<tensor1->numel; i++){

    y = fabsf(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;
}

int max_out(float* array){
    int res = -1;
    float val = 0;
    for(int i=0;i<sizeof(array);i++){
        if(array[i]>val){
            res = i;
            val = array[i];
        }
    }
    return res;
}


