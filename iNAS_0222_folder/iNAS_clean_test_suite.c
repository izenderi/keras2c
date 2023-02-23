#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "../include/k2c_include.h" 
#include "iNAS_clean.h" 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);
struct timeval GetTimeStamp();
int max_out_10(float* array);

int main(int argc, char* argv[])
{
    float* CONV_0_output_array; 
    float* CONV_0_kernel_array; 

    float* CONV_1_output_array; 
    float* CONV_1_kernel_array; 

    float* CONV_2_output_array; 
    float* CONV_2_kernel_array; 

    float* CONV_3_output_array; 
    float* CONV_3_kernel_array; 

    float* CONV_4_output_array; 
    float* CONV_4_kernel_array; 

    float* GAVGPOOL_0_output_array; 
    float* FC_END_kernel_array;

    int counter;
    if(argc==1)
        printf("\nNo Extra Command Line Argument Passed Other Than Program Name\n");
    if(argc>=2)
    {
        printf("\nNumber Of Arguments Passed: %d\n",argc);
        printf("\n----Following Are The Command Line Arguments Passed----");
        for(counter=0;counter<argc;counter++)
            printf("\nargv[%d]: %s",counter,argv[counter]);
        printf("\n");
    }

    for(int i=0;i<70;i++){
        printf("=");
    }
    printf("\n");

    printf("Starting to run %s\n", argv[0]);

    iNAS_clean_initialize(
        &CONV_0_output_array,&CONV_0_kernel_array,
        &CONV_1_output_array,&CONV_1_kernel_array,
        &CONV_2_output_array,&CONV_2_kernel_array,
        &CONV_3_output_array,&CONV_3_kernel_array,
        &CONV_4_output_array,&CONV_4_kernel_array,
        &GAVGPOOL_0_output_array,&FC_END_kernel_array); 

    //=========================================================
    float* data_buffer;
    float* label_buffer;

    float result = 0.0;
    float accuracy = 0.0;

    float label_line[10];
    int label[10000];
    float input_cifar[3072];

    // FILE *ptr_data, *ptr_label;

    float c_FC_END_test_cifar_array[10] = {0};  
    k2c_tensor c_FC_END_test_cifar = {&c_FC_END_test_cifar_array[0],1,10,{10, 1, 1, 1, 1}}; 

    // ptr_data = fopen("data.csv","rb");  // r for read, b for binary
    // ptr_label = fopen("label.csv","rb");
    data_buffer = k2c_read_array("../data.csv", 3072*10000);
    label_buffer = k2c_read_array("../label.csv", 10*10000);

    printf("loading complete\n");

    clock_t t_s = clock(); 
    for(int i=0;i<10000;i++){
        for(int j=0; j<3072; j++){
            input_cifar[j] = data_buffer[i*3072+j];
        }

        for(int j=0; j<10; j++){
            label_line[j] = label_buffer[i*10+j];
        }

        // for(int j=0; j<10; j++){
        //     printf("%f ", label_line[j]);
        // }
        // printf("\n");

        label[i] = max_out_10(label_line);
        // printf("label is: %d\n", label[i]);

        // for(int k = 0; k<3072; k++)
        //     printf("%f ", input_cifar[k]);

        k2c_tensor input_cifar_all = {&input_cifar[0],3,3072,{32,32, 3, 1, 1}};

        iNAS_clean(&input_cifar_all,&c_FC_END_test_cifar,
            CONV_0_output_array,CONV_0_kernel_array,
            CONV_1_output_array,CONV_1_kernel_array,
            CONV_2_output_array,CONV_2_kernel_array,
            CONV_3_output_array,CONV_3_kernel_array,
            CONV_4_output_array,CONV_4_kernel_array,
            GAVGPOOL_0_output_array,FC_END_kernel_array); 

        // for(size_t i=0; i<c_FC_END_test_cifar.numel; i++){
        //         printf("%e ", c_FC_END_test_cifar.array[i]);
        //     }
        // printf("\n");

        // printf("pred label: %d\n", max_out_10(c_FC_END_test_cifar.array));

        if(max_out_10(c_FC_END_test_cifar.array)==label[i]){
            result += 1.0 ;
        }
        accuracy = result/i;

    }
    clock_t t_e = clock();
    printf("Overall time over 10,000 tests: %e s \n", ((double)t_e-t_s)/(double)CLOCKS_PER_SEC);
    printf("Average time over 10,000 tests: %e s \n", ((double)t_e-t_s)/(double)CLOCKS_PER_SEC/(double)10000); 
    printf("acc: %f\n", accuracy);

    iNAS_clean_terminate(
        CONV_0_output_array,CONV_0_kernel_array,
        CONV_1_output_array,CONV_1_kernel_array,
        CONV_2_output_array,CONV_2_kernel_array,
        CONV_3_output_array,CONV_3_kernel_array,
        CONV_4_output_array,CONV_4_kernel_array,
        GAVGPOOL_0_output_array,FC_END_kernel_array); 

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

int max_out_10(float* array){
    int res = -1;
    float val = 0;
    // printf("size_of_arr: %ld", sizeof(array)/sizeof(array[0]));
    for(int i=0;i<10;i++){
        if(array[i]>val){
            res = i;
            val = array[i];
        }
    }
    return res;
}

