#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <time.h>

#define LOOP_TIMES (10241)

#define THREAD_PER_BLOCK 128
#define BLOCK_NUM 2048
#define NUM_THREAD THREAD_PER_BLOCK*BLOCK_NUM

#define CU_NUM  64
#define BANK_NUM 32

#define MAX_INT 0x7fffffff


__device__
void thread_swap(float* v1, float* v2)
{
	float reg_tmp;	
	reg_tmp = *v1;
	(*v1) = (*v2) + 1;
	*v2 = reg_tmp + 1;
	//lds_mem[BANK_NUM*(hipThreadIdx_y) + hipThreadIdx_x] = lds_mem[BANK_NUM*(hipThreadIdx_y+4) + hipThreadIdx_x]+1;
	//lds_mem[BANK_NUM*hipThreadIdx_y + hipThreadIdx_x + THREAD_PER_BLOCK] = reg_src1+1;		
}

__global__ 
void lds_fetch(float* dst, float* src, int* runtime)
{
	unsigned int thd_id = THREAD_PER_BLOCK*hipBlockIdx_x +hipBlockDim_x*hipThreadIdx_y + hipThreadIdx_x;
	
	__shared__ float lds_mem[2 * THREAD_PER_BLOCK];
	lds_mem[hipBlockDim_x*hipThreadIdx_y + hipThreadIdx_x]                    = src[thd_id];
	lds_mem[hipBlockDim_x*hipThreadIdx_y + hipThreadIdx_x + THREAD_PER_BLOCK] = src[thd_id];

	__syncthreads();
	int i = 0;
	float* lds_addr_1 = lds_mem + BANK_NUM*hipThreadIdx_y + hipThreadIdx_x;
	float* lds_addr_2 = lds_mem + BANK_NUM*hipThreadIdx_y + hipThreadIdx_x + THREAD_PER_BLOCK;

	clock_t begin = clock();
	for(;i < LOOP_TIMES; i++)
		thread_swap(lds_addr_1, lds_addr_2);	
	//__syncthreads();
	clock_t end = clock();

	dst[thd_id] = lds_mem[BANK_NUM*hipThreadIdx_y + hipThreadIdx_x];
	runtime[thd_id] = (int)(end - begin);
}	
	

int main(int argc, char** argv)
{
	// set grid sized 8x8 contianing only one block for each,
	// whose size is 8 * 8, just one warp
	float *hsrc = (float*)malloc(NUM_THREAD*sizeof(float));
	for(int i = 0; i < NUM_THREAD; i++)
	{
		hsrc[i] = i * 1.0;
	}
	
	float *hdst = (float*)malloc(NUM_THREAD*sizeof(float));
	for(int i = 0; i < NUM_THREAD; i++)
	{
		hdst[i] = 0;
	}

	int *runtime = (int*)malloc(NUM_THREAD*sizeof(int));
	for(int i = 0; i < NUM_THREAD; i ++)
	{
		runtime[i] = 0;
	}

	float *dsrc;
	float *ddst;
	int *dtime;

	hipMalloc(&dsrc, NUM_THREAD*sizeof(float));
	hipMalloc(&ddst, NUM_THREAD*sizeof(float));
	hipMalloc(&dtime, NUM_THREAD*sizeof(int));

	hipMemcpy(dsrc, hsrc, NUM_THREAD*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(ddst, hdst, NUM_THREAD*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(dtime, runtime, NUM_THREAD*sizeof(int), hipMemcpyHostToDevice);

	hipLaunchKernelGGL(lds_fetch, dim3(BLOCK_NUM), dim3(BANK_NUM, THREAD_PER_BLOCK/BANK_NUM),
						0, 0,
						ddst, dsrc, dtime);

	hipMemcpy(runtime, dtime, NUM_THREAD*sizeof(int), hipMemcpyDeviceToHost);

#ifdef debug_1
	for(int i =0; i < NUM_THREAD; i++)
	{
		printf("Cycle(%d) = %d\n",i,runtime[i]);
	}
#endif
	int sum = MAX_INT;
	for(int i = 0; i < NUM_THREAD; i++)
	{
		sum = (sum < runtime[i]) ? sum : runtime[i];
	}

	float data_load_num =  THREAD_PER_BLOCK * 4 * (LOOP_TIMES * 4.0/1024);
	data_load_num *= CU_NUM;
	printf("Everage Time(cycle)\tDataNumer(MB)\tBandWidth(TB/s)\n");
	printf("\t%d\t\t  %f\t\t  %f\n", sum, data_load_num/(1024), data_load_num*1.25/sum);
	return 0;
}
