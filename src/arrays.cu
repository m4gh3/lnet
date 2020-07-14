#include "../include/arrays.h"

__global__ void array_squares_sum_kernel(float *data, float *dest, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		dest[threadIdx.x] += data[i]*data[i];
}

float array_squares_sum(float *data, size_t len )
{
	float sqsum = 0, *device_sqsums, host_sqsums[64];
	cudaMalloc(&device_sqsums, 64*sizeof(float) );
	array_squares_sum_kernel<<<1,256>>>(data, device_sqsums, len );
	cudaMemcpy(host_sqsums, device_sqsums, 64*sizeof(float), cudaMemcpyDeviceToHost );
	cudaFree(device_sqsums);
	for(size_t i=0; i < 64; i++ )
		sqsum += host_sqsums[i];
	return sqsum;
}

__global__ void array_sum_kernel(float *data, float *dest, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		dest[threadIdx.x] += data[i];
}

float array_sum(float *data, size_t len )
{
	float sum = 0, *device_sums, host_sums[64];
	cudaMalloc(&device_sums, 64*sizeof(float) );
	array_sum_kernel<<<1,256>>>(data, device_sums, len );
	cudaMemcpy(host_sums, device_sums, 64*sizeof(float), cudaMemcpyDeviceToHost );
	cudaFree(device_sums);
	for(size_t i=0; i < 64; i++ )
		sum += host_sums[i];
	return sum;
}

__global__ void array_scale_up_kernel(float *data, float factor, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		data[i]*=factor;
}

void array_scale_up(float *data, float factor, size_t len )
{
	/*for(size_t i=0; i < len; i++ )
		data[i]*=factor;*/
	array_scale_up_kernel<<<1,256>>>(data, factor, len );
}

__global__ void array_scale_down_kernel(float *data, float factor, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		data[i]/=factor;
}

void array_scale_down(float *data, float factor, size_t len )
{
	/*for(size_t i=0; i < len; i++ )
		data[i] /= factor;*/
	array_scale_down_kernel<<<1,256>>>(data, factor, len );
}

__global__ void array_step_kernel(float *dest, float *src, float factor, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		dest[i]+=src[i]*factor;
}

void array_step(float *dest, float *src, float factor, size_t len )
{
	/*for(size_t i=0; i < len; i++ )
		dest[i] += src[i]*factor;*/
	array_step_kernel<<<1,256>>>(dest, src, factor, len );
}

__global__ void array_abs_kernel(float *dest, size_t len )
{
	for(int i=threadIdx.x; i < len; i += blockDim.x )
		dest[i] = fabs(dest[i]);
}

void array_abs(float *data, size_t len )
{
	/*for(size_t i=0; i < len; i++ )
		data[i] = fabs(data[i]);*/
	array_abs_kernel<<<1,256>>>(data, len );
}

void normalize(float *data, size_t len )
{
	/*float dist=0;
	for(size_t i=0; i < len; i++ )
		dist += data[i]*data[i];
	dist = sqrt(dist);
	for(size_t i=0; i < len; i++ )
		data[i] /= dist;*/
	float dist = array_squares_sum(data, len );
	dist = sqrt(dist);
	array_scale_down(data, dist, len );
}

