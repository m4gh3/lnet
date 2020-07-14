#include "../include/arrays.h"

float array_squares_sum(float *data, size_t len )
{
	float sqsum = 0;
	for(size_t i=0; i < len; i++ )
		sqsum += data[i]*data[i];
	return sqsum;
}

float array_sum(float *data, size_t len )
{
	float sum = 0;
	for(size_t i=0; i < len; i++ )
		sum += data[i];
	return sum;
}

void array_scale_up(float *data, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		data[i]*=factor;
}

void array_scale_down(float *data, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		data[i] /= factor;
}

void array_step(float *dest, float *src, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		dest[i] += src[i]*factor;
}

void array_abs(float *data, size_t len )
{
	for(size_t i=0; i < len; i++ )
		data[i] = fabs(data[i]);
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

