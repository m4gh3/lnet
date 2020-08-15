#include "../include/processing.h"

void gradient_taylor_merge(mmatrix_ut *gradient, size_t mergestart, size_t mergelen )
{
	for(size_t i=0; i < 3; i++ )
	{
		float v=-1;
		for(size_t j=1; j < mergelen; j++ )
		{
			array_step(gradient[i].data+gradient[i].size[1]*mergestart, gradient[i].data+gradient[i].size[1]*(j+mergestart), v, gradient[i].size[1] );
			v=-v;
		}
	}
}

float taylor_merge(matrix_ut *values, size_t mergestart, size_t mergelen )
{
	float v=1, retval=0;
	for(size_t j=0; j < mergelen; j++ )
	{
		retval += values->data[j+mergestart]*v;
		v=-v;
	}
	return retval;
}

float output_gradients_merge(float *expected_vals, matrix_ut *outputs, mmatrix_ut *gradient, size_t mergelen, size_t *merge_offsets )
{
	float errordist;
	for(size_t i=0; i < 3; i++ )
	{
		float *data_0 = gradient[i].data + merge_offsets[0]*gradient[i].size[1];
		array_scale_up(data_0, expected_vals[0]-outputs->data[merge_offsets[0]], gradient[i].size[1] );
		errordist = (expected_vals[0]-outputs->data[merge_offsets[0]])*(expected_vals[0]-outputs->data[merge_offsets[0]]);
		for(size_t j=1; j < mergelen; j++ )
		{
			float *data_j = gradient[i].data + gradient[i].size[1]*merge_offsets[j];
			errordist += (expected_vals[j]-outputs->data[merge_offsets[j]])*(expected_vals[j]-outputs->data[merge_offsets[j]]);
			array_step(data_0, data_j, expected_vals[j]-outputs->data[merge_offsets[j]], gradient[i].size[1] );
		}
	}
	return sqrt(errordist);
}

float triangle(float x, int i )
{
	if(!i) return x;
	return x < 0.5 ? triangle(2*x, i-1 ) : triangle(-2*x+2, i-1 );
}
