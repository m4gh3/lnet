#include "../include/processing.h"

void gradient_taylor_merge(mmatrix_ut *gradient, size_t arrlen, size_t mergelen )
{
	for(size_t i=0; i < 3; i++ )
	{
		float v=-1;
		for(size_t j=1; j < mergelen; j++ )
		{
			array_step(gradient[i].data, gradient[i].data+gradient[i].size[1]*j, v, gradient[i].size[1] );
			v=-v;
		}
	}
}

float taylor_merge(matrix_ut *values, size_t arrlen, size_t mergelen )
{
	float v=1, retval=0;
	for(size_t j=0; j < mergelen; j++ )
	{
		retval += values->data[j]*v;
		v=-v;
	}
	return retval;
}

void output_gradients_merge(float *expected_vals, matrix_ut *outputs, mmatrix_ut *gradient, size_t arrlen, size_t mergelen )
{
	for(size_t i=0; i < 3; i++ )
	{
		array_scale_up(gradient[i].data, expected_vals[0]-outputs->data[0], gradient[i].size[1] );
		for(size_t j=1; j < mergelen; j++ )
			array_step(gradient[i].data, gradient[i].data+gradient[i].size[1]*j, expected_vals[j]-outputs->data[j], gradient[i].size[1] );
	}
}

float triangle(float x, int i )
{
	if(!i) return x;
	return x < 0.5 ? triangle(2*x, i-1 ) : triangle(-2*x+2, i-1 );
}
