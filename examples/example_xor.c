#include "../include/mmatrix.h"
#include <stdio.h>

int main()
{
	float choice_mats[3][12] = {
					{	1, 0, 0, 0, 0, 0,
						0, 1, 0, 0, 0, 0
					}, 
					{	0, 1, 0, 0, 0, 0,
						0, 0, 0, 0, 1, 0
					},
					{	0, 0, 0, 1, 0, 0,
						0, 0, 0, 0, 0, 1
					}
				};

	float in_values[6] = {0, 1, 0, 0, 1, 0 };
	float lodelta0_values[144];
	matrix_ut xor_c[3] = {  (matrix_ut){ .size={12, 6}, .data = &choice_mats[0][0] }, (matrix_ut){ .size={12, 6}, .data = &choice_mats[1][0] }, (matrix_ut){ .size={12, 6}, .data = &choice_mats[2][0] }  };
	matrix_ut in =  (matrix_ut){ .size={6,1}, .data = in_values };
	matrix_ut out[4];
	mmatrix_ut out_ders[3][3];
	
	for(size_t i=0; i < 3; i++ )
	{
		set_mul_matrix_matrix_size(&xor_c[i], &in, &out[i] );
		matrix_alloc(&out[i]);
	}

	copy_matrix_size(&out[0], &out[3] );
	matrix_alloc(&out[3]);

	for(size_t i=0; i < 2; i++ )
	{
		mul_matrix_matrix(&xor_c[0], &in, &out[0] );
		mul_matrix_matrix(&xor_c[1], &in, &out[1] );
		mul_matrix_matrix(&xor_c[2], &in, &out[2] );
		mul_matrix_scalar(&out[0], -1, &out[3] );
		sum_matrix_scalar(&out[3], 1, &out[3] );
		hadamard_matrix_matrix(&out[3], &out[1], &out[1] );
		hadamard_matrix_matrix(&out[2], &out[0], &out[2] );
		sum_matrix_matrix(&out[2], &out[1], &out[1] );
		in_values[2] = out[1].data[0];
		in_values[3] = out[1].data[1];
	}

	print_matrix(&out[1], "output");

}
