#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef __L_MMATRIX__
#define __L_MMATRIX__

typedef struct
{
	size_t size[4];
	float *data;
} mmatrix_ut;

typedef struct
{
	size_t size[2];
	float *data;
} matrix_ut;

#ifdef __cplusplus
extern "C" {
#endif

void matrix_alloc(matrix_ut *m );
void mmatrix_alloc(mmatrix_ut *mm );
void copy_matrix_size(matrix_ut *src, matrix_ut *dest );
void copy_mmatrix_size(mmatrix_ut *src, mmatrix_ut *dest );
void hadamard_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out );
void set_mul_matrix_matrix_size(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out );
void mul_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out );
void set_mul_mmatrix_matrix_size(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out );
void mul_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out );
void set_mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out );
void mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out );
void hadamard_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out );
void set_mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out );
void mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out );
void sum_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out );
void sum_mmatrix_mmatrix(mmatrix_ut *mm0, mmatrix_ut *mm1, mmatrix_ut *mm_out );
void mul_matrix_scalar(matrix_ut *m0, float k, matrix_ut *m_out );
void mul_mmatrix_scalar(mmatrix_ut *mm0, float k, mmatrix_ut *mm_out );
void sum_matrix_scalar(matrix_ut *m0, float f, matrix_ut *m_out );
void sum_mmatrix_scalar(mmatrix_ut *mm0, float f, mmatrix_ut *mm_out );
void set_matrix_scalar(matrix_ut *m0, float f );
void set_mmatrix_scalar(mmatrix_ut *mm0, float f );
void print_mmatrix(mmatrix_ut *mm0, char *name );
void print_matrix(matrix_ut *m0, char *name );

#ifdef __cplusplus
}
#endif

/*void normalize(float *data, size_t len );
float array_sum(float *data, size_t len );
void array_scale_up(float *data, float factor, size_t len );
void array_scale_down(float *data, float factor, size_t len );
void array_step(float *dest, float *src, float factor, size_t len );*/

#endif
