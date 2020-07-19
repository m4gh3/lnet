#include "mmatrix.h"
#include "arrays.h"

#ifndef __L_PROCESSING__
#define __L_PROCESSING__

void gradient_taylor_merge(mmatrix_ut *gradient, size_t arrlen, size_t mergelen );
float taylor_merge(matrix_ut *values, size_t arrlen, size_t mergelen );
void output_gradients_merge(float *expected_vals, matrix_ut *outputs, mmatrix_ut *gradient, size_t arrlen, size_t mergelen );
float triangle(float x, int i );
#endif

