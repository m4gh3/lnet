#include "mmatrix.h"
#include "arrays.h"

#ifndef __L_PROCESSING__
#define __L_PROCESSING__

#ifdef __cplusplus
extern "C" {
#endif

void gradient_taylor_merge(mmatrix_ut *gradient, size_t arrlen, size_t mergelen );
float taylor_merge(matrix_ut *values, size_t mergestart, size_t mergelen );
float output_gradients_merge(float *expected_vals, matrix_ut *outputs, mmatrix_ut *gradient, size_t mergelen, size_t *merge_offsets );
float triangle(float x, int i );

#ifdef __cplusplus
}
#endif

#endif

