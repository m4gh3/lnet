#include <math.h>
#include <stddef.h>

#ifndef __L_ARRAYS__
#define __L_ARRAYS__

#ifdef __cplusplus
extern "C" {
#endif

float array_squares_sum(float *data, size_t len );
float array_sum(float *data, size_t len );
void array_scale_up(float *data, float factor, size_t len );
void array_scale_down(float *data, float factor, size_t len );
void array_step(float *dest, float *src, float factor, size_t len );
void array_abs(float *data, size_t len );
void normalize(float *data, size_t len );

#ifdef __cplusplus
}
#endif

#endif
