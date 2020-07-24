#include <stddef.h>

#ifndef __L_NETSORE__
#define __L_NETSORE__

typedef struct
{
	size_t header_len;
	size_t layers_n;
} lnet_header_ut;

typedef struct
{
	size_t inputs, outputs, stepsn;
} lnet_layer_header_ut;

#endif
