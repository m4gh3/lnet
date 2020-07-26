#include <stddef.h>
#include <stdio.h>
#include <string.h>

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

#ifdef __cplusplus
extern "C" {
#endif

int check_lnet_magic(FILE *fp);
void load_header(lnet_header_ut *header, FILE *fp );
void load_layer_header(lnet_layer_header_ut *header, FILE *fp );
void write_lnet_magic_and_header(lnet_header_ut *header, FILE *fp );
void write_lnet_layer_header(lnet_layer_header_ut *header, FILE *fp );

#ifdef __cplusplus
}
#endif

#endif
