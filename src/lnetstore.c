#include "../include/lnetstore.h"

int check_lnet_magic(FILE *fp)
{
	char str[5];
	fread(str, 1, 4, fp ); str[4] = 0;
	return !strcmp(str, "LNET" );	
}

void load_header(lnet_header_ut *header, FILE *fp )
{ fread(header, sizeof(lnet_header_ut), 1, fp ); }

void load_layer_header(lnet_layer_header_ut *header, FILE *fp )
{ fread(header, sizeof(lnet_layer_header_ut), 1, fp ); }

void write_lnet_magic_and_header(lnet_header_ut *header, FILE *fp )
{
	fwrite("LNET", 1, 4, fp );
	fwrite(header, sizeof(lnet_header_ut), 1, fp );
}

void write_lnet_layer_header(lnet_layer_header_ut *header, FILE *fp )
{ fwrite(header, sizeof(lnet_layer_header_ut), 1, fp ); }
