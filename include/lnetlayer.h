#ifndef _L_NETLAYER__
#define _L_NETLAYER__

#include "mmatrix.h"
#include "arrays.h"
#include "string.h"

typedef struct
{
	int inputs;
	int outputs;
	int final_outputs;
	int ev_steps;
	matrix_ut weights[3];
	matrix_ut output, input, choices[4];
	matrix_ut pre_output;
} lnn_ut;

typedef struct
{
	lnn_ut *lnn;
	mmatrix_ut in_gradient[3];
	mmatrix_ut mom_gradient[3];
	mmatrix_ut gradient[3]; //dO/dC0, dO/dC1, dO/dC2
	mmatrix_ut lodelta_mul_out;
	mmatrix_ut choices_ders[3];

} lnn_train_data_ut;

void lnn_init(lnn_ut *lnn);
void lnn_train_data_init(lnn_train_data_ut *lnn_train_data );
void lnn_evolve_step(lnn_ut *lnn);
void lnn_train_evolve_step(lnn_train_data_ut *lnn_train_data);
void lnn_evolve_copy(lnn_ut *lnn);
void lnn_train_evolve_copy(lnn_train_data_ut *lnn_train_data);
void lnn_gradient_step(mmatrix_ut *mom_gradient, mmatrix_ut *gradient, float error_dist, size_t gradient_offs, lnn_ut *lnn );


#endif
