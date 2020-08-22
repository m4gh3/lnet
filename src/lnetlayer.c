#include "../include/lnetlayer.h"

void lnn_init(lnn_ut *lnn)
{
	for(size_t i=0; i < 3; i++ )
		matrix_alloc(&lnn->weights[i]);
	matrix_alloc(&lnn->output);
	matrix_alloc(&lnn->input);
	set_matrix_scalar(&lnn->input, 0 );
	lnn->input.data[lnn->input.size[0]-1] = 1;
	for(size_t i=0; i < 4; i++ )
	{
		copy_matrix_size(&lnn->output, &lnn->choices[i] );
		matrix_alloc(&lnn->choices[i]);
	}
	copy_matrix_size(&lnn->output, &lnn->pre_output );
	matrix_alloc(&lnn->pre_output);
}

void lnn_train_data_init(lnn_train_data_ut *lnn_train_data )
{
	set_mul_lodelta_matrix(&lnn_train_data->lnn->weights[0], &lnn_train_data->lnn->input, &lnn_train_data->lodelta_mul_out );
	mmatrix_alloc(&lnn_train_data->lodelta_mul_out);
	for(size_t i=0; i < 3; i++ )
	{
		copy_mmatrix_size(&lnn_train_data->lodelta_mul_out, &lnn_train_data->choices_ders[i] );
		copy_mmatrix_size(&lnn_train_data->lodelta_mul_out, &lnn_train_data->gradient[i] );
		copy_mmatrix_size(&lnn_train_data->lodelta_mul_out, &lnn_train_data->mom_gradient[i] );
		mmatrix_alloc(&lnn_train_data->choices_ders[i]);
		mmatrix_alloc(&lnn_train_data->in_gradient[i]);
		mmatrix_alloc(&lnn_train_data->mom_gradient[i]);
		set_mmatrix_scalar(&lnn_train_data->mom_gradient[i], 0 );
		set_mmatrix_scalar(&lnn_train_data->in_gradient[i], 0 );
		mmatrix_alloc(&lnn_train_data->gradient[i]);
	}

}

void lnn_evolve_step(lnn_ut *lnn)
{
	for(size_t i=0; i < 3; i++ )
		mul_matrix_matrix(&lnn->weights[i], &lnn->input, &lnn->choices[i] );
	mul_matrix_scalar(&lnn->choices[2], -1, &lnn->choices[3] );
	sum_matrix_scalar(&lnn->choices[3], 1, &lnn->choices[3] );
	hadamard_matrix_matrix(&lnn->choices[3], &lnn->choices[0], &lnn->output );
	hadamard_matrix_matrix(&lnn->choices[2], &lnn->choices[1], &lnn->pre_output );
	sum_matrix_matrix(&lnn->pre_output, &lnn->output, &lnn->output );
}

void lnn_train_evolve_step(lnn_train_data_ut *lnn_train_data)
{
			for(size_t i=0; i < 3; i++ )
			{
				set_mmatrix_scalar(&lnn_train_data->gradient[i], 0 );
				for(size_t j=0; j < 3; j++ )
					mul_matrix_mmatrix(&lnn_train_data->lnn->weights[j], &lnn_train_data->in_gradient[j], &lnn_train_data->choices_ders[j] );
				mul_lodelta_matrix(&lnn_train_data->lnn->weights[i], &lnn_train_data->lnn->input, &lnn_train_data->lodelta_mul_out );
				sum_mmatrix_mmatrix(&lnn_train_data->lodelta_mul_out, &lnn_train_data->choices_ders[i], &lnn_train_data->choices_ders[i] );
				sum_mmatrix_scalar(&lnn_train_data->choices_ders[2], 0, &lnn_train_data->gradient[i] );
				mul_mmatrix_scalar(&lnn_train_data->gradient[i], -1, &lnn_train_data->gradient[i] );
				hadamard_mmatrix_matrix(&lnn_train_data->gradient[i], &lnn_train_data->lnn->choices[0], &lnn_train_data->gradient[i] );
				hadamard_mmatrix_matrix(&lnn_train_data->choices_ders[0], &lnn_train_data->lnn->choices[3], &lnn_train_data->lodelta_mul_out );
				sum_mmatrix_mmatrix(&lnn_train_data->gradient[i], &lnn_train_data->lodelta_mul_out, &lnn_train_data->gradient[i] );
				hadamard_mmatrix_matrix(&lnn_train_data->choices_ders[2], &lnn_train_data->lnn->choices[1], &lnn_train_data->lodelta_mul_out );
				sum_mmatrix_mmatrix(&lnn_train_data->gradient[i], &lnn_train_data->lodelta_mul_out, &lnn_train_data->gradient[i] );
				hadamard_mmatrix_matrix(&lnn_train_data->choices_ders[1], &lnn_train_data->lnn->choices[2], &lnn_train_data->lodelta_mul_out );
				sum_mmatrix_mmatrix(&lnn_train_data->gradient[i], &lnn_train_data->lodelta_mul_out, &lnn_train_data->gradient[i] );
			}
}

void lnn_evolve_copy(lnn_ut *lnn)
{ memcpy(lnn->input.data+lnn->inputs, lnn->output.data, lnn->output.size[0]*sizeof(float) ); }

void lnn_train_evolve_copy(lnn_train_data_ut *lnn_train_data)
{
	for(size_t i=0; i < 3; i++ )
		memcpy(lnn_train_data->in_gradient[i].data+lnn_train_data->lnn->inputs*(lnn_train_data->in_gradient[i].size[1]), lnn_train_data->gradient[i].data, lnn_train_data->gradient[i].size[0]*sizeof(float) );
}

void lnn_gradient_step(mmatrix_ut *mom_gradient, mmatrix_ut *gradient, float error_dist, size_t gradient_offs, lnn_ut *lnn )
{
	float sqsum=0;
	for(size_t i=0; i < 3; i++ )
	{
		float *gradient_data = gradient[i].data + gradient[i].size[1]*gradient_offs;
       		sqsum += array_squares_sum(gradient_data, gradient[i].size[1] );
	}
	sqsum = sqrt(sqsum);
	if(sqsum)
	{
		for(size_t i=0; i < 3; i++ )
			array_scale_down(gradient[i].data, sqsum, gradient[i].size[1] );
		
		for(size_t i=0; i < 3; i++ )
		{
			float *gradient_data = gradient[i].data + gradient[i].size[1]*gradient_offs;
			array_scale_up(mom_gradient[i].data, 0.9, gradient[i].size[1] );
			array_step(mom_gradient[i].data, gradient_data, 0.1*error_dist, gradient[i].size[1] );
			array_step(lnn->weights[i].data, mom_gradient[i].data, 1, gradient[i].size[1] );

			for(size_t j=0; j < lnn->weights[i].size[0]; j+=lnn->weights[i].size[1] )
			{
				array_abs(lnn->weights[i].data+j, lnn->weights[i].size[1] );
				float f = array_sum(lnn->weights[i].data+j, lnn->weights[i].size[1] );
				array_scale_down(lnn->weights[i].data+j, f, lnn->weights[i].size[1] );
			}
		}
	}
}

