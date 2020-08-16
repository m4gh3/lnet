#include "../include/mmatrix.h"
#include "../include/arrays.h"
#include "../include/processing.h"
#include <pngdraw/png_plot.h>
#include <time.h>
#include <string.h>

const int inputs = 2;
const int outputs = 10;
const int final_outputs = 1;
const int ev_steps = 10;

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

float lnn_plot(float x, lnn_ut *lnn )
{
	set_matrix_scalar(&lnn->input, 0 );
	lnn->input.data[0] = x;
	lnn->input.data[1] = triangle(x, 1 );

	for(size_t ev_step=0; ev_step < ev_steps; ev_step++ )
	{
		for(size_t i=0; i < 3; i++ )
			mul_matrix_matrix(&lnn->weights[i], &lnn->input, &lnn->choices[i] );
		mul_matrix_scalar(&lnn->choices[2], -1, &lnn->choices[3] );
		sum_matrix_scalar(&lnn->choices[3], 1, &lnn->choices[3] );
		hadamard_matrix_matrix(&lnn->choices[3], &lnn->choices[0], &lnn->output );
		hadamard_matrix_matrix(&lnn->choices[2], &lnn->choices[1], &lnn->pre_output );
		sum_matrix_matrix(&lnn->pre_output, &lnn->output, &lnn->output );
		memcpy(lnn->input.data+inputs, lnn->output.data, lnn->output.size[0]*sizeof(float) );
	}
	float retval = lnn->output.data[0]-lnn->output.data[1]+lnn->output.data[2]-lnn->output.data[3];
	return retval;
}

float expected_plot(float x, void *data )
{ return /*0.001/(1+exp(-8*x+4));*/ 0.001*sin(2*M_PI*x); }

float triangle_plot(float x, void *data )
{
	return 0.001*triangle(x, 1 );
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

int main()
{

	lnn_ut lnn = (lnn_ut)
	{
		.inputs = 2,
		.outputs = 10,
		.final_outputs = 1,
		.ev_steps = 10,
		.weights =
		{ 
			(matrix_ut){ .size={ outputs*(inputs+outputs+2), inputs+outputs+2 } },
			(matrix_ut){ .size={ outputs*(inputs+outputs+2), inputs+outputs+2 } },
			(matrix_ut){ .size={ outputs*(inputs+outputs+2), inputs+outputs+2 } }
		},
		.output = (matrix_ut){ .size={outputs,1} },
		.input = (matrix_ut){ .size={(inputs+outputs+2),1} }
	};

	mmatrix_ut in_gradient[3] =
	{
		(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } },
		(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } },
		(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } }
	};
	mmatrix_ut mom_gradient[3];
	mmatrix_ut gradient[3]; //dO/dC0, dO/dC1, dO/dC2
	mmatrix_ut lodelta_mul_out;
	mmatrix_ut choices_ders[3];
	
	for(size_t i=0; i<3; i++ )
		matrix_alloc(&lnn.weights[i]);
	matrix_alloc(&lnn.output); matrix_alloc(&lnn.input); set_matrix_scalar(&lnn.input, 0 ); lnn.input.data[lnn.input.size[0]-1] = 1;
	set_mul_lodelta_matrix(&lnn.weights[0], &lnn.input, &lodelta_mul_out );
	mmatrix_alloc(&lodelta_mul_out);
	for(size_t i=0; i < 3; i++ )
	{
		copy_matrix_size(&lnn.output, &lnn.choices[i] );
		copy_mmatrix_size(&lodelta_mul_out, &choices_ders[i] );
		copy_mmatrix_size(&lodelta_mul_out, &gradient[i] );
		copy_mmatrix_size(&lodelta_mul_out, &mom_gradient[i] );
		matrix_alloc(&lnn.choices[i]);
		mmatrix_alloc(&choices_ders[i]);
		mmatrix_alloc(&in_gradient[i]);
		mmatrix_alloc(&mom_gradient[i]);
		set_mmatrix_scalar(&mom_gradient[i], 0 );
		set_mmatrix_scalar(&in_gradient[i], 0 );
		mmatrix_alloc(&gradient[i]);
	}
	copy_matrix_size(&lnn.output, &lnn.choices[3] );
	copy_matrix_size(&lnn.output, &lnn.pre_output );
	matrix_alloc(&lnn.choices[3]);
	matrix_alloc(&lnn.pre_output);

	srand(time(0));

	while(1)
	{	
	FILE *weights_file = fopen("weights.data", "rb+" );

	fseek(weights_file, 0, SEEK_END );

	if( ftell(weights_file) == 0 )
	{
		printf("Init weights...\n");
		for(size_t i=0; i < 3; i++ )
			for(size_t j=0; j < lnn.weights[i].size[0]; j+=lnn.weights[i].size[1] )
			{
				float rowsum=1;
				for(size_t k=0;	k < lnn.weights[i].size[1]-1; k++ )
				{
					float randnum = rowsum*((float)rand()/(float)RAND_MAX);
					lnn.weights[i].data[j+k] = randnum;
					rowsum -= randnum;
				}
				lnn.weights[i].data[j+lnn.weights[i].size[1]-1] = rowsum;
			}
	}
	else
	{
		printf("Loading weights...\n");
		fseek(weights_file, 0, SEEK_SET );
		for(int i=0; i < 3; i++ )
		{
			fread(lnn.weights[i].data, sizeof(float), lnn.weights[i].size[0]-1, weights_file );
			print_matrix(&lnn.weights[i], "weights" );
		}
	}
	float sample = 0.05;
	for(size_t trc=0; trc < 5000; trc++ )
	{

		printf("iteration: %lu/%lu\n", trc, 5000 );

		set_matrix_scalar(&lnn.input, 0 );
		lnn.input.data[0] = (float)rand()/(float)RAND_MAX;
		float expected = 0.001*sin(2*M_PI*lnn.input.data[0]);
		//float expected = 0.001/(1+exp(-8*lnn.input.data[0]+4));
		lnn.input.data[1] = triangle(lnn.input.data[0], 1 );

		for(size_t i=0; i < 3; i++ )
			set_mmatrix_scalar(&in_gradient[i], 0 );

		//print_matrix(&lnn.input, "input" );
	
		for(size_t ev_step=0; ev_step < lnn.ev_steps; ev_step++ )
		{
			for(size_t i=0; i < 3; i++ )
				mul_matrix_matrix(&lnn.weights[i], &lnn.input, &lnn.choices[i] );
			mul_matrix_scalar(&lnn.choices[2], -1, &lnn.choices[3] );
			sum_matrix_scalar(&lnn.choices[3], 1, &lnn.choices[3] );
			hadamard_matrix_matrix(&lnn.choices[3], &lnn.choices[0], &lnn.output );
			hadamard_matrix_matrix(&lnn.choices[2], &lnn.choices[1], &lnn.pre_output );
			sum_matrix_matrix(&lnn.pre_output, &lnn.output, &lnn.output );

			//print_matrix(&output, "output" );
		
			for(size_t i=0; i < 3; i++ )
			{
				set_mmatrix_scalar(&gradient[i], 0 );
				for(size_t j=0; j < 3; j++ )
					mul_matrix_mmatrix(&lnn.weights[j], &in_gradient[j], &choices_ders[j] );
				mul_lodelta_matrix(&lnn.weights[i], &lnn.input, &lodelta_mul_out );
				sum_mmatrix_mmatrix(&lodelta_mul_out, &choices_ders[i], &choices_ders[i] );
				sum_mmatrix_scalar(&choices_ders[2], 0, &gradient[i] );
				mul_mmatrix_scalar(&gradient[i], -1, &gradient[i] );
				hadamard_mmatrix_matrix(&gradient[i], &lnn.choices[0], &gradient[i] );
				hadamard_mmatrix_matrix(&choices_ders[0], &lnn.choices[3], &lodelta_mul_out );
				sum_mmatrix_mmatrix(&gradient[i], &lodelta_mul_out, &gradient[i] );
				hadamard_mmatrix_matrix(&choices_ders[2], &lnn.choices[1], &lodelta_mul_out );
				sum_mmatrix_mmatrix(&gradient[i], &lodelta_mul_out, &gradient[i] );
				hadamard_mmatrix_matrix(&choices_ders[1], &lnn.choices[2], &lodelta_mul_out );
				sum_mmatrix_mmatrix(&gradient[i], &lodelta_mul_out, &gradient[i] );
				memcpy(in_gradient[i].data+lnn.inputs*(in_gradient[i].size[1]), gradient[i].data, gradient[i].size[0]*sizeof(float) );
			}
			memcpy(lnn.input.data+lnn.inputs, lnn.output.data, lnn.output.size[0]*sizeof(float) );
		}
			
		gradient_taylor_merge(gradient, 0, 4 );
		lnn.output.data[0] = taylor_merge(&lnn.output, 0, 4 );

		size_t merge_offsets[1] = {0};

		float error_dist = output_gradients_merge(&expected, &lnn.output, gradient, 1, merge_offsets );
		lnn_gradient_step(mom_gradient, gradient, error_dist, 0,  &lnn );

	}

	putchar('\n');

	png_buffer_ut png_buf = (png_buffer_ut){ .pixel_size=3, .width=512, .height=512, .depth=8  };
	png_buffer_init(&png_buf);

	plot(&png_buf, 0, 1, -0.001, 0.001, 0, 511, 0, 511, lnn_plot, &lnn, 0x0000ff );
	plot(&png_buf, 0, 1, -0.001, 0.001, 0, 511, 0, 511, expected_plot, NULL, 0xff0000 );
	plot(&png_buf, 0, 1, -0.001, 0.001, 0, 511, 0, 511, triangle_plot, NULL, 0x00ff00 );


	write_png_buf("result.png", &png_buf );

	fseek(weights_file, 0, SEEK_SET );
		for(int i=0; i < 3; i++ )
			fwrite(lnn.weights[i].data, sizeof(float), lnn.weights[i].size[0]-1, weights_file );

	fclose(weights_file);
	} //end while
}

