#include "../include/mmatrix.h"
#include "../include/arrays.h"
#include "../include/processing.h"
#include "../include/lnetlayer.h"
#include <pngdraw/png_plot.h>
#include <time.h>
#include <string.h>

float lnn_plot(float x, lnn_ut *lnn )
{
	set_matrix_scalar(&lnn->input, 0 );
	lnn->input.data[0] = x;
	lnn->input.data[1] = triangle(x, 1 );

	for(size_t ev_step=0; ev_step < lnn->ev_steps; ev_step++ )
	{
		lnn_evolve_step(lnn);
		lnn_evolve_copy(lnn);
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
			(matrix_ut){ .size={ lnn.outputs*(lnn.inputs+lnn.outputs+2), lnn.inputs+lnn.outputs+2 } },
			(matrix_ut){ .size={ lnn.outputs*(lnn.inputs+lnn.outputs+2), lnn.inputs+lnn.outputs+2 } },
			(matrix_ut){ .size={ lnn.outputs*(lnn.inputs+lnn.outputs+2), lnn.inputs+lnn.outputs+2 } }
		},
		.output = (matrix_ut){ .size={lnn.outputs,1} },
		.input = (matrix_ut){ .size={(lnn.inputs+lnn.outputs+2),1} }
	};
	
	lnn_train_data_ut lnn_train_data = (lnn_train_data_ut)
	{
		.lnn = &lnn,
		.in_gradient = 
		{
			(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } },
			(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } },
			(mmatrix_ut){ .size={ lnn.weights[0].size[0]*lnn.weights[0].size[1], lnn.weights[0].size[0], lnn.weights[0].size[0], lnn.weights[0].size[1] } }
		}
	};

	lnn_init(&lnn);	
	lnn_train_data_init(&lnn_train_data);	
	
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

		printf("iteration: %lu/%lu\n", trc, 5000l );

		set_matrix_scalar(&lnn.input, 0 );
		lnn.input.data[0] = (float)rand()/(float)RAND_MAX;
		float expected = 0.001*sin(2*M_PI*lnn.input.data[0]);
		lnn.input.data[1] = triangle(lnn.input.data[0], 1 );

		for(size_t i=0; i < 3; i++ )
			set_mmatrix_scalar(&lnn_train_data.in_gradient[i], 0 );

		for(size_t ev_step=0; ev_step < lnn.ev_steps; ev_step++ )
		{
			lnn_evolve_step(&lnn);
			lnn_train_evolve_step(&lnn_train_data);
			lnn_train_evolve_copy(&lnn_train_data);
			lnn_evolve_copy(&lnn);

		}
			
		gradient_taylor_merge(lnn_train_data.gradient, 0, 4 );
		lnn.output.data[0] = taylor_merge(&lnn.output, 0, 4 );

		size_t merge_offsets[1] = {0};

		float error_dist = output_gradients_merge(&expected, &lnn.output, lnn_train_data.gradient, 1, merge_offsets );
		lnn_gradient_step(lnn_train_data.mom_gradient, lnn_train_data.gradient, error_dist, 0,  &lnn );

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

