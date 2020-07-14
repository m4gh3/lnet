#include "../include/mmatrix.h"
#include "../include/arrays.h"
#include <time.h> 

int main()
{
	float identity_values[4] = { 0.3, 0.5, 0.2 };
	float in_values[3]={1,1,0};
	matrix_ut identity = (matrix_ut){ .size={3,3}, .data=identity_values };
	matrix_ut input = (matrix_ut){ .size={3,1}, .data=in_values }, output;
	mmatrix_ut gradient;

	srand(time(0));	

	set_mul_lodelta_matrix(&identity, &input, &gradient );
	printf("gradient size: %lu,%lu,%lu,%lu\n", gradient.size[0], gradient.size[1], gradient.size[2], gradient.size[3] );
	
	set_mul_matrix_matrix_size(&identity, &input, &output );
	printf("output size: %lu,%lu\n", output.size[0], output.size[1] );

       	mmatrix_alloc(&gradient);
	matrix_alloc(&output);

	for(size_t i=0; i < 200; i++ )
	{
		float scalef;
		in_values[0] = ((float)rand()/(float)RAND_MAX);
		in_values[1] = ((float)rand()/(float)RAND_MAX);
		printf("test values: %f %f\n", in_values[0], in_values[1] );
		mul_lodelta_matrix(&identity, &input, &gradient );
		normalize(gradient.data, 3 );
		print_mmatrix(&gradient, "gradient" );
		print_matrix(&identity, "identity" );
		mul_matrix_matrix(&identity, &input, &output );
		if(output.data[0] != in_values[0] )
		{
			if(output.data[0] > in_values[0] )
				array_step(identity_values, gradient.data, -0.01, 3 );
			else
				array_step(identity_values, gradient.data,  0.01, 3 );
			array_abs(identity_values, 3 );
			scalef = array_sum(identity_values, 3 );
			array_scale_down(identity_values, scalef, 3 );	
		}	
	}
}
