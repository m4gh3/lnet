#include "../include/mmatrix.h"

void matrix_alloc(matrix_ut *m )
{ m->data = malloc(sizeof(float)*m->size[0]); }

void mmatrix_alloc(mmatrix_ut *mm )
{ mm->data = malloc(sizeof(float)*mm->size[0]); }

void copy_matrix_size(matrix_ut *src, matrix_ut *dest )
{
	dest->size[0] = src->size[0];
	dest->size[1] = src->size[1];
}

void copy_mmatrix_size(mmatrix_ut *src, mmatrix_ut *dest )
{
	for(size_t i=0; i < 4; i++ )
		dest->size[i] = src->size[i];
}

void hadamard_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	m_out->size[0] = m0->size[0]; m_out->size[1] = m0->size[1];
	for(size_t i=0; i < m0->size[0]; i++ )
		m_out->data[i] = m0->data[i]*m1->data[i];
}

void set_mul_matrix_matrix_size(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	m_out->size[1] = m1->size[1];
	m_out->size[0] = m0->size[0]/m0->size[1] * m_out->size[1];
}

void mul_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	for(size_t k=0; k < m_out->size[0]; k++ )
		m_out->data[k] = 0;
	for(size_t i_in=0,i_out=0; i_in < m0->size[0]; i_in+=m0->size[1],i_out+=m_out->size[1] )
		for(size_t j_in=0,j_out=0; j_in < m1->size[1]; j_in++,j_out++ )
			for(size_t k0=0,k1=0; k0 < m0->size[1]; k0++,k1+=m_out->size[1] )
				m_out->data[i_out+j_out] += m0->data[i_in+k0]*m1->data[k1+j_in];
}

void set_mul_mmatrix_matrix_size(mmatrix_ut *mm0, matrix_ut *m1, matrix_ut *mm_out )
{
	mm_out->size[3] = mm0->size[3]; mm_out->size[2] = mm0->size[2];
	mm_out->size[1] = mm_out->size[2] * m1->size[1]; mm_out->size[0] = mm_out->size[1] * mm0->size[0] / mm0->size[1];
}

void mul_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	for(size_t m=0; m < mm_out->size[0]; m++ )
		mm_out->data[m] = 0;
	for(size_t i_in=0, i_out=0; i_in < mm0->size[0]; i_in+=mm0->size[1],i_out+=mm_out->size[1] )
		for(size_t j=0,j_out=0; j < m1->size[1]; j++,j_out+=mm_out->size[2] )
		{
			for(size_t k=0,k_in=0; k < m1->size[0]; k+=m1->size[1],k_in+=mm0->size[2] )
			{
				for(size_t m=0; m < mm_out->size[2]; m++ )
					mm_out->data[i_out+j_out+m] += mm0->data[i_in+k_in+m]*m1->data[k+j];
			}
		}
}

void set_mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out )
{
	mm_out->size[3] = mm1->size[3]; mm_out->size[2] = mm1->size[2]; mm_out->size[1] = mm1->size[1];
	mm_out->size[0] = mm_out->size[1] * m0->size[0] / m0->size[1];
}	

void mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out ) 
{
		for(size_t m=0; m < mm_out->size[0]; m++ )
			mm_out->data[m] = 0;
		for(size_t i=0, i_out=0; i < m0->size[0]; i+=m0->size[1],i_out+=mm_out->size[1] )
			for(size_t j_in=0, j_out=0; j_in < mm1->size[1]; j_in+=mm1->size[2],j_out+=mm_out->size[2] )
			{
				for(size_t k=0, k_in=0; k < m0->size[1]; k++,k_in+=mm1->size[1] )
					for(size_t m=0; m < mm_out->size[2]; m++ )
						mm_out->data[i_out+j_out+m] += m0->data[i+k]*mm1->data[j_in+k_in+m];
			} 
}

void hadamard_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	for(size_t i=0,i_in=0; i < m1->size[0]; i+=m1->size[1],i_in+=mm0->size[1] )
       		for(size_t j=0,j_in=0; j < m1->size[1]; j++,j_in+=mm0->size[2] )
			for(size_t m=0; m < mm0->size[2]; m++ )
				mm_out->data[i_in+j_in+m] = mm0->data[i_in+j_in+m]*m1->data[i+j];
}

void set_mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	mm_out->size[3] = m0->size[1]; mm_out->size[2] = m0->size[0]; mm_out->size[1] = m1->size[1]*mm_out->size[2];
	mm_out->size[0] = m0->size[0] / m0->size[1] * mm_out->size[1];
}

void mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	for(size_t m=0; m < mm_out->size[0]; m++ )
		mm_out->data[m] = 0;
	for(size_t i_in=0,i_out=0,m=0; i_in < m0->size[0]; i_in+=m0->size[1],i_out+=mm_out->size[1],m+=mm_out->size[3] )
		for(size_t j_in=0,j_out=0; j_in < m1->size[1]; j_in++,j_out+=mm_out->size[2] )
			for(size_t k0=0,k1=0; k0 < m0->size[1]; k0++,k1+=m1->size[1] )
				for(size_t l=0; l < m0->size[1]; l++ )
					mm_out->data[i_out+j_out+m+l] += ( k0==l ? 1-m0->data[i_in+k0] : -m0->data[i_in+k0] )*m1->data[k1+j_in];
}

void sum_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m] + m1->data[m];
}

void sum_mmatrix_mmatrix(mmatrix_ut *mm0, mmatrix_ut *mm1, mmatrix_ut *mm_out )
{
	for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m] + mm1->data[m];
}

void mul_matrix_scalar(matrix_ut *m0, float k, matrix_ut *m_out )
{
	for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m]*k;
}

void mul_mmatrix_scalar(mmatrix_ut *mm0, float k, mmatrix_ut *mm_out )
{
	for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m]*k;
}
void sum_matrix_scalar(matrix_ut *m0, float f, matrix_ut *m_out )
{
	for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m] + f;
}

void sum_mmatrix_scalar(mmatrix_ut *mm0, float f, mmatrix_ut *mm_out )
{
	for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m]+f;
}

void set_matrix_scalar(matrix_ut *m0, float f )
{
	for(size_t m=0; m < m0->size[0]; m++ )
		m0->data[m] = f;
}

void set_mmatrix_scalar(mmatrix_ut *mm0, float f )
{
	for(size_t m=0; m < mm0->size[0]; m++ )
		mm0->data[m] = f;
}

void print_mmatrix(mmatrix_ut *mm0, char *name )
{
	printf("%s:\n", name );
	for(size_t i=0; i < mm0->size[0]; i+= mm0->size[1] )
	{
		for(size_t m=0; m < mm0->size[2]; m+=mm0->size[3] )
		{
			for(size_t j=0; j < mm0->size[1]; j+=mm0->size[2] )
			{
				for(size_t n=0; n < mm0->size[3]; n++ )
					printf("%f ", mm0->data[i+j+m+n] );
				putchar(' ');
			}
			putchar('\n');
		}
		putchar('\n');
	}
	putchar('\n');
}

void print_matrix(matrix_ut *m0, char *name )
{
	printf("%s:\n", name );
	for(size_t i=0; i < m0->size[0]; i+=m0->size[1] )
	{
		for(size_t j=0; j < m0->size[1]; j++ )
			printf("%f ", m0->data[i+j] );
		putchar('\n');
	}
	putchar('\n');
}

/*void normalize(float *data, size_t len )
{
	float dist=0;
	for(size_t i=0; i < len; i++ )
		dist += data[i]*data[i];
	dist = sqrt(dist);
	for(size_t i=0; i < len; i++ )
		data[i] /= dist;
}

float array_sum(float *data, size_t len )
{
	float sum = 0;
	for(size_t i=0; i < len; i++ )
		sum += data[i];
	return sum;
}

void array_scale_up(float *data, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		data[i]*=factor;
}

void array_scale_down(float *data, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		data[i] /= factor;
}

void array_step(float *dest, float *src, float factor, size_t len )
{
	for(size_t i=0; i < len; i++ )
		dest[i] += src[i]*factor;
}*/
