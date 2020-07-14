#include "../include/mmatrix.h"

void matrix_alloc(matrix_ut *m )
{ cudaMalloc((void**)&m->data, sizeof(float)*m->size[0] ); }

void mmatrix_alloc(mmatrix_ut *mm )
{ cudaMalloc((void **)&mm->data, sizeof(float)*mm->size[0] ); }

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

__global__ void hadamard_matrix_matrix_kernel(float *m0, float *m1, float *m_out, size_t size )
{
	for(int i=threadIdx.x; i < size; i += blockDim.x )
		m_out[i] = m0[i] * m1[i];
}

void hadamard_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	m_out->size[0] = m0->size[0]; m_out->size[1] = m0->size[1];
	hadamard_matrix_matrix_kernel<<<1,256>>>(m0->data, m1->data, m_out->data, m0->size[0] );
}

void set_mul_matrix_matrix_size(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	m_out->size[1] = m1->size[1];
	m_out->size[0] = m0->size[0]/m0->size[1] * m_out->size[1];
}

__global__ void mul_matrix_matrix_kernel(float *m0, float *m1, float *m_out, size_t m, size_t n, size_t p )
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < m && col < p )
	{
		m_out[row*p+col] = 0;
		for(int vert_m=0, horiz_m=0; horiz_m < n; horiz_m++, vert_m+=p )
			m_out[row*p+col] += m0[row*n+horiz_m] * m1[vert_m+col];
        }

}

void mul_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	/*for(size_t k=0; k < m_out->size[0]; k++ )
		m_out->data[k] = 0;
	for(size_t i_in=0,i_out=0; i_in < m0->size[0]; i_in+=m0->size[1],i_out+=m_out->size[1] )
		for(size_t j_in=0,j_out=0; j_in < m1->size[1]; j_in++,j_out++ )
			for(size_t k0=0,k1=0; k0 < m0->size[1]; k0++,k1+=m_out->size[1] )
				m_out->data[i_out+j_out] += m0->data[i_in+k0]*m1->data[k1+j_in];*/
	size_t m = m0->size[0] / m0->size[1], n = m0->size[1], p = m1->size[1];
	mul_matrix_matrix_kernel<<<1,dim3(m,p,1)>>>(m0->data, m1->data, m_out->data, m, n, p );	
}

void set_mul_mmatrix_matrix_size(mmatrix_ut *mm0, matrix_ut *m1, matrix_ut *mm_out )
{
	mm_out->size[3] = mm0->size[3]; mm_out->size[2] = mm0->size[2];
	mm_out->size[1] = mm_out->size[2] * m1->size[1]; mm_out->size[0] = mm_out->size[1] * mm0->size[0] / mm0->size[1];
}

__global__ void mul_mmatrix_matrix_kernel(float *mm0, float *m1, float *mm_out, size_t m, size_t n, size_t p, size_t l )
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < m && col < p )
	{
		for(int i=threadIdx.z; i < l; i += blockDim.z ) 
			mm_out[row*p+col*l+i] = 0;
		for(int vert_m=0, horiz_m=0; horiz_m < n; horiz_m++, vert_m+=p )
			for(int i=threadIdx.z; i < l; i += blockDim.z )
				mm_out[row*p+col*l+i] += mm0[row*n*l+horiz_m*l+i] * m1[vert_m+col];
        }

}

void mul_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	/*for(size_t m=0; m < mm_out->size[0]; m++ )
		mm_out->data[m] = 0;
	for(size_t i_in=0, i_out=0; i_in < mm0->size[0]; i_in+=mm0->size[1],i_out+=mm_out->size[1] )
		for(size_t j=0,j_out=0; j < m1->size[1]; j++,j_out+=mm_out->size[2] )
		{
			for(size_t k=0,k_in=0; k < m1->size[0]; k+=m1->size[1],k_in+=mm0->size[2] )
			{
				for(size_t m=0; m < mm_out->size[2]; m++ )
					mm_out->data[i_out+j_out+m] += mm0->data[i_in+k_in+m]*m1->data[k+j];
			}
		}*/
	size_t m = mm0->size[0] / mm0->size[1], n = mm0->size[1]/mm0->size[2], p = m1->size[1], l = mm0->size[2];
	mul_mmatrix_matrix_kernel<<<1,dim3(m,p,4)>>>(mm0->data, m1->data, mm_out->data, m, n, p, l );
}

void set_mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out )
{
	mm_out->size[3] = mm1->size[3]; mm_out->size[2] = mm1->size[2]; mm_out->size[1] = mm1->size[1];
	mm_out->size[0] = mm_out->size[1] * m0->size[0] / m0->size[1];
}	

__global__ void mul_matrix_mmatrix_kernel(float *m0, float *mm1, float *mm_out, size_t m, size_t n, size_t p, size_t l )
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < m && col < p )
	{
		for(int i=threadIdx.z; i < l; i += blockDim.z ) 
			mm_out[row*p+col*l+i] = 0;
		for(int vert_m=0, horiz_m=0; horiz_m < n; horiz_m++, vert_m+=p*l )
			for(int i=threadIdx.z; i < l; i += blockDim.z )
				mm_out[row*p*l+col*l+i] += m0[row*n+horiz_m] * mm1[vert_m+col*l+i];
        }

}

void mul_matrix_mmatrix(matrix_ut *m0, mmatrix_ut *mm1, mmatrix_ut *mm_out ) 
{
		/*for(size_t m=0; m < mm_out->size[0]; m++ )
			mm_out->data[m] = 0;
		for(size_t i=0, i_out=0; i < m0->size[0]; i+=m0->size[1],i_out+=mm_out->size[1] )
			for(size_t j_in=0, j_out=0; j_in < mm1->size[1]; j_in+=mm1->size[2],j_out+=mm_out->size[2] )
			{
				for(size_t k=0, k_in=0; k < m0->size[1]; k++,k_in+=mm1->size[1] )
					for(size_t m=0; m < mm_out->size[2]; m++ )
						mm_out->data[i_out+j_out+m] += m0->data[i+k]*mm1->data[j_in+k_in+m];
			} */
	size_t m = m0->size[0] / m0->size[1], n = m0->size[1], p = mm1->size[1]/mm1->size[2], l = mm1->size[2];
	mul_matrix_mmatrix_kernel<<<1,dim3(m,p,4)>>>(m0->data, mm1->data, mm_out->data, m, n, p, l );
}

__global__ void hadamard_mmatrix_matrix_kernel(float *mm0, float *m1, float *mm_out, size_t n, size_t m, size_t l )
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	for(size_t i=threadIdx.z; i < l; i +=blockDim.z )
		mm_out[(row*n+col)*l+i] = mm0[(row*n+col)*l+i] * m1[row*n+col];	

}

void hadamard_mmatrix_matrix(mmatrix_ut *mm0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	/*for(size_t i=0,i_in=0; i < m1->size[0]; i+=m1->size[1],i_in+=mm0->size[1] )
       		for(size_t j=0,j_in=0; j < m1->size[1]; j++,j_in+=mm0->size[2] )
			for(size_t m=0; m < mm0->size[2]; m++ )
				mm_out->data[i_in+j_in+m] = mm0->data[i_in+j_in+m]*m1->data[i+j];*/
	size_t m = m1->size[0]/m1->size[1], n = m1->size[1], l = mm0->size[2];
	hadamard_mmatrix_matrix_kernel<<<1,dim3(m,n,4)>>>(mm0->data, m1->data, mm_out->data, m, n, l );
}

void set_mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	mm_out->size[3] = m0->size[1]; mm_out->size[2] = m0->size[0]; mm_out->size[1] = m1->size[1]*mm_out->size[2];
	mm_out->size[0] = m0->size[0] / m0->size[1] * mm_out->size[1];
}

__global__ void mul_lodelta_matrix_kernel(float *m0, float *m1, float *mm_out, size_t m, size_t n, size_t p )
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if( row < m && col < p )
	{
		//for(int i=threadIdx.z; i < l; i += blockDim.z ) 
		//	mm_out[row*p+col*l+i] = 0;
		for(int vert_m=0, horiz_m=0; horiz_m < n; horiz_m++, vert_m+=p )
			for(int i=threadIdx.z; i < n; i += blockDim.z )
				mm_out[row*(p*m*n+n)+col*m*n+i] += ( (col==i) - m0[row*n+horiz_m] ) * m1[vert_m+col];
        }

}

void mul_lodelta_matrix(matrix_ut *m0, matrix_ut *m1, mmatrix_ut *mm_out )
{
	/*for(size_t m=0; m < mm_out->size[0]; m++ )
		mm_out->data[m] = 0;
	for(size_t i_in=0,i_out=0,m=0; i_in < m0->size[0]; i_in+=m0->size[1],i_out+=mm_out->size[1],m+=mm_out->size[3] )
		for(size_t j_in=0,j_out=0; j_in < m1->size[1]; j_in++,j_out+=mm_out->size[2] )
			for(size_t k0=0,k1=0; k0 < m0->size[1]; k0++,k1+=m1->size[1] )
				for(size_t l=0; l < m0->size[1]; l++ )
					mm_out->data[i_out+j_out+m+l] += ( k0==l ? 1-m0->data[i_in+k0] : -m0->data[i_in+k0] )*m1->data[k1+j_in];*/
	set_mmatrix_scalar(mm_out, 0 );
	size_t m = m0->size[0]/m0->size[1], n = m0->size[1], p = m1->size[1];
	mul_lodelta_matrix_kernel<<<1,dim3(m,p,4)>>>(m0->data, m1->data, mm_out->data, m, n, p );
}

__global__ void sum_kernel(float *m0, float *m1, float *m_out, size_t size )
{
	for(int i=threadIdx.x; i < size; i+=blockDim.x )
		m_out[i] = m0[i] + m1[i];
}

void sum_matrix_matrix(matrix_ut *m0, matrix_ut *m1, matrix_ut *m_out )
{
	/*for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m] + m1->data[m];*/
	sum_kernel<<<1,256>>>(m0->data, m1->data, m_out->data, m0->size[0] );
}

void sum_mmatrix_mmatrix(mmatrix_ut *mm0, mmatrix_ut *mm1, mmatrix_ut *mm_out )
{
	/*for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m] + mm1->data[m];*/
	sum_kernel<<<1,256>>>(mm0->data, mm1->data, mm_out->data, mm0->size[0] );
}

__global__ void mul_scalar_kernel(float *m0, float *m_out, float k, size_t size )
{
	for(int i=threadIdx.x; i < size; i+=blockDim.x )
		m_out[i] = m0[i] * k;
}

void mul_matrix_scalar(matrix_ut *m0, float k, matrix_ut *m_out )
{
	/*for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m]*k;*/
	mul_scalar_kernel<<<1,256>>>(m0->data, m_out->data, k, m0->size[0] );
}

void mul_mmatrix_scalar(mmatrix_ut *mm0, float k, mmatrix_ut *mm_out )
{
	/*for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m]*k;*/
	mul_scalar_kernel<<<1,256>>>(mm0->data, mm_out->data, k, mm0->size[0] );
}

__global__ void sum_scalar_kernel(float *m0, float f, float *m_out, size_t size )
{
	for(int i=threadIdx.x; i < size; i+=blockDim.x )
		m_out[i] = m0[i] + f;
}

void sum_matrix_scalar(matrix_ut *m0, float f, matrix_ut *m_out )
{
	/*for(size_t m=0; m < m0->size[0]; m++ )
		m_out->data[m] = m0->data[m] + f;*/
	sum_scalar_kernel<<<1,256>>>(m0->data, f, m_out->data, m0->size[0] );
}

void sum_mmatrix_scalar(mmatrix_ut *mm0, float f, mmatrix_ut *mm_out )
{
	/*for(size_t m=0; m < mm0->size[0]; m++ )
		mm_out->data[m] = mm0->data[m]+f;*/
	sum_scalar_kernel<<<1,256>>>(mm0->data, f, mm_out->data, mm0->size[0] );
}

__global__ void set_scalar_kernel(float *m0, float f, size_t size )
{
	for(int i=threadIdx.x; i < size; i+=blockDim.x )
		m0[i] = f;
}	

void set_matrix_scalar(matrix_ut *m0, float f )
{ set_scalar_kernel<<<1,256>>>(m0->data, f, m0->size[0] ); }

void set_mmatrix_scalar(mmatrix_ut *mm0, float f )
{ set_scalar_kernel<<<1,256>>>(mm0->data, f, mm0->size[0] ); }

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
