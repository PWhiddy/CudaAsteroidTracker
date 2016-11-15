/*
 ============================================================================
 Name        :
 Author      : Peter Whidden
 Version     :
 Copyright   :
 Description :
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cstdlib>

#include <fitsio.h>
#include "GeneratorPSF.h"
#include "FakeAsteroid.h"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}

float *cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
	return rc;
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}

int main(int argc, char* argv[])
{


	float psfSigma = 1.0;
	if (argc > 1) psfSigma = atof(argv[1]);

	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix test = gen->createGaussian(psfSigma);

	gen->printPSF(test);

	FakeAsteroid *asteroid = new FakeAsteroid();





	////////////////////////////////
	fitsfile *fptr;
	/* pointer to the FITS file; defined in fitsio.h */
	int status;
	long fpixel = 1, naxis = 2, nelements;//, exposure;
	long naxes[2] = { 800, 600 };
	/* image is 300 pixels wide by 200 rows */
	short *array = new short[naxes[0]*naxes[1]];
	//short array[200][300];
	status = 0;
	/* initialize status before calling fitsio routines */
	fits_create_file(&fptr, "testfile.fits", &status);
	/* create new file */
	/* Create the primary array image (16-bit short integer pixels */
	fits_create_img(fptr, SHORT_IMG, naxis, naxes, &status);
	/* Write a keyword; must pass the ADDRESS of the value */

	/* Initialize the values in the image with noisy astro */

	asteroid->createImage(array, naxes[0], naxes[1], 0.5, 0.5, test, 50.0, 1.0);
	/*
	int ii, jj;
	for (jj = 0; jj < naxes[1]; jj++)
		for (ii = 0; ii < naxes[0]; ii++)
			array[jj*naxes[0]+ii] = ii + jj;
	*/
	nelements = naxes[0] * naxes[1];

	/* number of pixels to write */
	/* Write the array of integers to the image */
	fits_write_img(fptr, TSHORT, fpixel, nelements, array, &status);
	fits_close_file(fptr, &status);
	fits_report_error(stderr, status);
	////return( status );



	/////////////////////////////

	static const int WORK_SIZE = 65530;
	float *data = new float[WORK_SIZE];

	initialize (data, WORK_SIZE);

	float *recCpu = cpuReciprocal(data, WORK_SIZE);
	float *recGpu = gpuReciprocal(data, WORK_SIZE);
	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

	/* Verify the results */
	std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

	/* Free memory */
	delete[] data;
	delete[] recCpu;
	delete[] recGpu;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

