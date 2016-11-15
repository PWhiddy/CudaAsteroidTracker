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
#include <sstream>
#include <ctime>

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

__global__ void convolvePSF(int width, int height, int imageCount, short *images, short *results,
		float *psf, int psfRad, int psfDim)
{
	// Test bounds of image
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	const int minX = max(x-psfRad, 0);
	const int minY = max(y-psfRad, 0);
	const int maxX = min(x+psfRad, width);
	const int maxY = min(y+psfRad, height);
	const int dx = maxX-minX;
	const int dy = maxY-minY;
	if (dx < 1 || dy < 1) return;
	// Read Image
	__shared__ float convArea[5][5]; //convArea[dx][dy];

	float sum = 0.0;
	for (int i=0; i<dx; ++i)
	{
		for (int j=0; j<dy; ++j)
		{
			float value = float(images[0*width*height+(minX+i)*height+minY+j]);
			sum += value;
			convArea[i][j] = value;
		}
	}
	/*
	int xCorrection = x-psfRad < 0 ? 0 : psfDim-dx;
	int yCorrection = y-psfRad < 0 ? 0 : psfDim-dy;
	float sumDifference = 0.0;
	for (int i=0; i<dx; ++i)
	{
		for (int j=0; j<dy; ++j)
		{
			sumDifference += abs(convArea[i][j]/sum - psf[(i+xCorrection)*psfDim+j+yCorrection] );
		}
	}
	*/
	results[0*width*height+x*height+y] = images[0*width*height+x*height+y];//int(5000*convArea[psfRad][psfRad]);

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

	float psfSigma = argc > 1 ? atof(argv[1]) : 1.0;

	int imageCount = argc > 2 ? atoi(argv[2]) : 1;

	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix test = gen->createGaussian(psfSigma);

	gen->printPSF(test);

	FakeAsteroid *asteroid = new FakeAsteroid();



	////////////////////////////////
	fitsfile *fptr;
	/* pointer to the FITS file; defined in fitsio.h */
	int status;
	long fpixel = 1, naxis = 2, nelements;//, exposure;
	long naxes[2] = { 1024, 1024 }; // X and Y dimensions
	nelements = naxes[0] * naxes[1];
	std::stringstream ss;
	short **pixelArray = new short*[imageCount];//naxes[0]*naxes[1]];
	//short array[200][300];

	std::clock_t t1 = std::clock();

	// Create asteroid images
	for (int imageIndex=0; imageIndex<imageCount; ++imageIndex)
	{

		/* Initialize the values in the image with noisy astro */

		float kernelNorm = 1.0/test.kernel[test.dim/2*test.dim+test.dim/2];

		pixelArray[imageIndex] = new short[nelements];
		asteroid->createImage(pixelArray[imageIndex], naxes[0], naxes[1],
				0.03*float(imageIndex)+0.25, 0.04*float(imageIndex)+0.2, test, 850.0*kernelNorm, 0.5);

	}


	// Process images on GPU
	short *result = new short[nelements];
	float *gpuPsf;
	short *gpuImageSource;
	short *gpuImageResult;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuPsf, sizeof(float)*test.dim*test.dim));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuImageSource, sizeof(short)*nelements));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuImageResult, sizeof(short)*nelements));

	CUDA_CHECK_RETURN(cudaMemcpy(gpuPsf, test.kernel, sizeof(float)*test.dim*test.dim, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuImageSource, pixelArray[0], sizeof(short)*nelements, cudaMemcpyHostToDevice));

	dim3 blocks(32,32);
	dim3 threads(32,32);

	convolvePSF<<<blocks, threads>>> (naxes[0], naxes[1], imageCount, gpuImageSource,
			gpuImageResult, test.kernel, test.dim/2, test.dim); //gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(result, gpuImageResult, sizeof(short)*nelements, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuPsf));
	CUDA_CHECK_RETURN(cudaFree(gpuImageSource));
	CUDA_CHECK_RETURN(cudaFree(gpuImageResult));

	std::clock_t t2 = std::clock();

	std::cout << imageCount << " images, " <<
			1000.0*(t2 - t1)/(double) (CLOCKS_PER_SEC*imageCount) << " ms per image\n";


	// Write images to file
	for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
	{
		status = 0;
		/* initialize status before calling fitsio routines */
		ss << "Asteroid" << writeIndex+1 << ".fits";
		fits_create_file(&fptr, ss.str().c_str(), &status);
		ss.str("");
		ss.clear();
		/* create new file */
		/* Create the primary array image (16-bit short integer pixels */
		fits_create_img(fptr, SHORT_IMG, naxis, naxes, &status);
		/* Write a keyword; must pass the ADDRESS of the value */

		/* number of pixels to write */
		/* Write the array of integers to the image */
		fits_write_img(fptr, TSHORT, fpixel, nelements, result/*pixelArray[writeIndex]*/, &status);
		fits_close_file(fptr, &status);
		fits_report_error(stderr, status);
		////return( status );
	}





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

	delete[] pixelArray;

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

