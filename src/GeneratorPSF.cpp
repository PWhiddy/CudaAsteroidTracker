/*
 * GeneratorPSF.cpp
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#define MAX_KERNEL_RADIUS 7
#define INTEGRATION_STEPS 100000

#include "GeneratorPSF.h"

GeneratorPSF::GeneratorPSF(){}

psfTable GeneratorPSF::createGaussian(float stdev)
{

	float simpleGauss[MAX_KERNEL_RADIUS];
	double psfCoverage = 0.0;
	double normFactor = stdev*sqrt(2.0);
	int i = 0;

	while (psfCoverage < 0.98 && i < MAX_KERNEL_RADIUS)
	{
		float currentBin = 0.5*(std::erf( (float(i)+0.5)/normFactor ) - std::erf( (float(i)-0.5)/normFactor ) );
		simpleGauss[i] = currentBin;
		if (i == 0)
		{
			psfCoverage += currentBin;
		}
		else {
			psfCoverage += 2.0*currentBin;
		}
		i++;
	}

	std::cout << "Using Kernel Size " << 2*i-1 << "X" << 2*i-1 << "\n";

	psfTable p;
	p.dim = 2*i-1;
	p.kernel = new float*[p.dim];
	for (int ii=0; ii<p.dim; ++ii)
	{
		p.kernel[ii] = new float[p.dim];
		for (int jj=0; jj<p.dim; ++jj)
			p.kernel[ii][jj] = simpleGauss[abs(i-ii-1)]*simpleGauss[abs(i-jj-1)];
	}

	return p;
}

float GeneratorPSF::gaussian(float x, float twoSigSquare)
{
	return exp(-x*x/(twoSigSquare));
}

void GeneratorPSF::printPSF(psfTable p)
{
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    std::cout.precision(3);
    float sum = 0.0;
	for (int row=0; row<p.dim; ++row)
	{
		std::cout << "| ";
		for (int col=0; col<p.dim; ++col)
			{
				sum += p.kernel[row][col];
				std::cout << p.kernel[row][col] << " | ";
			}
		std::cout << "\n ";
	    for (int space=0; space<p.dim*8-1; ++space)
	    	std::cout << "-";
		std::cout << "\n";
	}
	std::cout << 100.0*sum << "% of PSF contained within kernel\n";
}