/*
 * GeneratorPSF.h
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#ifndef GENERATORPSF_H_
#define GENERATORPSF_H_

#include <cmath>
#include <iostream>

struct psfTable {
	float **kernel;
	unsigned int dim;
};

class GeneratorPSF
{
	public:
		GeneratorPSF();
		psfTable createGaussian(float stdev);
		void printPSF(psfTable p);
	private:
		float gaussian(float x, float stdev);
};

#endif /* GENERATORPSF_H_ */
