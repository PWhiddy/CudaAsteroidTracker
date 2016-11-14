/*
 * FakeAsteroid.h
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#ifndef FAKEASTEROID_H_
#define FAKEASTEROID_H_

#include <random>
#include <fitsio.h>
#include "GeneratorPSF.h"

class FakeAsteroid
{
	public:
		FakeAsteroid();
		short **createImage(int width, int height, float xpos, float ypox, psfTable psf, float noiseLevel);
	private:
		std::default_random_engine generator;
		std::uniform_int_distribution<int> noise;
		//int dice_roll = distribution(generator);  // generates number in the range 1..6
};

#endif /* FAKEASTEROID_H_ */
