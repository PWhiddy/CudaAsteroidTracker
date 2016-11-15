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
		void createImage(short *image, int width, int height,
			float xpos, float ypox, psfMatrix psf, float asteroidLevel, float noiseLevel);
	private:
		std::default_random_engine generator;
		std::normal_distribution<double> noise;
		//int dice_roll = distribution(generator);  // generates number in the range 1..6
};

#endif /* FAKEASTEROID_H_ */
