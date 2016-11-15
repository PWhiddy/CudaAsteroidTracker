/*
 * FakeAsteroid.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#include "FakeAsteroid.h"

FakeAsteroid::FakeAsteroid(){
	std::normal_distribution<double> noise(1000.0, 200.0);
};

void FakeAsteroid::createImage(short *image, int width, int height,
	float xpos, float ypos, psfMatrix psf, float asteroidLevel, float noiseLevel)
{
	int xPixel = int(xpos * width);
	int yPixel = int(ypos * height);
	int psfMid = psf.dim / 2;
	for (int i=0; i<height; ++i)
	{
		int dy = i-yPixel;
		for (int j=0; j<width; ++j) {
			int dx = j-xPixel;
			float asteroid = abs(dx) < psfMid && abs(dy) < psfMid ? asteroidLevel*psf.kernel[dx+psfMid][dy+psfMid] : 0.0;
			image[i*width+j] = 100*int(std::max(noise(generator), 0.0)*noiseLevel + asteroid);
		}
	}
}
