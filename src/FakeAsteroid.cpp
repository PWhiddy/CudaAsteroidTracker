/*
 * FakeAsteroid.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#include "FakeAsteroid.h"

FakeAsteroid::FakeAsteroid(){
	std::uniform_int_distribution<int> noise(0,SHRT_MAX);
};

short **FakeAsteroid::createImage(int width, int height, float xpos, float ypos, psfTable psf, float noiseLevel)
{
	short** image = new short*[width];
	for (int i=0; i<width; ++i)
	{
		image[i] = new short[height];
		for (int j=0; j<height; +j)
			image[i][j] = noise(generator);
	}
	return image;
}
