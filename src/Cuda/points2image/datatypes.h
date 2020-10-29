/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_DATATYPES_H
#define EPHOS_DATATYPES_H

#include <vector>

#include <common/datatypes_base.h>


typedef struct TransformationInfo {
	//Mat33 cameraMat;
	Mat33 invRotation;
	Vec5 distCoeff;
	Mat13 invTranslation;
	Vec2 camOffset;
	Vec2 camScale;
	int pointNo;
	int pointStep;
	ImageSize imageSize;
} TransformationInfo;

typedef struct TransformPixel {
	int posX;
	int posY;
	float depth;
	float intensity;
} TransformPixel;

#endif

