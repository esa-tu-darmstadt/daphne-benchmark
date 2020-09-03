/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attached files)
 */
#ifndef EPHOS_DATATYPES_H
#define EPHOS_DATATYPES_H

typedef struct TransformInfo {
	//double cameraExtrinsics[4][4];
	double initRotation[3][3];
	double initTranslation[3];
	double imageScale[2];
	double imageOffset[2];
	double distCoeff[5];
	int imageSize[2];
	int cloudPointNo;
	int cloudPointStep;
} TransformInfo;
//
typedef struct PixelData {
	int position[2];
	float depth;
	float intensity;
} PixelData;

#endif // EPHOS_DATATYPES_H
