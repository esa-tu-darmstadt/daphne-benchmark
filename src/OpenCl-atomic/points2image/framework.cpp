/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019 - 2020
 * License: Apache 2.0 (see attachached File)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <string>

#include "points2image.h"

void  points2image::parsePointCloud(std::ifstream& input_file, PointCloud& pointcloud) {
	try {
		input_file.read((char*)&pointcloud.height, sizeof(int32_t));
		input_file.read((char*)&pointcloud.width, sizeof(int32_t));
		input_file.read((char*)&pointcloud.point_step, sizeof(uint32_t));

		// TODO maybe start timer for this call
		prepare_compute_buffers(pointcloud);

		input_file.read((char*)pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}
void points2image::cleanupTestcases(int count) {
	for (int i = 0; i < count; i++) {
		delete[] results[i].intensity;
		delete[] results[i].distance;
		delete[] results[i].min_height;
		delete[] results[i].max_height;
#ifndef EPHOS_PINNED_MEMORY
		delete[] pointcloud[i].data;
#endif
	}
	results.clear();
	cameraExtrinsicMat.clear();
	cameraMat.clear();
	distCoeff.clear();
	imageSize.clear();
	pointcloud.clear();
}