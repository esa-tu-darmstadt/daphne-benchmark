/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attached files)
 */
#include <iostream>
#include <fstream>

#include "points2image.h"

/**
 * Reads the next point cloud
 */
void  points2image::parsePointCloud(std::ifstream& input_file, PointCloud& pointcloud) {
	try {
		input_file.read((char*)&pointcloud.height, sizeof(int32_t));
		input_file.read((char*)&pointcloud.width, sizeof(int32_t));
		input_file.read((char*)&pointcloud.point_step, sizeof(uint32_t));

		cudaMallocManaged(&pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
		input_file.read((char*)pointcloud.data, pointcloud.height*pointcloud.width*pointcloud.point_step);
    }  catch (std::ifstream::failure) {
		throw std::ios_base::failure("Error reading the next point cloud.");
    }
}

void points2image::cleanupTestcases(int count) {
	for (int i = 0; i < count; i++) {
		cudaFree(pointcloud[i].data);
		// avoid double free in framework code
		pointcloud[i].data = nullptr;
	}
	points2image_base::cleanupTestcases(count);
}
void points2image::init() {
	std::cout << "init\n";
	points2image_base::init();
	std::cout << "done\n" << std::endl;
}
void points2image::quit() {
	points2image_base::quit();
}


// set the external kernel instance used in main()
points2image a;
benchmark& myKernel = a;