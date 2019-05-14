#include "benchmark.h"
#include "datatypes.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <omp.h>
//#include "../include/meassurement_AverageOnly.h"

#define MAX_EPS 0.001

/**
   Author: Florian Stock 2018

   Kernel extracted from Autoware suite.
   Dependencies on the PCL (PointCloudLib) and CV (OpenCV) libs are removed.
   For their licenses see license folder.

   Kernel uses 2500 invocations of the pointcloud2_to_image function from the
   points2image-package/node
   (see Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp)

   Computed results are compared with the Autoware computed result.

 */

class points2image : public kernel {
private:
        // the number of testcases read
        int read_testcases = 0;
        // testcase and reference data streams
        std::ifstream input_file, output_file;
        // whether critical deviation from the reference data has been detected
        bool error_so_far = false;
        // deviation from the reference data
        double max_delta = 0.0;
        // the point clouds to process in one iteration
        PointCloud2* pointcloud2 = nullptr;
        // the associated camera extrinsic matrices
        Mat44* cameraExtrinsicMat = nullptr;
        // the associated camera intrinsic matrices
        Mat33* cameraMat = nullptr;
        // distance coefficients for the current iteration
        Vec5* distCoeff = nullptr;
        // image sizes for the current iteration
        ImageSize* imageSize = nullptr;
        // Algorithm results for the current iteration
        PointsImage* results = nullptr;
public:
        /*
         * Initializes the kernel. Must be called before run().
         */
        virtual void init();
        /**
         * Performs the kernel operations on all input and output data.
         * p: number of testcases to process in one step
         */
        virtual void run(int p = 1);
        /**
         * Finally checks whether all input data has been processed successfully.
         */
        virtual bool check_output();

protected:
        /**
        * Reads the next test cases.
        * count: the number of testcases to read
        * returns: the number of testcases actually read
        */
        virtual int read_next_testcases(int count);
        /**
         * Compares the results from the algorithm with the reference data.
         * count: the number of testcases processed 
         */
        virtual void check_next_outputs(int count);
        /**
         * Reads the number of testcases in the data set.
         */
        int read_number_testcases(std::ifstream& input_file);

};

/*class points2image : public kernel {
public:
  virtual void init();
  virtual void run(int p = 1);
  virtual bool check_output();
  PointCloud2* pointcloud2 = NULL;
  Mat44* cameraExtrinsicMat = NULL;
  Mat33* cameraMat = NULL;
  Vec5* distCoeff = NULL;
  ImageSize* imageSize = NULL;
  PointsImage* results = NULL;
protected:
  virtual int read_next_testcases(int count);
  virtual void check_next_outputs(int count);
  int read_testcases = 0;
  std::ifstream input_file, output_file;
  bool error_so_far;
  double max_delta;
};*/


/**
 * Parses the next point cloud from the input stream.
 */
void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
        try {
                input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
                input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
                input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
                pointcloud2->data = new float[pointcloud2->height * pointcloud2->width * pointcloud2->point_step];
                input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
    }  catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the next point cloud.");
    }
}
/**
 * Parses the next camera extrinsic matrix.
 */
void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
        try {
                for (int h = 0; h < 4; h++)
                        for (int w = 0; w < 4; w++)
                                input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
        } catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the next extrinsic matrix.");
        }
}
/**
 * Parses the next camera matrix.
 */
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
        try {
        for (int h = 0; h < 3; h++)
                for (int w = 0; w < 3; w++)
                        input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
        } catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the next camera matrix.");
    }
}
/**
 * Parses the next distance coefficients.
 */
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
        try {
                for (int w = 0; w < 5; w++)
                        input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
        } catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the next set of distance coefficients.");
        }
}
/**
 * Parses the next image sizes.
 */
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
        try {
                input_file.read((char*)&(imageSize->width), sizeof(int32_t));
                input_file.read((char*)&(imageSize->height), sizeof(int32_t));
        } catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the next image size.");
        }
}
/**
 * Parses the next reference image.
 */
void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
        try {
                // read data of static size
                output_file.read((char*)&(goldenResult->image_width), sizeof(int32_t));
                output_file.read((char*)&(goldenResult->image_height), sizeof(int32_t));
                output_file.read((char*)&(goldenResult->max_y), sizeof(int32_t));
                output_file.read((char*)&(goldenResult->min_y), sizeof(int32_t));
                int pos = 0;
                int elements = goldenResult->image_height * goldenResult->image_width;
                goldenResult->intensity = new float[elements];
                goldenResult->distance = new float[elements];
                goldenResult->min_height = new float[elements];
                goldenResult->max_height = new float[elements];
                // read data of variable size
                for (int h = 0; h < goldenResult->image_height; h++)
                        for (int w = 0; w < goldenResult->image_width; w++)
                        {
                                output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
                                output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
                                output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
                                output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
                                pos++;
                        }
        } catch (std::ios_base::failure) {
                throw std::ios_base::failure("Error reading the next reference image.");
        }
}

int points2image::read_next_testcases(int count)
{
        // free the memory that has been allocated in the previous iteration
        // and allocate new for the currently required data sizes
        if (pointcloud2) 
                for (int m = 0; m < count; ++m)
                        delete [] pointcloud2[m].data;
        delete [] pointcloud2;
        pointcloud2 = new PointCloud2[count];
        delete [] cameraExtrinsicMat;
        cameraExtrinsicMat = new Mat44[count];
        delete [] cameraMat;
        cameraMat = new Mat33[count];
        delete [] distCoeff;
        distCoeff = new Vec5[count];
        delete [] imageSize;
        imageSize = new ImageSize[count];
        if (results)
        for (int m = 0; m < count; ++m)
        {
                delete [] results[m].intensity;
                delete [] results[m].distance;
                delete [] results[m].min_height;
                delete [] results[m].max_height;
        }
        delete [] results;
        results = new PointsImage[count];

        // iteratively read the data for the test cases
        int i;
        for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
        {
                try {
                        parsePointCloud(input_file, pointcloud2 + i);
                        parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
                        parseCameraMat(input_file, cameraMat + i);
                        parseDistCoeff(input_file, distCoeff + i);
                        parseImageSize(input_file, imageSize + i);
                } catch (std::ios_base::failure& e) {
                        std::cerr << e.what() << std::endl;
                        exit(-3);
                }
        }
        return i;
}
int points2image::read_number_testcases(std::ifstream& input_file)
{
        // reads the number of testcases in the data stream
        int32_t number;
        try {
                input_file.read((char*)&(number), sizeof(int32_t));
        } catch (std::ifstream::failure) {
                throw std::ios_base::failure("Error reading the number of testcases.");
        }

        return number;
}

void points2image::init() {
        std::cout << "init\n";

        // open testcase and reference data streams
        input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
        output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
        try {
                input_file.open("../../../data/p2i_input.dat", std::ios::binary);
        } catch (std::ifstream::failure) {
                std::cerr << "Error opening the input data file" << std::endl;
                exit(-2);
        }
        try {
                output_file.open("../../../data/p2i_output.dat", std::ios::binary);
        } catch (std::ifstream::failure) {
                std::cerr << "Error opening the output data file" << std::endl;
                exit(-2);
        }
        try {
        // consume the total number of testcases
                testcases = read_number_testcases(input_file);
        } catch (std::ios_base::failure& e) {
                std::cerr << e.what() << std::endl;
                exit(-3);
        }

        // prepare the first iteration
        error_so_far = false;
        max_delta = 0.0;
        pointcloud2 = nullptr;
        cameraExtrinsicMat = nullptr;
        cameraMat = nullptr;
        distCoeff = nullptr;
        imageSize = nullptr;
        results = nullptr;

        std::cout << "done\n" << std::endl;
}

/**
    Helperfunction which allocates and sets everything to a given value.
*/
float* assign(uint32_t count, float value) {
  float* result;

  result = new float[count];
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    result[i] = value;
  }
  return result;
}

/*
 *    This code is extracted from Autoware, file:
   ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
*/

/**
   This code is extracted from Autoware, file:
   ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
*/
PointsImage
pointcloud2_to_image(const PointCloud2& pointcloud2,
                     const Mat44& cameraExtrinsicMat,
                     const Mat33& cameraMat, const Vec5& distCoeff,
                     const ImageSize& imageSize)
{
        int w = imageSize.width;
        int h = imageSize.height;
        float* cp = (float *)pointcloud2.data;

        PointsImage msg;
        msg.intensity = assign(w * h, 0);
        msg.distance = assign(w * h, 0);
        msg.min_height = assign(w * h, 0);
        msg.max_height = assign(w * h, 0);
        msg.max_y = -1;
        msg.min_y = h;
        msg.image_height = imageSize.height;
        msg.image_width = imageSize.width;

        Mat33 invR;
        Mat13 invT;
        for (int row = 0; row < 3; row++)
          for (int col = 0; col < 3; col++)
            invR.data[row][col] = cameraExtrinsicMat.data[col][row];
        for (int row = 0; row < 3; row++) {
          invT.data[row] = 0.0;
          for (int col = 0; col < 3; col++)
            //invT = -invR*(cameraExtrinsicMat(cv::Rect(3,0,1,3)));
            invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
        }

        int sizeMat = pointcloud2.width * pointcloud2.height;
        int sizeMaxCp = pointcloud2.height * pointcloud2.width * pointcloud2.point_step;
        double pointValue2Times100Array[sizeMat];
        Point2d imagePointArray[sizeMat];
        int pCHeight = pointcloud2.height;
        int pCWidth = pointcloud2.width;
        int pCStepsize = pointcloud2.point_step;

        #pragma omp target map(from:pointValue2Times100Array[:sizeMat],imagePointArray[:sizeMat]) map(to:cp[:sizeMaxCp],distCoeff,cameraMat,invT,invR,pCHeight,pCWidth,pCStepsize)
        #pragma omp teams distribute parallel for collapse(2)
            for (uint32_t x = 0; x < pCWidth; ++x) {
              for (uint32_t y = 0; y < pCHeight; ++y) {
                    int indexMat =x + y * pCWidth;
                    float* fp = (float *)(((uintptr_t)cp) + (x + y*pCWidth) * pCStepsize);

                    double intensity = fp[4]; //private

                    Mat13 point, point2; //private
                    point2.data[0] = double(fp[0]);
                    point2.data[1] = double(fp[1]);
                    point2.data[2] = double(fp[2]);

                    for (int row = 0; row < 3; row++) {
                      point.data[row] = invT.data[row];
                      for (int col = 0; col < 3; col++)
                        point.data[row] += point2.data[col] * invR.data[row][col];
                    }
                    pointValue2Times100Array[indexMat] = point.data[2] * 100.0;
                    if (point.data[2] <= 2.5) {
                      Point2d imagepointError;
                      imagepointError.x = -1;
                      imagepointError.y = -1;
                      imagePointArray[indexMat] = imagepointError;
                            continue;
                    }

                    double tmpx = point.data[0] / point.data[2];
                    double tmpy = point.data[1]/ point.data[2];
                    double r2 = tmpx * tmpx + tmpy * tmpy;
                    double tmpdist = 1 + distCoeff.data[0] * r2
                            + distCoeff.data[1] * r2 * r2
                            + distCoeff.data[4] * r2 * r2 * r2;

                    Point2d imagepoint;
                    imagepoint.x = tmpx * tmpdist
                            + 2 * distCoeff.data[2] * tmpx * tmpy
                            + distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
                    imagepoint.y = tmpy * tmpdist
                            + distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
                            + 2 * distCoeff.data[3] * tmpx * tmpy;
                    imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
                    imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];

                    imagePointArray[indexMat] = imagepoint;
                 }
               }




         for (uint32_t x = 0; x < pointcloud2.width; ++x) {
             for (uint32_t y = 0; y < pointcloud2.height; ++y) {
               int indexMat =x + y * pCWidth;
              //restore values
              double p2Times100 = pointValue2Times100Array[indexMat];


              if (p2Times100 <= (2.5 * 100.0)) {
                      continue;
              }
              float* fp = (float *)(((uintptr_t)cp) + (x + y*pointcloud2.width) * pointcloud2.point_step);
              double intensity = fp[4]; //private
              Point2d imagepoint = imagePointArray[indexMat];
              int px = int(imagepoint.x + 0.5); // runde ab 0.5 auf ansonsten ab
              int py = int(imagepoint.y + 0.5);
              if(0 <= px && px < w && 0 <= py && py < h)
              {
                int pid = py * w + px;
                if(msg.distance[pid] == 0 || msg.distance[pid] > p2Times100)
                  {
                    msg.distance[pid] = float(p2Times100); //msg is das problem beim paralelisieren
                    msg.intensity[pid] = float(intensity);
                    msg.max_y = py > msg.max_y ? py : msg.max_y;
                    msg.min_y = py < msg.min_y ? py : msg.min_y;
                  }
                  if (0 == y && pointcloud2.height == 2)
                  {
                    float* fp2 = (float *)(cp + (x + (y+1)*pointcloud2.width) * pointcloud2.point_step);
                    msg.min_height[pid] = fp[2];
                    msg.max_height[pid] = fp2[2];
                  }
                  else
                  {
                    msg.min_height[pid] = -1.25;
                    msg.max_height[pid] = 0;
                  }
                }

            }

          }
        return msg;
}



void points2image::run(int p) {
  pause_func();
  //std::cout << "Define number of threads" << '\n';
  //int corenum;
  //std::cin >> corenum;
  //omp_set_num_threads(corenum);
  //  std::string a[] = {"init","Mainloop","initLoop"};
  //  init_timer(a,3);
  std::cout << "Processing Testcases"<< '\n';
  while (read_testcases < testcases)
    {
      int count = read_next_testcases(p);
      unpause_func();
      for (int i = 0; i < count; i++)
        {
          // actual kernel invocation
          results[i] = pointcloud2_to_image(pointcloud2[i],
                                            cameraExtrinsicMat[i],
                                            cameraMat[i], distCoeff[i],
                                            imageSize[i]);
        }
      pause_func();
      check_next_outputs(count);
    }
    //std::cout << "Processing Done"<< '\n';
    //print_timer;
}

void points2image::check_next_outputs(int count)
{
  PointsImage reference;

  for (int i = 0; i < count; i++)
    {
      parsePointsImage(output_file, &reference);
      if ((results[i].image_height != reference.image_height)
          || (results[i].image_width != reference.image_width))
        {
          error_so_far = true;
        }
      if ((results[i].min_y != reference.min_y)
          || (results[i].max_y != reference.max_y))
        {
          error_so_far = true;
        }

      int pos = 0;
      for (int h = 0; h < reference.image_height; h++)
        for (int w = 0; w < reference.image_width; w++)
          {
            if (fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
              max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
            if (fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
              max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
            if (fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
              max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
            if (fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
              max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
            pos++;
          }

      delete [] reference.intensity;
      delete [] reference.distance;
      delete [] reference.min_height;
      delete [] reference.max_height;
    }
}

bool points2image::check_output() {
    std::cout << "checking output \n";

    input_file.close();
    output_file.close();

    std::cout << "max delta: " << max_delta << "\n";
    if ((max_delta > MAX_EPS) || error_so_far)
          return false;
    return true;
}

points2image a = points2image();
kernel& myKernel = a;
