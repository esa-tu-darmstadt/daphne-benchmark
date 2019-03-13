// points2image OpenCL kernel

// Structs must be known to the device
//#define MAX_NUM_WORKITEMS 32

// See datatypes.h
typedef struct Mat44 {
  /*double*/ float data[4][4];
} Mat44;

typedef struct Mat33 {
  /*double*/ float data[3][3];
} Mat33;

typedef struct Mat13 {
  /*double*/ float data[3];
} Mat13;

typedef struct Vec5 {
  /*double*/ float data[5];
} Vec5;

typedef struct Point2d {
  /*double*/ float x;
  /*double*/ float y;
} Point2d;

typedef struct ImageSize {
  int width;
  int height;
} ImageSize;

/*
typedef struct PointCloud2 {
  int    height;
  int    width;
  int    point_step;
  float* data;
} PointCloud2;

typedef struct PointsImage {
  // arrays of size image_heigt*image_width
  float* intensity;
  float* distance;
  float* min_height;
  float* max_height;
  int max_y;
  int min_y;
  int image_height;
  int image_width;
} PointsImage;
*/

//#define PRINT_KERNEL
//#define PRINT_KERNEL_CRITICAL
//#define IT_NUMBER 45285
__kernel
void
//__attribute__ ((reqd_work_group_size(MAX_NUM_WORKITEMS,1,1)))
pointcloud2_to_image(
			  //__global const PointCloud2 pointcloud2,
					 int          pointcloud2_height,
				         int          pointcloud2_width,
					 int          pointcloud2_point_step,
			  __global const float*  restrict	pointcloud2_data,
                          					Mat44     cameraExtrinsicMat,
                          					Mat33     cameraMat,
			 					Vec5      distCoeff,
                          					ImageSize imageSize,

			  // OpenCL: return data goes through global mem
			  //__global     PointsImage*  msg
			  	//__global       float*  restrict	msg_intensity,
			  	//__global       float*  restrict       msg_distance,
			  	//__global       float*  restrict       msg_min_height,
		          	//__global       float*  restrict       msg_max_height,
			  	//__global       int*    restrict       msg_scalars,

			  // Global storage for serial in-order execution part
			  __global       int*    restrict       Glob_pids,
			  __global       int*    restrict       Glob_enable_pids,
			  __global       /*double**/ float* restrict	Glob_pointdata2,
			  __global       /*double**/ float* restrict   	Glob_intensity,
			  __global       int*    restrict      	Glob_py,
			  __global       float*  restrict      	Glob_fp_2
			)
{
	// Getting width and heights
	const int w          = imageSize.width;
        const int h          = imageSize.height;
	const int pc2_width  = pointcloud2_width;
	const int pc2_height = pointcloud2_height;
	const int pc2_pstep  = pointcloud2_point_step;
/*
        // Initializing msg in global memory
	// Note single-precision float notation 0.0f
	for (unsigned int cnt = get_global_id(0);
			  cnt < w*h;
			  cnt += get_global_size(0)){
		msg_distance [cnt] = 0.0f;
	}
*/

	#if 0 //defined (PRINT_KERNEL)
	// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/workItemFunctions.html
	//printf("krnl: dimensions: %u\n", get_work_dim());	// =1
	//printf("krnl: global_size: %u\n", get_global_size(0));
        //printf("krnl: wi-global-id: %u\n", get_global_id(0));
	//printf("krnl: local_size: %u\n", get_local_size(0));
        //printf("krnl: wi-local-id: %u\n", get_local_id(0));
        //printf("krnl: num-groups: %u\n", get_num_groups(0));
	//printf("krnl: group-id: %u\n", get_group_id(0));

	if (get_local_id(0) == 0) {
		printf("\lid: %u | w : %i | h: %i | pc2_width : %i | pc2_height : %i | pc2_pstep: % i\n", 
			get_local_id(0), w, h, pc2_width, pc2_height, pc2_pstep);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#endif

// As suggested by CodePlay for 0.3.2
// Replacing __local by private + removing barrier
#if 0
        __local Mat33 invR;
	__local Mat13 invT;
	// invR= cameraExtrinsicMat(cv::Rect(0,0,3,3)).t();
	// row = get_local_id(0), if get_local_id(0) <3
	if (get_local_id(0) < 3) {
		for (int col = 0; col < 3; col++) {
		    invR.data[get_local_id(0)][col] = cameraExtrinsicMat.data[col][get_local_id(0)];
		}

		// row = get_local_id(0), if get_local_id(0) <3
		invT.data[get_local_id(0)] = 0.0;

		for (int col = 0; col < 3; col++) {
		    //invT = -invR*(cameraExtrinsicMat(cv::Rect(3,0,1,3)));
		    invT.data[get_local_id(0)] -= invR.data[get_local_id(0)][col] * cameraExtrinsicMat.data[col][3];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if 1
      Mat33 invR;
      Mat13 invT;
      // invR= cameraExtrinsicMat(cv::Rect(0,0,3,3)).t();
      // row = get_local_id(0), if get_local_id(0) <3
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          invR.data[row][col] = cameraExtrinsicMat.data[col][row];
        }

        // row = get_local_id(0), if get_local_id(0) <3
        invT.data[row] = 0.0;

        for (int col = 0; col < 3; col++) {
          // invT = -invR*(cameraExtrinsicMat(cv::Rect(3,0,1,3)));
          invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
        }
      }
      /* barrier(CLK_LOCAL_MEM_FENCE); */
#endif




        #if 0 //defined (PRINT_KERNEL)
	if (get_local_id(0) == 0) {
	        printf("(Mat33) invR: \n");
        	for (unsigned int idx = 0; idx < 3; idx ++) {
	        	for (unsigned int idy = 0; idy < 3; idy ++) {
        	        	printf("%u | %u | %f \n", idx, idy, invR.data[idx][idy]);
                	}
	        }
        	printf("\n");

	        printf("(Mat13) invT: \n");
        	for (unsigned int idx = 0; idx < 3; idx ++) {
        		printf("%u | %f \n",  idx, invT.data[idx]);
	        }
        	printf("\n");
	}

	barrier(CLK_LOCAL_MEM_FENCE);
        #endif




	// Defining a global const pointer
        __global const float* cp = (__global const float *)(pointcloud2_data);

	// Main loop. Inner loop can run in parallel (outer bound=1)
#if 0
	for (/*unsigned*/ int y = 0; y < pc2_height; ++y) {
#endif
	const int y = 0;
		/*unsigned*/ int x = (int)get_global_id(0);
		if (x < pc2_width) {
			int offset1 = (x + y*pc2_width) * pc2_pstep;

	                //__global float* fp = (__global const float *)(cp + offset1);
			float fp[5];
			for (/*unsigned*/ int k = 0; k < 5; k ++) {
				fp[k] = cp [offset1/4 + k];
			}
			/*
			double intensity = convert_double(fp[4]);
			*/
			float intensity = fp[4];

			#if 0 //defined (PRINT_KERNEL)
			//printf("%-15s %10u\n", "inner loop id (x) : ", x);
			printf("%-15s %15u\n", "offset1 : ", (x + y*pointcloud2.width) * pointcloud2.point_step);
			//printf("%s %15f\n", "intensity: ", intensity);

			#endif

			Mat13 point2;
			/*
                        point2.data[0] = convert_double(fp[0]);
                        point2.data[1] = convert_double(fp[1]);
                        point2.data[2] = convert_double(fp[2]);
			*/
                        point2.data[0] = fp[0];
                        point2.data[1] = fp[1];
                        point2.data[2] = fp[2];

                        Mat13 point;
                        //point = point * invR.t() + invT.t();
			for (int row = 0; row < 3; row++) {
			  point.data[row] = invT.data[row];
			  for (int col = 0; col < 3; col++) 
			    point.data[row] += point2.data[col] * invR.data[row][col];
			}

                        //if (point.data[2] <= 2.5) {
                        //        continue;
                        //}
                        if (point.data[2] <= 2.5) {
			    	Glob_enable_pids [x] = 0; // disabled
                        }

			else {
		                /*double*/ float tmpx = point.data[0] / point.data[2];
		                /*double*/ float tmpy = point.data[1] / point.data[2];
		                /*double*/ float r2 = tmpx * tmpx + tmpy * tmpy;
		                /*double*/ float tmpdist = 1 + distCoeff.data[0] * r2
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

		                int px = convert_int(imagepoint.x + 0.5);
		                int py = convert_int(imagepoint.y + 0.5);

				if( (0 <= px) && (px < w) && (0 <= py) && (py < h) ) {
					int pid = py * w + px;
					Glob_pids	 [x] = pid;
				    	Glob_enable_pids [x] = 1; // enabled
					Glob_pointdata2  [x] = point.data[2];
					Glob_intensity   [x] = intensity;
					Glob_py          [x] = py;
					Glob_fp_2        [x] = fp[2];
				}
				else {
				    	Glob_enable_pids [x] = 0; // disabled
				} // End: if(0 <= px && px < w && 0 <= py && py < h) {
			} // End: if (point.data[2] <= 2.5) {
               } // End: for (/*unsigned*/ int x = 0; x < pc2_width; ++x) {
#if 0
        } // End: for (/*unsigned*/ int y = 0; y < pc2_height; ++y) {
#endif
}

