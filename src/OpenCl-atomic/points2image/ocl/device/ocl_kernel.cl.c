/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
// points2image OpenCL kernel
typedef struct Mat44 {
  double data[4][4];
} Mat44;

typedef struct Mat33 {
  double data[3][3];
} Mat33;

typedef struct Mat13 {
  double data[3];
} Mat13;

typedef struct Vec5 {
  double data[5];
} Vec5;

typedef struct Point2d {
  double x;
  double y;
} Point2d;

typedef struct ImageSize {
  int width;
  int height;
} ImageSize;


__kernel void pointcloud2_to_image(
	int          pointcloud2_height,
	int          pointcloud2_width,
	int          pointcloud2_point_step,
	__global const float*  restrict	pointcloud2_data,
	Mat44     cameraExtrinsicMat,
	Mat33     cameraMat,
	Vec5      distCoeff,
	ImageSize imageSize,
	__global       int*    restrict       Glob_pids,
	__global       float* restrict	Glob_pointdata2,
	__global       float* restrict   	Glob_intensity,
	__global int* Glob_pid_no,
	__local int* Loc_pid_no,
	__local int* Loc_pid_start)
{
	// initialize local variables
	if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
		*Loc_pid_no = 0;
		*Loc_pid_start = -1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Getting width and heights
	const int w          = imageSize.width;
	const int h          = imageSize.height;
	const int pc2_width  = pointcloud2_width;
	const int pc2_height = pointcloud2_height;
	const int pc2_pstep  = pointcloud2_point_step;
	
	// build transformation matrix
	Mat33 invR;
	Mat13 invT;
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			invR.data[row][col] = cameraExtrinsicMat.data[col][row];
		}
		invT.data[row] = 0.0;
		for (int col = 0; col < 3; col++) {
			invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
		}
	}
	// cloud data pointer
	__global const float* cp = (__global const float *)(pointcloud2_data);

	// transformation for one point in the cloud
	const int y = 0;
	int x = (int)get_global_id(0);
	int px, py;
	float pointData;
	float intensity;
	int startIndex = -1;
	// discard the calculations for excess elements
	if (x < pc2_width) {
		int offset1 = (x + y*pc2_width) * pc2_pstep;
		intensity = cp[offset1/4 + 4];
		// apply the first transformation
		Mat13 point2 = {
			{
				cp[offset1/4 + 0],
				cp[offset1/4 + 1],
				cp[offset1/4 + 2]
			}
		};

		Mat13 point;
		for (int row = 0; row < 3; row++) {
			point.data[row] = invT.data[row];
			for (int col = 0; col < 3; col++) 
				point.data[row] += point2.data[col] * invR.data[row][col];
		}
		// discard elements with low depth after transformation
		if (point.data[2] <= 2.5) {
			//Glob_enable_pids [x] = 0; // disabled
		} else {
			// perspective division
			double tmpx = point.data[0] / point.data[2];
			double tmpy = point.data[1] / point.data[2];
			// apply the second transformation
			double r2 = tmpx * tmpx + tmpy * tmpy;
			double tmpdist = 1 + distCoeff.data[0] * r2
					+ distCoeff.data[1] * r2 * r2
					+ distCoeff.data[4] * r2 * r2 * r2;
			double imgx = tmpx * tmpdist
						+ 2 * distCoeff.data[2] * tmpx * tmpy
						+ distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
			double imgy = tmpy * tmpdist
						+ distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
						+ 2 * distCoeff.data[3] * tmpx * tmpy;
			// apply the third transformation
			px = (int)(cameraMat.data[0][0] * imgx + cameraMat.data[0][2] + 0.5);
			py = (int)(cameraMat.data[1][1] * imgy + cameraMat.data[1][2] + 0.5);
			// output points inside image bounds
			if( (0 <= px) && (px < w) && (0 <= py) && (py < h) ) {
				startIndex = atomic_inc(Loc_pid_no);
				pointData = point.data[2];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
		if (*Loc_pid_no > 0) {
			*Loc_pid_start = atomic_add(Glob_pid_no, *Loc_pid_no);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (startIndex > -1) {
		int pid = py*w + px;
		int iResult = *Loc_pid_start + startIndex;
		Glob_pids[iResult] = pid;
		Glob_pointdata2[iResult] = pointData;
		Glob_intensity[iResult] = intensity;
	}
}

