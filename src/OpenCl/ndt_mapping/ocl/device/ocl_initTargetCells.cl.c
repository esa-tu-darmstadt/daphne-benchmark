/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#if defined (DOUBLE_FP)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
  #if defined (DOUBLE_FP)
  double data[3][3];
  #else
  float data[3][3];
  #endif
} Mat33;

#if defined (DOUBLE_FP)
typedef double Vec3[3];
#else
typedef float Vec3[3];
#endif

typedef struct {
    Mat33 invCovariance;
    Vec3 mean;
    int numberPoints;
} TargetGridLeadConstPtr;

/**
 * Initializes a voxel grid.
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initTargetCells(
	__global TargetGridLeadConstPtr* restrict targetcells,
	int targetcells_size)
{
    int gid = get_global_id(0);
	if (gid < targetcells_size) {
		#if defined (DOUBLE_FP)
		TargetGridLeadConstPtr temp = {
			{{
				{0.0, 0.0, 1.0},
				{0.0, 1.0, 0.0},
				{1.0, 0.0, 0.0}
			}},
			{0.0, 0.0, 0.0}, 
			0
		};
		#else
		TargetGridLeadConstPtr temp = {
			{{
				{0.0f, 0.0f, 1.0f},
				{0.0f, 1.0f, 0.0f},
				{1.0f, 0.0f, 0.0f}
			}}, 
			{0.0f, 0.0f, 0.0f}, 
			0
		};
		#endif
		targetcells[gid] = temp;
    }
}
