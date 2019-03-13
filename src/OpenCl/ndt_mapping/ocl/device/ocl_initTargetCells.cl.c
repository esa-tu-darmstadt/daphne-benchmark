/**
Init targetcells elements
*/

#if defined (DOUBLE_FP)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Copied from ndt_mapping/datatypes.h
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

// typedef std::vector<TargetGridLeafConstPtr> VoxelGrid;
typedef struct {
    Mat33 invCovariance;
    Vec3 mean;
    int numberPoints;
    /*
    int voxel_x;
    int voxel_y;
    int voxel_z;
    */
} TargetGridLeadConstPtr;

// Using local memory: GPU utilization 20%

// Using only private memory: GPU utilization 100%
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initTargetCells(
                __global       TargetGridLeadConstPtr* restrict targetcells,
                               int                              targetcells_size
               )
{
    // Indices
    int gid = get_global_id(0);

    if (gid < targetcells_size) {
        /*
        TargetGridLeadConstPtr temp;

        temp.numberPoints = 0;
        temp.mean[0] = 0;
        temp.mean[1] = 0;
        temp.mean[2] = 0;
	
        for (int i=0;i<3;i++) {
	    //temp.mean[i] = 0;

            for (int j=0;j<3;j++) {
                temp.invCovariance.data[i][j] = 0.0;
            }
        }

        temp.invCovariance.data[2][0] = 1.0;
        temp.invCovariance.data[1][1] = 1.0;
        temp.invCovariance.data[0][2] = 1.0;
        */

        #if defined (DOUBLE_FP)
	TargetGridLeadConstPtr temp = {{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0 /*, 0, 0, 0*/};
        #else
	TargetGridLeadConstPtr temp = {{0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0 /*, 0, 0, 0*/};
        #endif

        // Copying local contents into global memory
        targetcells[gid] = temp;
    }
}
