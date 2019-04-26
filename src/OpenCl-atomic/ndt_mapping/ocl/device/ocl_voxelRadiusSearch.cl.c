#ifndef RESOUTION
#define RESOLUTION 1
#endif
#ifndef RADIUS
#define RADIUS 1
#endif

__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
radiusSearch(
	__global PointXYZI* restrict input,
	__global Voxel* restrict target_cells,
	__global PointVoxel* result,
	__global int* resultNo,
	__local int* l_startIndex,
	__local int* l_resultNo,
	int pointNo,
	int minVoxel_0,
	int minVoxel_1,
	int minVoxel_2,
	int voxelDimension_0,
	int voxelDimension_1) {

}