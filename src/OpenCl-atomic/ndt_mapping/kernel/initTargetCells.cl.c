/**
 * Initializes a voxel grid.
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 */
//__kernel
//void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
//initTargetCells(
__kernel void initTargetCells(
	__global Voxel* restrict targetcells,
	int targetcells_size)
{
    int iVoxel = get_global_id(0);
	if (iVoxel < targetcells_size) {
		// initialize all members to standard values
		Voxel voxel;
		voxel.invCovariance = (Mat33){{
			{ 0.0, 0.0, 1.0 },
			{ 0.0, 1.0, 0.0 },
			{ 1.0, 0.0, 0.0 }
		}};
		voxel.mean[0] = 0;
		voxel.mean[1] = 0;
		voxel.mean[2] = 0;
		voxel.pointListBegin = -1;
#ifdef EPHOS_VOXEL_POINT_STORAGE
		voxel.pointStorageLevel = 0;
		// point storage content may remain uninitialized
#endif
// 		= {
// 			{{
// 				{0.0, 0.0, 1.0},
// 				{0.0, 1.0, 0.0},
// 				{1.0, 0.0, 0.0}
// 			}},
// 			{0.0, 0.0, 0.0},
// 			-1
// 		};
		targetcells[iVoxel] = voxel;
    }
}
