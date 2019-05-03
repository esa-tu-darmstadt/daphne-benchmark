/**
 * Initializes a voxel grid.
 * targetcells: voxel grid
 * targetcells_size: number of cells in the voxel grid
 */
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
initTargetCells(
	__global Voxel* restrict targetcells,
	int targetcells_size)
{
    int gid = get_global_id(0);
	if (gid < targetcells_size) {
		#if defined (DOUBLE_FP)
		Voxel temp = {
			{{
				{0.0, 0.0, 1.0},
				{0.0, 1.0, 0.0},
				{1.0, 0.0, 0.0}
			}},
			{0.0, 0.0, 0.0}, 
			-1
		};
		#else
		Voxel temp = {
			{{
				{0.0f, 0.0f, 1.0f},
				{0.0f, 1.0f, 0.0f},
				{1.0f, 0.0f, 0.0f}
			}}, 
			{0.0f, 0.0f, 0.0f}, 
			-1
		};
		#endif
		targetcells[gid] = temp;
    }
}
