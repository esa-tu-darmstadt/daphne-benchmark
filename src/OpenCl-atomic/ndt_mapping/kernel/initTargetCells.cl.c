/**
 * Author:  Leonardo Solis, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2020
 * License: Apache 2.0 (see attachached File)
 */

/**
 * Initializes a voxel grid with default values.
 * voxelGrid: voxel grid to initialize
 * targetcells_size: number of cells in the voxel grid
 */
__kernel void initTargetCells(
	__global VoxelGridInfo* restrict gridInfo,
	__global Voxel* restrict voxelGrid)
{
    int iVoxel = get_global_id(0);
	if (iVoxel < gridInfo->gridSize) {
		// initialize all members to standard values
		Voxel voxel;
		voxel.invCovariance = (VoxelCovariance){{
			{ 0.0, 0.0, 1.0 },
			{ 0.0, 1.0, 0.0 },
			{ 1.0, 0.0, 0.0 }
		}};
		voxel.mean = (VoxelMean){{ 0.0f, 0.0f, 0.0f }};
		voxel.pointListBegin = -1;
#ifdef EPHOS_VOXEL_POINT_STORAGE
		voxel.pointStorageLevel = 0;
		// point storage content may remain uninitialized
#endif
		voxelGrid[iVoxel] = voxel;
    }
}
