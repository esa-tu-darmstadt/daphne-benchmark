/**
Find min/max
Whole range (from which a min/max is to be found) is distributed among work-groups.
Similarly, the work-group range is distributed amoung its work-items.

https://stackoverflow.com/questions/24267280/efficiently-find-minimum-of-large-array-using-opencl
*/

// Copied from ndt_mapping/datatypes.h
// 3d euclidean point with intensity
typedef struct {
    float data[4];
} PointXYZI;

// Location of min/max within the entire range isn't relevant.
// Only min/max values matters.
// Then just use PointXYZI struct and remove loc from the kernel
/*
typedef struct {
    float data[4];
    int   loc;
} PointXYZIOut;
*/

__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
findMinMax(
           __global const PointXYZI* restrict input,       // host: target_
                          int                 input_size,  // host: target_->size()
           __global       PointXYZI* restrict gmins,       // each location stores the min of each wg
           __global       PointXYZI* restrict gmaxs        // each location stores the max of each wg
          )
{
    // Indices
    int gid   = get_global_id(0);
    int lid   = get_local_id(0);
    int wgid  = get_group_id(0);
    int lsize = get_local_size(0);

    // Storing min/max values and in local memory
    __local PointXYZI lmins [NUMWORKITEMS_PER_WORKGROUP];
    __local PointXYZI lmaxs [NUMWORKITEMS_PER_WORKGROUP];

    // Storing min/max private variables
    float mymin [3];
    float mymax [3];

    // Initializing min/max values of local and private variables
    for (char n = 0; n < 3; n++) {
        lmins [lid].data[n] = INFINITY;
        lmaxs [lid].data[n] = - INFINITY;
        mymin[n] = INFINITY;
        mymax[n] = -INFINITY;
    }
    
    // # work-groups that execute this kernel
    int num_wg = get_num_groups(0); 

    // # elements (from which a min/max will be found) assigned to each work-group
    int num_elems_per_wg = /*input_size / num_wg*/ lsize;

    // Offsets 
    int offset_wg = num_elems_per_wg * wgid; // of each work-group
    int offset_wi = offset_wg + lid;         // of each work-item within a work-group

    // Iteration upper-bound for each wg
    int upper_bound = num_elems_per_wg * (wgid + 1);

/*
    if (lid == 0) {
        printf("lid: %i | wgid: %i | #wg: %i | #elems_per_wg: %i | offset_wg: %i | offset_wi: %i | ubound: %i\n", lid, wgid, num_wg, num_elems_per_wg, offset_wg, offset_wi, upper_bound);
    }
*/
  
    // Finding min/max
    // The offset is different for each lid leading to sequential memory access
    // Understood that consecutive global memory locations are accesses in this fashion:
    // | gm0 | gm1 | gm2 | gm3 | ... | gmX | gmX+1 | gmX+2 | ... |
    // | wi0 | wi1 | wi2 | wi3 | ... | wi0 | wi1   | wi2   | ... |
    //
    // And not that consecutive global memory locations are read by a single work-item:
    // | gm0 | gm1 | gm2 | gm3 | ... | gmX | gmX+1 | gmX+2 | ... |
    // | wi0 | wi0 | wi0 | wi0 | ... | wi1 | wi1   | wi1   | ... |
    PointXYZI temp;
    if (offset_wi < input_size) {
        for (int i = offset_wi; i < upper_bound; i += lsize) {
            temp = input [i];
	    for (char n = 0; n < 3; n++) {
		if (temp.data[n] < mymin[n]) {
		   mymin[n] = temp.data[n];
		}

		if (temp.data[n] > mymax[n]) {
		   mymax[n] = temp.data[n];
		}
            }
        }
        
        // Storing the min/max found by each work-item in local memory
	for (char n = 0; n < 3; n++) {
            lmins[lid].data[n] = mymin[n];
	    lmaxs[lid].data[n] = mymax[n];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

// Disabling enclosing if-statement is OK because
// additionally-evaluated work-items do not affect functional correctness.
// This is achieved by the appropriate initialization 
// to either INFINITY for mins, or -INFINITY for maxs.
//    if (offset_wi < input_size) {
        // Finding the work-group min/max (reduces global memory accesses)
        lsize = lsize >> 1; // divided size by 2
        while (lsize > 0) {
            if (lid < lsize) {
                for (char n = 0; n < 3; n++) {
                    if (lmins[lid].data[n] > lmins[lid+lsize].data[n]) {
		        lmins[lid].data[n] = lmins[lid+lsize].data[n];
		    }

                    if (lmaxs[lid].data[n] < lmaxs[lid+lsize].data[n]) {
		        lmaxs[lid].data[n] = lmaxs[lid+lsize].data[n];
		    }
                }
            }
            lsize = lsize >> 1;
	    barrier(CLK_LOCAL_MEM_FENCE);
        }
//    }

    // Writing work-group minimum/maximum to global memory
    if (lid == 0) {
        gmins [wgid] = lmins[0];
        gmaxs [wgid] = lmaxs[0];
    }

}
