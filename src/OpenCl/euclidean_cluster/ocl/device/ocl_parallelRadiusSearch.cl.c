/** 
    own radiusSearch, just goes linear through the array of distances
    returns number of found points

    Performs multiple searches in parallel, for all points belonging to point_index
    In indices each value is set to 1 in case it is result of the search, 0 otherwise
*/
    
__kernel
void __attribute__ ((reqd_work_group_size(NUMWORKITEMS_PER_WORKGROUP,1,1)))
parallelRadiusSearch(
__global const int*  restrict point_index,
__global       bool* restrict indices, 
__global const bool* restrict sqr_distances, 
int start_index, 
int search_points, 
int cloud_size)
{

    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    int id = get_global_id(0);

    if (id >= cloud_size) {
      // Not supported in OpenCL
      //return;
    }
    else {
      bool found = false;
      bool is_skipped = false;
      for (int search_point_index = start_index; search_point_index < search_points; search_point_index++)
        {
	  if (id == point_index[search_point_index])
	    {
	      found = true;
              // Not supported in OpenCL 
	      //continue;
              is_skipped = true;
	    }
          else {
            is_skipped = false;
          }

          if (is_skipped == false) {
	    int array_index = point_index[search_point_index] * cloud_size+id;;
	    if ( sqr_distances[array_index])
	      found = true;
          }
        }
      indices[id] = found;
    }
}


