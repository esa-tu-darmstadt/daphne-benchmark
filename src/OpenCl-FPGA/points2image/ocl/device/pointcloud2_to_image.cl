typedef struct Mat13 {
    float data[3];
} Mat13;

typedef struct Vec5 {
    float data[5];
} Vec5;

typedef struct Mat33 {
    float data[3][3];
} Mat33;

typedef struct Point2d {
    float x;
    float y;
} Point2d;

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
pointcloud2_to_image(
__global const float* restrict pointcloud2_data,
__global       float* restrict msg_intensity,
__global       float* restrict msg_distance,
__global       float* restrict msg_min_height,
__global       float* restrict msg_max_height,
__global       int*   restrict msg_max_y_min_y,
               uint            pointcloud2_height,
               uint            pointcloud2_width,
               uint            pointcloud2_point_step,
               uint            width_times_step,
               Mat13           invT,
               Mat33           invR,
	       Vec5            distCoeff,
               Mat33           cameraMat,
               uint            w,
               uint            h,
               int             msg_max_y,
	       int             msg_min_y
)
{
    int local_msg_max_y = msg_max_y;
    int local_msg_min_y = msg_min_y;

    // Defining a global const pointer
    __global const float* cp = (__global const float *)(pointcloud2_data);

    // Main loop
    __attribute__((xcl_pipeline_loop))
        loop_main_for_x:
	for (uint x = 0; x < pointcloud2_width; x++) {

            //__global const float* fp = cp + (x * pointcloud2_point_step);
	    const int offset1 = x * pointcloud2_point_step;
	    float fp[5];

/*
            loop_cp:
	    for (uchar k = 0; k < 5; k ++) {
		fp[k] = cp [offset1/4 + k];
	    }
*/
	    fp[0] = cp [offset1/4];
            fp[1] = cp [offset1/4 + 1];
            fp[2] = cp [offset1/4 + 2];
            fp[3] = cp [offset1/4 + 3];
            fp[4] = cp [offset1/4 + 4];

	    float intensity = fp[4];

	    Mat13 point, point2;
	    point2.data[0] = fp[0]; // float(fp[0]);
	    point2.data[1] = fp[1]; // float(fp[1]);
	    point2.data[2] = fp[2]; // float(fp[2]);

            // Originally-nested loop is replaced with fully-unrolled code.
            // Nested loop produced error during FPGA build with sdx182, but not with sdx174.
            point.data[0] = invT.data[0] + point2.data[0] * invR.data[0][0] + 
                                           point2.data[1] * invR.data[0][1] + 
                                           point2.data[2] * invR.data[0][2];

	    point.data[1] = invT.data[1] + point2.data[0] * invR.data[1][0] + 
                                           point2.data[1] * invR.data[1][1] + 
                                           point2.data[2] * invR.data[1][2];

	    point.data[2] = invT.data[2] + point2.data[0] * invR.data[2][0] + 
                                           point2.data[1] * invR.data[2][1] + 
                                           point2.data[2] * invR.data[2][2];
	            
	    if (point.data[2] <= 2.5) {
		// Not supported in OpenCL, therefore else {} is added
		/*continue;*/
	    }
            else {
	        float tmpx = point.data[0] / point.data[2];
	        float tmpy = point.data[1]/ point.data[2];
	        float r2 = tmpx * tmpx + tmpy * tmpy;
	        float tmpdist = 1 + distCoeff.data[0] * r2
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

	        int px = imagepoint.x + 0.5; // int(imagepoint.x + 0.5);
	        int py = imagepoint.y + 0.5; // int(imagepoint.y + 0.5);
	        
                if(0 <= px && px < w && 0 <= py && py < h) {

		    int pid = py * w + px;

                    float msgdistancepid = msg_distance[pid];
                    float pd2times100    = point.data[2] * 100.0;
                    
                    bool  is_msgdistancepid_eq_zero          = (msgdistancepid == 0.0);
                    bool  is_msgdistancepid_gteq_pd2times100 = (msgdistancepid >= pd2times100);

                    bool  is_msgdistancepid_eq_pd2times100   = (msgdistancepid == pd2times100);
                    bool  is_msgdistancepid_gt_pd2times100   = (msgdistancepid >  pd2times100);

		    if (is_msgdistancepid_eq_zero || 
                        is_msgdistancepid_gteq_pd2times100) {
	                // added to make the result always deterministic and independent from the point order
		        // in case two points get the same distance, take the one with high intensity
                        bool cmp_intensity = (msg_intensity[pid] < intensity);

                        if ((is_msgdistancepid_eq_pd2times100  && cmp_intensity) ||
                           is_msgdistancepid_gt_pd2times100 ||
                           is_msgdistancepid_eq_zero) {
                            msg_intensity[pid] = intensity;
                        }
                    
			msg_distance[pid] = pd2times100;

         		local_msg_max_y = py > local_msg_max_y ? py : local_msg_max_y;
			local_msg_min_y = py < local_msg_min_y ? py : local_msg_min_y;
                    }

                    if (pointcloud2_height == 2) {
                        //process simultaneously min and max during the first layer
                        //__global const float* fp2 = (cp + (x + (y+1)*pointcloud2_width) * pointcloud2_point_step);
                        __global const float* fp2 = cp + offset1 + width_times_step;
			msg_min_height[pid] = fp[2];
			msg_max_height[pid] = fp2[2];
		    }
		    else
		    {
			msg_min_height[pid] = -1.25;
			msg_max_height[pid] = 0;
		    }
		} // End of if(0 <= px && px < w && 0 <= py && py < h)
            } // End of if (point.data[2] <= 2.5)
	} // End of for (uint x = 0; x < pointcloud2_width; ++x)

    msg_max_y_min_y [0] = local_msg_max_y;
    msg_max_y_min_y [1] = local_msg_min_y;

    return;
}
