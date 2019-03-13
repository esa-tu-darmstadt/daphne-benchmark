#include "benchmark.h"
#include <iostream>
#include <fstream>
#include "datatypes.h"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#if defined (OPENCL_EPHOS)
#include "ocl_ephos.h"
#include "stringify.h"

// Stringify preprocessor directive
// to pass numeric value of local-size 
// from Makefile to kernel code
// https://stackoverflow.com/a/240361/1616865
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define NUMWORKITEMS_PER_WORKGROUP_STRING STRINGIZE(NUMWORKITEMS_PER_WORKGROUP) 

/*
OCL_Struct OCL_objs;
*/
#include <cstring>

#endif

// constants, defined in the original code as parameters

const int _cluster_size_min = 20;
const int _cluster_size_max = 100000;
const bool _pose_estimation = true;

#define MAX_EPS 0.001

class euclidean_clustering : public kernel {
public:
    virtual void init();
    virtual void run(int p = 1);
    virtual bool check_output();
protected:
    void clusterAndColor(
                    #if defined (OPENCL_EPHOS)
                    OCL_Struct* OCL_objs,
                    #endif
                    /*const PointCloud *in_cloud_ptr,*/
  		    const PointCloud in_cloud_ptr,
                    int cloud_size,
		    PointCloudRGB *out_cloud_ptr,
		    BoundingboxArray *in_out_boundingbox_array,
		    Centroid *in_out_centroids,
		    #if defined (DOUBLE_FP)
		    double in_max_cluster_distance
		    #else
                    float in_max_cluster_distance
		    #endif
                         );
    void segmentByDistance(
                           #if defined (OPENCL_EPHOS)
                           OCL_Struct* OCL_objs,
                           #endif
                           /*const PointCloud *in_cloud_ptr,*/
                           const PointCloud in_cloud_ptr,
			   int cloud_size,
			   PointCloudRGB *out_cloud_ptr,
			   BoundingboxArray *in_out_boundingbox_array,
			   Centroid *in_out_centroids,
		           #if defined (DOUBLE_FP)
			   double in_max_cluster_distance
	                   #else
                           float in_max_cluster_distance
                           #endif
                           );
    virtual int read_next_testcases(int count);
    virtual void check_next_outputs(int count);
    int read_number_testcases(std::ifstream& input_file);
    PointCloud *in_cloud_ptr;
    int *cloud_size;
    PointCloudRGB *out_cloud_ptr;
    BoundingboxArray *out_boundingbox_array;
    Centroid *out_centroids;
    int read_testcases = 0;
    std::ifstream input_file, output_file;
    bool error_so_far;
    double max_delta;
};

int euclidean_clustering::read_number_testcases(std::ifstream& input_file)
{
    int32_t number;
    try {
	input_file.read((char*)&(number), sizeof(int32_t));
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }

    return number;    
}


// helper function to get the min area enclosing rectangle
static void rotatingCalipers( const Point2D* points, int n, float* out )
{
    float minarea = std::numeric_limits<float>::max();
    float max_dist = 0;
    char buffer[32] = {};
    int i, k;
    //    AutoBuffer<float> abuf(n*3);
    float* abuf = (float*)alloca(n * 3 * sizeof(float));
    float* inv_vect_length = abuf;
    Point2D* vect = (Point2D*)(inv_vect_length + n);
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = { -1, -1, -1, -1 };

    /* rotating calipers sides will always have coordinates
     (a,b) (-b,a) (-a,-b) (b, -a)
     */
    /* this is a first base bector (a,b) initialized by (1,0) */
    float orientation = 0;
    float base_a;
    float base_b = 0;

    float left_x, right_x, top_y, bottom_y;
    Point2D pt0 = points[0];

    left_x = right_x = pt0.x;
    top_y = bottom_y = pt0.y;

    for( i = 0; i < n; i++ )
    {
        #if defined (DOUBLE_FP)
        double dx, dy;
        #else
        float dx, dy;
        #endif

        if( pt0.x < left_x )
            left_x = pt0.x, left = i;

        if( pt0.x > right_x )
            right_x = pt0.x, right = i;

        if( pt0.y > top_y )
            top_y = pt0.y, top = i;

        if( pt0.y < bottom_y )
            bottom_y = pt0.y, bottom = i;

        Point2D pt = points[(i+1) & (i+1 < n ? -1 : 0)];

        dx = pt.x - pt0.x;
        dy = pt.y - pt0.y;

        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1./std::sqrt(dx*dx + dy*dy));

        pt0 = pt;
    }

    // find convex hull orientation
    {
        #if defined (DOUBLE_FP)
        double ax = vect[n-1].x;
        double ay = vect[n-1].y;
        #else
        float ax = vect[n-1].x;
        float ay = vect[n-1].y;
        #endif

        for( i = 0; i < n; i++ )
        {
            #if defined (DOUBLE_FP)
            double bx = vect[i].x;
            double by = vect[i].y;

            double convexity = ax * by - ay * bx;
            #else
            float bx = vect[i].x;
            float by = vect[i].y;

            float convexity = ax * by - ay * bx;
            #endif

            if( convexity != 0 )
            {
                orientation = (convexity > 0) ? 1.f : (-1.f);
                break;
            }
            ax = bx;
            ay = by;
        }
        //CV_Assert( orientation != 0 );
    }
    base_a = orientation;

    /*****************************************************************************************/
    /*                         init calipers position                                        */
    seq[0] = bottom;
    seq[1] = right;
    seq[2] = top;
    seq[3] = left;
    /*****************************************************************************************/
    /*                         Main loop - evaluate angles and rotate calipers               */

    /* all of edges will be checked while rotating calipers by 90 degrees */
    for( k = 0; k < n; k++ )
    {
        /* sinus of minimal angle */
        /*float sinus;*/

        /* compute cosine of angle between calipers side and polygon edge */
        /* dp - dot product */
        float dp[4] = {
            +base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
            -base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
            -base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
            +base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
        };

        float maxcos = dp[0] * inv_vect_length[seq[0]];

        /* number of calipers edges, that has minimal angle with edge */
        int main_element = 0;

        /* choose minimal angle */
        for ( i = 1; i < 4; ++i )
        {
            float cosalpha = dp[i] * inv_vect_length[seq[i]];
            if (cosalpha > maxcos)
            {
                main_element = i;
                maxcos = cosalpha;
            }
        }

        /*rotate calipers*/
        {
            //get next base
            int pindex = seq[main_element];
            float lead_x = vect[pindex].x*inv_vect_length[pindex];
            float lead_y = vect[pindex].y*inv_vect_length[pindex];
            switch( main_element )
            {
            case 0:
                base_a = lead_x;
                base_b = lead_y;
                break;
            case 1:
                base_a = lead_y;
                base_b = -lead_x;
                break;
            case 2:
                base_a = -lead_x;
                base_b = -lead_y;
                break;
            case 3:
                base_a = -lead_y;
                base_b = lead_x;
                break;
            default:
		std::cout << "Error in rotatingCalipers function:\n";
		std::cout << "   main_element should be 0, 1, 2 or 3\n";
		exit(-2);
            }
        }
        /* change base point of main edge */
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];

	/* find area of rectangle */
	{
            float height;
            float area;
	    
            /* find vector left-right */
            float dx = points[seq[1]].x - points[seq[3]].x;
            float dy = points[seq[1]].y - points[seq[3]].y;
	    
            /* dotproduct */
            float width = dx * base_a + dy * base_b;
	    
            /* find vector left-right */
            dx = points[seq[2]].x - points[seq[0]].x;
            dy = points[seq[2]].y - points[seq[0]].y;
	    
            /* dotproduct */
            height = -dx * base_b + dy * base_a;
	    
            area = width * height;
            if( area <= minarea )
		{
                float *buf = (float *) buffer;
		
                minarea = area;
                /* leftist point */
                ((int *) buf)[0] = seq[3];
                buf[1] = base_a;
                buf[2] = width;
                buf[3] = base_b;
                buf[4] = height;
                /* bottom point */
                ((int *) buf)[5] = seq[0];
                buf[6] = area;
		}
	}
    }                           /* for */

    float *buf = (float *) buffer;
    
    float A1 = buf[1];
    float B1 = buf[3];
    
    float A2 = -buf[3];
    float B2 = buf[1];
    
    float C1 = A1 * points[((int *) buf)[0]].x + points[((int *) buf)[0]].y * B1;
    float C2 = A2 * points[((int *) buf)[5]].x + points[((int *) buf)[5]].y * B2;
    
    float idet = 1.f / (A1 * B2 - A2 * B1);
    
    float px = (C1 * B2 - C2 * B1) * idet;
    float py = (A1 * C2 - A2 * C1) * idet;
    
    out[0] = px;
    out[1] = py;
    
    out[2] = A1 * buf[2];
    out[3] = B1 * buf[2];
    
    out[4] = A2 * buf[4];
    out[5] = B2 * buf[4];
}

//helper function for computing the convex hull
static int Sklansky_( Point2D** array, int start, int end, int* stack, int nsign, int sign2 )
{
    int incr = end > start ? 1 : -1;
    // prepare first triangle
    int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
    int stacksize = 3;

    if( start == end ||
       (array[start]->x == array[end]->x &&
        array[start]->y == array[end]->y) )
    {
        stack[0] = start;
        return 1;
    }

    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;

    end += incr; // make end = afterend

    while( pnext != end )
    {
        // check the angle p1,p2,p3
        float cury = array[pcur]->y;
        float nexty = array[pnext]->y;
        float by = nexty - cury;

        if( SIGN( by ) != nsign )
        {
            float ax = array[pcur]->x - array[pprev]->x;
            float bx = array[pnext]->x - array[pcur]->x;
            float ay = cury - array[pprev]->y;
            float convexity = ay*bx - ax*by; // if >0 then convex angle

            if( SIGN( convexity ) == sign2 && (ax != 0 || ay != 0) )
            {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize++;
            }
            else
            {
                if( pprev == start )
                {
                    pcur = pnext;
                    stack[1] = pcur;
                    pnext += incr;
                    stack[2] = pnext;
                }
                else
                {
                    stack[stacksize-2] = pnext;
                    pcur = pprev;
                    pprev = stack[stacksize-4];
                    stacksize--;
                }
            }
        }
        else
        {
            pnext += incr;
            stack[stacksize-1] = pnext;
        }
    }

    return --stacksize;
}


// helper function to compare 2 points
struct CHullCmpPoints
{
    bool operator()(const Point2D* p1, const Point2D* p2) const
    { return p1->x < p2->x || (p1->x == p2->x && p1->y < p2->y); }
};


// helper function
// computes the convex hull
void convexHull( std::vector<Point2D> _points, std::vector<Point2D>&  _hull, bool clockwise, bool returnPoints )
{
    // CV_INSTRUMENT_REGION()

    //CV_Assert(_points.getObj() != _hull.getObj());
    //Mat points = _points.getMat();
    // assume always 2 elements vectors, so the total = size of vector
    int i, total = _points.size(), nout = 0; //depth = points.depth()
    int miny_ind = 0, maxy_ind = 0;
    //CV_Assert(total >= 0 && (depth == CV_32F || depth == CV_32S));

    if( total == 0 )
    {
        _hull.clear();
        return;
    }

    //returnPoints = !_hull.fixedType() ? returnPoints : _hull.type() != CV_32S;

    // AutoBuffer<Point*> _pointer(total); 
    //AutoBuffer<int> _stack(total + 2), _hullbuf(total);
    Point2D** _pointer = (Point2D**)alloca(total * sizeof(Point2D*));
    int* _stack = (int*)alloca((total +2) * sizeof(int));
    int* _hullbuf= (int*)alloca(total * sizeof(int));
    Point2D** pointer = _pointer;
    Point2D** pointerf = (Point2D**)pointer;
    Point2D* data0 = _points.data();
    int* stack = _stack;
    int* hullbuf = _hullbuf;

    //CV_Assert(points.isContinuous());

    for( i = 0; i < total; i++ )
        pointer[i] = &data0[i];

    // sort the point set by x-coordinate, find min and max y
    std::sort(pointerf, pointerf + total, CHullCmpPoints());
    for( i = 1; i < total; i++ )
        {
            float y = pointerf[i]->y;
            if( pointerf[miny_ind]->y > y )
                miny_ind = i;
            if( pointerf[maxy_ind]->y < y )
                maxy_ind = i;
        }

    if( pointer[0]->x == pointer[total-1]->x &&
        pointer[0]->y == pointer[total-1]->y )
    {
        hullbuf[nout++] = 0;
    }
    else
    {
        // upper half
        int *tl_stack = stack;
        int tl_count = Sklansky_( pointerf, 0, maxy_ind, tl_stack, -1, 1);
        int *tr_stack = stack + tl_count;
        int tr_count = Sklansky_( pointerf, total-1, maxy_ind, tr_stack, -1, -1);

        // gather upper part of convex hull to output
        if( !clockwise )
        {
            std::swap( tl_stack, tr_stack );
            std::swap( tl_count, tr_count );
        }

        for( i = 0; i < tl_count-1; i++ )
            hullbuf[nout++] = int(pointer[tl_stack[i]] - data0);
        for( i = tr_count - 1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[tr_stack[i]] - data0);
        int stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;

        // lower half
        int *bl_stack = stack;
        int bl_count = Sklansky_( pointerf, 0, miny_ind, bl_stack, 1, -1);
        int *br_stack = stack + bl_count;
        int br_count = Sklansky_( pointerf, total-1, miny_ind, br_stack, 1, 1);

        if( clockwise )
        {
            std::swap( bl_stack, br_stack );
            std::swap( bl_count, br_count );
        }

        if( stop_idx >= 0 )
        {
            int check_idx = bl_count > 2 ? bl_stack[1] :
            bl_count + br_count > 2 ? br_stack[2-bl_count] : -1;
            if( check_idx == stop_idx || (check_idx >= 0 &&
                                          pointer[check_idx]->x == pointer[stop_idx]->x &&
                                          pointer[check_idx]->y == pointer[stop_idx]->y) )
            {
                // if all the points lie on the same line, then
                // the bottom part of the convex hull is the mirrored top part
                // (except the exteme points).
                bl_count = MIN( bl_count, 2 );
                br_count = MIN( br_count, 2 );
            }
        }

        for( i = 0; i < bl_count-1; i++ )
            hullbuf[nout++] = int(pointer[bl_stack[i]] - data0);
        for( i = br_count-1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[br_stack[i]] - data0);
    }


    //_hull.create(nout, 1, CV_MAKETYPE(depth, 2));
    //Mat hull = _hull.getMat();
    //size_t step = !hull.isContinuous() ? hull.step[0] : sizeof(Point2D);
    for( i = 0; i < nout; i++ )
	//*(Point2D*)(hull.ptr() + i*step) = data0[hullbuf[i]];
	_hull.push_back(data0[hullbuf[i]]);

}

// we need from the minarea rectangle which contains the points
// but we just need the rotation angle
float minAreaRectAngle(std::vector<Point2D>& points)
{
    float angle = 0.0f;
    std::vector<Point2D> hull;
    Point2D out[3];
 
    convexHull(points, hull, true, true);

    int n = points.size();
    const Point2D* hpoints = hull.data();

    if( n > 2 )
    {
        rotatingCalipers( hpoints, n, (float*)out );
        #if defined (DOUBLE_FP)
        angle = (float)atan2( (double)out[1].y, (double)out[1].x );
        #else
        angle = (float)atan2( (float)out[1].y, (float)out[1].x );
        #endif
    }
    else if( n == 2 )
    {
        #if defined (DOUBLE_FP)
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
        #else
        float dx = hpoints[1].x - hpoints[0].x;
        float dy = hpoints[1].y - hpoints[0].y;
        #endif
        angle = (float)atan2( dy, dx );
    }
    else
    {
	//nothing as we just need the angle
        //if( n == 1 )
            //box.center = hpoints[0];
    }

    return (float)(angle*180.0/PI);
   
}

// Commented because it was replaced
#if 0
/**
   Precomputes all distances and stores the results in sqr_distances.
   Due to symmetry of the distance function (a to b as far apart as b to a),
   only one is stored.
   Size of the aray is (N*(N-1))/2.
   Distance of point i to point j (0..N-1) is stored at index:
       i == j?  nothing stored, distance is 0
       i  > j?  nothing stored, distance is equal to j, i
       j <  i?  distance is stored at:  (((i-1) * i)/2) + j
   To save computations, only the squared distance is stored
   (sufficient for comparison, does not require square root).
*/
void initRadiusSearch(const std::vector<Point> &points, bool**  sqr_distances, float radius)
{
    int n = points.size();
    float radius_sqr = radius * radius;
    *sqr_distances = (bool*) malloc(n * n * sizeof(bool));
    for (int j = 0; j < n; j++)
	for (int i = 0; i < n; i++)
	    {
		float dx = points[i].x - points[j].x;
		float dy = points[i].y - points[j].y;
		float dz = points[i].z - points[j].z;
                float sqr_distance = dx*dx + dy*dy + dz*dz;
                (*sqr_distances)[j*n+i] = sqr_distance <= radius_sqr;
	    }
}


/** 
    own radiusSearch, just goes linear through the array of distances
    returns number of found points
*/
    
int radiusSearch(const int point_index, std::vector<int> & indices, const bool* sqr_distances, int total_points)
{

    indices.clear();

    for (int i = 0; i < point_index; i++)
	{
	    if (sqr_distances[point_index*total_points+i])
		indices.push_back(i);	    
	}
    
    for (int i = point_index+1; i < total_points; ++i){
        if(sqr_distances[point_index * total_points + i])
            indices.push_back(i);
    }

    return indices.size();
}
#endif

// Commented because it was replaced
#if 0
// from pcl library (and heavily modified)
void
extractEuclideanClusters (const PointCloud &cloud, 
			  float tolerance, std::vector<PointIndices> &clusters,
			  unsigned int min_pts_per_cluster, 
			  unsigned int max_pts_per_cluster)
{
  int nn_start_idx = 0;

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (cloud.size (), false);

  std::vector<int> nn_indices;

  // fs: instead of using the flann radiussearch, we use a far simpler one:
  // precompute once distance of all points to all
  // and just iterate of them to find all points from another within a given distance
  // for the used sizes (<26000 points) this is as fast or even faster
  bool *sqr_distances;
  initRadiusSearch(cloud, &sqr_distances, tolerance);

  // Process all points in the indices vector
  for (int i = 0; i < static_cast<int> (cloud.size ()); ++i)
  {
    if (processed[i])
      continue;

    std::vector<int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back (i);

    processed[i] = true;

    while (sq_idx < static_cast<int> (seed_queue.size ()))
	{
	    // Search for sq_idx
	    //int ret = tree->radiusSearch (cloud.points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances);
	    int ret = radiusSearch(seed_queue[sq_idx], nn_indices, sqr_distances, cloud.size());
	    if (!ret)
		{
		    sq_idx++;
		    continue;
		}
	    
	    for (size_t j = nn_start_idx; j < nn_indices.size (); ++j)             // can't assume sorted (default isn't!)
		{
		    if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
			continue;
		    
		    // Perform a simple Euclidean clustering
		    seed_queue.push_back (nn_indices[j]);
		    processed[nn_indices[j]] = true;
		}
	    
	    sq_idx++;
	}

    // If this queue is satisfactory, add to the clusters
    if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
	{
	    PointIndices r;
	    r.indices.resize (seed_queue.size ());
	    for (size_t j = 0; j < seed_queue.size (); ++j)
		// This is the only place where indices come into play
		r.indices[j] = seed_queue[j];
	    
	    // These two lines should not be needed: (can anyone confirm?) -FF
	    //r.indices.assign(seed_queue.begin(), seed_queue.end());
	    std::sort (r.indices.begin (), r.indices.end ());
	    r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
	    //r.header = cloud.header;
	    clusters.push_back (r);   // We could avoid a copy by working directly in the vector
	}
  }
  free(sqr_distances);
}
#endif

void
extractEuclideanClusters (const PointCloud cloud,
			  int cloud_size,
			  float tolerance,
			  std::vector<PointIndices> &clusters,
			  unsigned int min_pts_per_cluster, 
			  unsigned int max_pts_per_cluster
                          #if defined (OPENCL_EPHOS)
                          ,
                          OCL_Struct* OCL_objs
                          #endif
                          )
{

  /*int nn_start_idx = 0;*/

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (/*cloud.size ()*/ cloud_size, false);

  /*std::vector<int> nn_indices;*/
  bool* nn_indices;

  //cudaMallocManaged(&nn_indices, cloud_size * sizeof(bool));
  size_t nbytes_nn_indices = cloud_size * sizeof(bool);

  #if defined (OPENCL_EPHOS)
	  cl_int err;
	  #if defined (OPENCL_CPP_WRAPPER)
	  cl::Buffer buff_nn_indices (OCL_objs->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_nn_indices);
	  #else // No OPENCL_CPP_WRAPPER
	  cl_mem buff_nn_indices =  clCreateBuffer(OCL_objs->rcar_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_nn_indices, NULL, &err);
	  #endif
  #endif

  int* seed_queue;
  // maxsize is number of points
  //cudaMallocManaged(&seed_queue, cloud_size * sizeof(int));
  size_t nbytes_seed_queue = cloud_size * sizeof(int);
  seed_queue = (int*) malloc(nbytes_seed_queue);

  #if defined (OPENCL_EPHOS)
  	#if defined (OPENCL_CPP_WRAPPER)
	cl::Buffer buff_seed_queue (OCL_objs->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_seed_queue);
	#else // No OPENCL_CPP_WRAPPER
	cl_mem buff_seed_queue =  clCreateBuffer(OCL_objs->rcar_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_seed_queue, NULL, &err);
	#endif
  #endif

  // fs: instead of using the flann radiussearch, we use a far simpler one:
  // precompute once distance of all points to all
  // and just iterate of them to find all points from another within a given distance
  // for the used sizes (<26000 points) this is as fast or even faster

  /*unsigned char *sqr_distances;*/
  //cudaMallocManaged(&sqr_distances, (cloud_size * cloud_size * sizeof(bool)));
  size_t nbytes_sqr_distances = cloud_size * cloud_size * sizeof(bool);

  #if defined (OPENCL_EPHOS)
	#if defined (OPENCL_CPP_WRAPPER)
  	cl::Buffer buff_sqr_distances (OCL_objs->context, CL_MEM_READ_WRITE /*| CL_MEM_ALLOC_HOST_PTR*/, nbytes_sqr_distances); // written in initRS, and read in parallelRS kernel
	#else // No OPENCL_CPP_WRAPPER
	cl_mem buff_sqr_distances =  clCreateBuffer(OCL_objs->rcar_context, CL_MEM_READ_WRITE, nbytes_sqr_distances, NULL, &err);
	#endif
  #endif

  // make the initRadiusSearch on the GPU
  /*initRadiusSearch(cloud, &sqr_distances, tolerance);*/
  
  #if defined (OPENCL_EPHOS)
  // Set offset, global & local sizes
  size_t offset = 0;
  size_t local_size     = NUMWORKITEMS_PER_WORKGROUP;
  size_t workgroup_size = (cloud_size + NUMWORKITEMS_PER_WORKGROUP - 1); // rounded up, se we don't miss one
  size_t global_size    = workgroup_size * local_size;

  #if defined (PRINTINFO)
  std::cout << "\t        offset : " << offset          << std::endl;
  std::cout << "\t    local size : " << local_size      << std::endl;
  std::cout << "\tworkgroup size : " << workgroup_size  << std::endl;
  std::cout << "\t   global size : " << global_size     << std::endl;
  #endif

  #if defined (OPENCL_CPP_WRAPPER)
  cl::NDRange ndrange_offset(offset);
  cl::NDRange ndrange_localsize (local_size);
  cl::NDRange ndrange_globalsize(global_size);
  #endif

  // Allocating buffer (in CUDA version this is done within parse PointCloud())
  size_t nbytes_cloud = sizeof(Point) * (cloud_size);

	#if defined (OPENCL_CPP_WRAPPER)	
  	cl::Buffer buff_cloud (OCL_objs->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_cloud);
	#else // No OPENCL_CPP_WRAPPER
	cl_mem buff_cloud =  clCreateBuffer(OCL_objs->rcar_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, nbytes_cloud, NULL, &err);
	#endif

  	// Copying cloud data into its corresponding buffer
	#if defined (OPENCL_CPP_WRAPPER)
  	Point* tmp_cloud = (Point *) OCL_objs->cmdqueue.enqueueMapBuffer(buff_cloud, CL_TRUE, /*CL_MAP_WRITE*/ CL_MAP_WRITE_INVALIDATE_REGION, 0, nbytes_cloud);
  	memcpy(tmp_cloud, cloud, nbytes_cloud);
  	OCL_objs->cmdqueue.enqueueUnmapMemObject(buff_cloud, tmp_cloud);
	#else // No OPENCL_CPP_WRAPPER
        Point* tmp_cloud = (Point *) clEnqueueMapBuffer(OCL_objs->cvengine_command_queue,
							buff_cloud, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0,
							nbytes_cloud, 0, 0, NULL, &err);

	memcpy(tmp_cloud, cloud, nbytes_cloud);
	clEnqueueUnmapMemObject(OCL_objs->cvengine_command_queue, buff_cloud, tmp_cloud, 0, NULL, NULL);
	#endif

	#if defined (OPENCL_CPP_WRAPPER)
	OCL_objs->kernel_initRS.setArg(0, buff_cloud);
  	OCL_objs->kernel_initRS.setArg(1, buff_sqr_distances);
  	OCL_objs->kernel_initRS.setArg(2, cloud_size);
  	#if defined (DOUBLE_FP)
  	OCL_objs->kernel_initRS.setArg(3, static_cast<double>(tolerance*tolerance)); // tolerance is a float arg in this function, but was declared a double in kernel
  	#else
  	OCL_objs->kernel_initRS.setArg(3, (tolerance*tolerance));
  	#endif
	#else // No OPENCL_CPP_WRAPPER
	err = clSetKernelArg (OCL_objs->kernel_initRS, 0, sizeof(cl_mem),       &buff_cloud);
	err = clSetKernelArg (OCL_objs->kernel_initRS, 1, sizeof(cl_mem),       &buff_sqr_distances);
	err = clSetKernelArg (OCL_objs->kernel_initRS, 2, sizeof(int),          &cloud_size);

  	#if defined (DOUBLE_FP)
        double tmp_scalar_sqr_radius = static_cast<double>(tolerance*tolerance);
	err = clSetKernelArg (OCL_objs->kernel_initRS, 3, sizeof(double),       &tmp_scalar_sqr_radius);
  	#else
	float tmp_scalar_sqr_radius = tolerance*tolerance;
	err = clSetKernelArg (OCL_objs->kernel_initRS, 3, sizeof(float),        &tmp_scalar_sqr_radius);
  	#endif
	#endif

	#if defined (OPENCL_CPP_WRAPPER)
  	OCL_objs->cmdqueue.enqueueNDRangeKernel(OCL_objs->kernel_initRS, 
				         ndrange_offset,
                                         ndrange_globalsize,
                                         ndrange_localsize);
	#else // No OPENCL_CPP_WRAPPER
 	err = clEnqueueNDRangeKernel(OCL_objs->cvengine_command_queue, OCL_objs->kernel_initRS, 1, NULL,  &global_size, &local_size, 0, NULL, NULL);
	#endif


  // Marked as unnecessary by CodeXL on Vega56
  /*
  // http://horacio9573.no-ip.org/cuda/group__CUDART__DEVICE_gb76422145b5425829597ebd1003303fe.html
  //cudaDeviceSynchronize();
  OCL_objs->cmdqueue.finish();
  */

  /*
  // Some kernel arguments could have been set here (instead of inside loops)
  // but doing so results in slightly slower execution times
  OCL_objs->kernel_parallelRS.setArg(0, buff_seed_queue);
  OCL_objs->kernel_parallelRS.setArg(1, buff_nn_indices);
  OCL_objs->kernel_parallelRS.setArg(2, buff_sqr_distances);
  OCL_objs->kernel_parallelRS.setArg(5, cloud_size);
  */
  #endif

  // Process all points in the indices vector
  for (int i = 0; i < static_cast<int> (/*cloud.size ()*/ cloud_size); ++i)
  {
    if (processed[i])
      continue;

    // See above different definition as: int* seed_queue;
    /*
    std::vector<int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back (i);
    */

    int queue_last_element = 0;

    seed_queue[queue_last_element++] = i;

    processed[i] = true;

    int new_elements = 1;

    // Commented because it was replaced
    #if 0
    while (sq_idx < static_cast<int> (seed_queue.size ()))
	{
	    // Search for sq_idx
	    //int ret = tree->radiusSearch (cloud.points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances);
	    int ret = radiusSearch(seed_queue[sq_idx], nn_indices, sqr_distances, cloud.size());
	    if (!ret)
		{
		    sq_idx++;
		    continue;
		}
	    
	    for (size_t j = nn_start_idx; j < nn_indices.size (); ++j)             // can't assume sorted (default isn't!)
		{
		    if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
			continue;
		    
		    // Perform a simple Euclidean clustering
		    seed_queue.push_back (nn_indices[j]);
		    processed[nn_indices[j]] = true;
		}
	    
	    sq_idx++;
	}
      #endif


    while (new_elements > 0) // repeat until we got not more additional points
	{
	    // Enqueue kernel
	    #if defined (OPENCL_EPHOS)

	    #if defined (OPENCL_CPP_WRAPPER)
            // Copying seed_queue into its corresponding buffer
            int* tmp_seed_queue = (int *) OCL_objs->cmdqueue.enqueueMapBuffer(buff_seed_queue, CL_TRUE, /*CL_MAP_WRITE*/ CL_MAP_WRITE_INVALIDATE_REGION, 0, nbytes_seed_queue);
            memcpy(tmp_seed_queue, seed_queue, nbytes_seed_queue);
            OCL_objs->cmdqueue.enqueueUnmapMemObject(buff_seed_queue, tmp_seed_queue);
		
	    // Some kernel arguments could have been set outside both enclosing loops
	    // but doing so results in slightly slower execution times
	    OCL_objs->kernel_parallelRS.setArg(0, buff_seed_queue);
	    OCL_objs->kernel_parallelRS.setArg(1, buff_nn_indices);
            OCL_objs->kernel_parallelRS.setArg(2, buff_sqr_distances);
            OCL_objs->kernel_parallelRS.setArg(3, queue_last_element - new_elements);
            OCL_objs->kernel_parallelRS.setArg(4, queue_last_element);
            OCL_objs->kernel_parallelRS.setArg(5, cloud_size);

            OCL_objs->cmdqueue.enqueueNDRangeKernel(OCL_objs->kernel_parallelRS, 
				         ndrange_offset,
                                         ndrange_globalsize,
                                         ndrange_localsize);

    	    // Reading nn_indices from its corresponding buffer
            bool* nn_indices = (bool *) OCL_objs->cmdqueue.enqueueMapBuffer(buff_nn_indices, CL_TRUE, CL_MAP_READ, 0, nbytes_nn_indices);

	    OCL_objs->cmdqueue.finish();
 
            #else // No OPENCL_CPP_WRAPPER
            int* tmp_seed_queue  = (int *) clEnqueueMapBuffer(OCL_objs->cvengine_command_queue,
							      buff_seed_queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0,
							      nbytes_seed_queue, 0, 0, NULL, &err);
	    memcpy(tmp_seed_queue, seed_queue, nbytes_seed_queue);
	    clEnqueueUnmapMemObject(OCL_objs->cvengine_command_queue, buff_seed_queue, tmp_seed_queue, 0, NULL, NULL);

	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 0, sizeof(cl_mem),       &buff_seed_queue);
	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 1, sizeof(cl_mem),       &buff_nn_indices);
	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 2, sizeof(cl_mem),       &buff_sqr_distances);
            int tmp_scalar_sub = queue_last_element - new_elements;
	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 3, sizeof(int),          &tmp_scalar_sub);
	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 4, sizeof(int),          &queue_last_element);
	    err = clSetKernelArg (OCL_objs->kernel_parallelRS, 5, sizeof(int),          &cloud_size);
	    err = clEnqueueNDRangeKernel(OCL_objs->cvengine_command_queue, OCL_objs->kernel_parallelRS, 1, NULL,  &global_size, &local_size, 0, NULL, NULL);

            bool* nn_indices = (bool *) clEnqueueMapBuffer(OCL_objs->cvengine_command_queue,
				       	                   buff_nn_indices, CL_TRUE, CL_MAP_READ, 0,
							   nbytes_nn_indices, 0, 0, NULL, &err);

            clFinish(OCL_objs->cvengine_command_queue);
            #endif // OPENCL_CPP_WRAPPER

	    #endif // OPENCL_EPHOS

            new_elements = 0;

            for (size_t j = 0; j < cloud_size; ++j)             // can't assume sorted (default isn't!)
		{
		  if (nn_indices[j] == false)
		    continue;
		  if (processed[j])        // Has this point been processed before ?
		    continue;
		  // Perform a simple Euclidean clustering
		  seed_queue[queue_last_element++] = j;
		  processed[j] = true;
		  new_elements++; 
		}

	    #if defined (OPENCL_EPHOS)

	    #if defined (OPENCL_CPP_WRAPPER)
	    OCL_objs->cmdqueue.enqueueUnmapMemObject(buff_nn_indices, nn_indices);
	    #else // No OPENCL_CPP_WRAPPER
            clEnqueueUnmapMemObject(OCL_objs->cvengine_command_queue, buff_nn_indices, nn_indices, 0, NULL, NULL);
	    #endif

  	    #endif

	}

    // If this queue is satisfactory, add to the clusters
    //std::cout<<"queue size: " << queue_last_element << "\n";
    /*if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)*/
    if (queue_last_element >= min_pts_per_cluster && queue_last_element <= max_pts_per_cluster)
	{
	    PointIndices r;
	    r.indices.resize (/*seed_queue.size ()*/ queue_last_element);

	    for (size_t j = 0; j < /*seed_queue.size ()*/ queue_last_element; ++j)
		// This is the only place where indices come into play
		r.indices[j] = seed_queue[j];
	    
	    // These two lines should not be needed: (can anyone confirm?) -FF
	    //r.indices.assign(seed_queue.begin(), seed_queue.end());
	    std::sort (r.indices.begin (), r.indices.end ());
	    r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
	    //r.header = cloud.header;
	    clusters.push_back (r);   // We could avoid a copy by working directly in the vector
	}
  }

  // sqr_distances is not required anymore, buffer_sqr_distances is used instead  
  free(seed_queue);

  #if defined (OPENCL_EPHOS)
  
  #if defined (OPENCL_CPP_WRAPPER)

  #else // No OPENCL_CPP_WRAPPER
  clReleaseMemObject(buff_nn_indices);
  clReleaseMemObject(buff_seed_queue);
  clReleaseMemObject(buff_sqr_distances);
  clReleaseMemObject(buff_cloud);
  #endif

  #endif
}

// from pcl
inline bool 
comparePointClusters (const PointIndices &a, const PointIndices &b)
{
    return (a.indices.size () < b.indices.size ());
}

// from pcl (EuclideanCluster)
void
extract (/*const PointCloud *input_,*/
	 const PointCloud input_,
	 int cloud_size,
	 std::vector<PointIndices> &clusters, 
         #if defined (DOUBLE_FP)
         double cluster_tolerance_
         #else
         float cluster_tolerance_
         #endif

         #if defined (OPENCL_EPHOS)
         ,
         OCL_Struct* OCL_objs
         #endif
)
{
    /*if (input_->empty())*/
    if (cloud_size == 0)
	{
	    clusters.clear ();
	    return;
	}

    // Send the input dataset to the spatial locator
    /*extractEuclideanClusters (*input_, static_cast<float> (cluster_tolerance_), clusters, _cluster_size_min, _cluster_size_max );*/
    extractEuclideanClusters (input_,
			      cloud_size,
			      static_cast<float> (cluster_tolerance_),
			      clusters,
                              _cluster_size_min,
			      _cluster_size_max 
                              #if defined (OPENCL_EPHOS)
                              ,
                              OCL_objs
                              #endif
                             );

    // Sort the clusters based on their size (largest one first)
    std::sort (clusters.rbegin (), clusters.rend (), comparePointClusters);

}

// clusterAndColor from Autoware (using PCL)
void euclidean_clustering::clusterAndColor(
                                           #if defined (OPENCL_EPHOS)
                                           OCL_Struct* OCL_objs,
                                           #endif
				           /*const PointCloud *in_cloud_ptr,*/
					   const PointCloud in_cloud_ptr,
					   int cloud_size,
		     			   PointCloudRGB *out_cloud_ptr,
		                           BoundingboxArray* in_out_boundingbox_array,
		                           Centroid* in_out_centroids,
                                           #if defined (DOUBLE_FP)
		                           double in_max_cluster_distance=0.5
                                           #else
                                           float in_max_cluster_distance=0.5
	                                   #endif
                                           )
{
    //pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    //tree->setInputCloud (in_cloud_ptr);

    std::vector<PointIndices> cluster_indices;

    /*extract (in_cloud_ptr, cluster_indices, in_max_cluster_distance);*/
    extract (in_cloud_ptr, 
             cloud_size,
             cluster_indices,
             in_max_cluster_distance
             #if defined (OPENCL_EPHOS)
             ,
             OCL_objs
             #endif
             );
    
    /////////////////////////////////
    //---   3. Color clustered points
    /////////////////////////////////
    int j = 0;
    unsigned int k = 0;


    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
	    PointCloudRGB *current_cluster = new PointCloudRGB;//coord + color cluster
	    //assign color to each cluster
	    PointDouble centroid = {0.0, 0.0, 0.0};
	    for (auto pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                {
		    //fill new colored cluster point by point
		    PointRGB p;
		    /*
		    p.x = (*in_cloud_ptr)[*pit].x;
		    p.y = (*in_cloud_ptr)[*pit].y;
		    p.z = (*in_cloud_ptr)[*pit].z;
		    */
		    p.x = (in_cloud_ptr)[*pit].x;
		    p.y = (in_cloud_ptr)[*pit].y;
		    p.z = (in_cloud_ptr)[*pit].z;
		    p.r = 10; //_colors[k].val[0];
		    p.g = 20; //_colors[k].val[1];
		    p.b = 30; //_colors[k].val[2];

		    /*
		    centroid.x += (*in_cloud_ptr)[*pit].x;
		    centroid.y += (*in_cloud_ptr)[*pit].y;
		    centroid.z += (*in_cloud_ptr)[*pit].z;
		    */
		    centroid.x += (in_cloud_ptr)[*pit].x;
		    centroid.y += (in_cloud_ptr)[*pit].y;
		    centroid.z += (in_cloud_ptr)[*pit].z;

		    current_cluster->push_back(p);
                }

	    centroid.x /= it->indices.size();
	    centroid.y /= it->indices.size();
	    centroid.z /= it->indices.size();

	    //get min, max
	    float min_x=std::numeric_limits<float>::max();float max_x=-std::numeric_limits<float>::max();
	    float min_y=std::numeric_limits<float>::max();float max_y=-std::numeric_limits<float>::max();
	    float min_z=std::numeric_limits<float>::max();float max_z=-std::numeric_limits<float>::max();
	    for(unsigned int i=0; i<current_cluster->size();i++)
                {
		    if((*current_cluster)[i].x<min_x)  min_x = (*current_cluster)[i].x;
		    if((*current_cluster)[i].y<min_y)  min_y = (*current_cluster)[i].y;
		    if((*current_cluster)[i].z<min_z)  min_z = (*current_cluster)[i].z;
		    if((*current_cluster)[i].x>max_x)  max_x = (*current_cluster)[i].x;
		    if((*current_cluster)[i].y>max_y)  max_y = (*current_cluster)[i].y;
		    if((*current_cluster)[i].z>max_z)  max_z = (*current_cluster)[i].z;
                }

	    //	    Point min_point(min_x, min_y, min_z), max_point(max_x, max_y, max_z);

	    float l = max_x - min_x;
	    float w = max_y - min_y;
	    float h = max_z - min_z;

	    Boundingbox bounding_box;
	    //bounding_box.header = _velodyne_header;

	    bounding_box.position.x = min_x + l/2;
	    bounding_box.position.y = min_y + w/2;
	    bounding_box.position.z = min_z + h/2;

	    bounding_box.dimensions.x = ((l<0)?-1*l:l);
	    bounding_box.dimensions.y = ((w<0)?-1*w:w);
	    bounding_box.dimensions.z = ((h<0)?-1*h:h);

            #if defined (DOUBLE_FP)
	    double rz = 0;
            #else
            float rz = 0;
            #endif
	    
	    if (_pose_estimation) 
                {
		    std::vector<Point2D> inner_points;
		    for (unsigned int i=0; i<current_cluster->size(); i++)
                        {
			    Point2D ip;
			    ip.x = ((*current_cluster)[i].x + fabs(min_x))*8;
			    ip.y = ((*current_cluster)[i].y + fabs(min_y))*8;
			    inner_points.push_back(ip);
                        }

		    if (inner_points.size() > 0)
                        {
			    rz = minAreaRectAngle(inner_points) * PI / 180.0;
                        }
                }

	    //tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
	    //void setRPY(const btScalar& roll, const btScalar& pitch, const btScalar& yaw);
            #if defined (DOUBLE_FP)
	    double halfYaw = rz * 0.5;  
	    double cosYaw = cos(halfYaw);
	    double sinYaw = sin(halfYaw);
            #else
	    float halfYaw = rz * 0.5;  
	    float cosYaw = cos(halfYaw);
	    float sinYaw = sin(halfYaw);
            #endif

	    // schreibt x,y,z und w in eine Message (bounding_box.pose.orientiation)
	    //tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);
	    bounding_box.orientation.x = 0.0; //x
	    bounding_box.orientation.y = 0.0; //y
	    bounding_box.orientation.z = sinYaw; //z
	    bounding_box.orientation.w = cosYaw; //w, formerly yzx

	    if (  bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z > 0 &&
		  bounding_box.dimensions.x < 15 && bounding_box.dimensions.y >0 && bounding_box.dimensions.y < 15 &&
		  max_z > -1.5 && min_z > -1.5 && min_z < 1.0 )
		{
		    in_out_boundingbox_array->boxes.push_back(bounding_box);
		    in_out_centroids->points.push_back(centroid);
		}

	    //*out_cloud_ptr = *out_cloud_ptr + *current_cluster;//sum up all the colored cluster into a complete pc
	    out_cloud_ptr->insert(out_cloud_ptr->end(), current_cluster->begin(), current_cluster->end());

	    j++; k++;
        }

}

/**
   using this from Autoware as main kernel, which is invoking clusterAndColor, 
   as this invokes the following with multiple times with similar parameters
   (and avoids storing a partial result multiple times)

   even if named in_out, boundingbox_array and centroids are output values
*/
void euclidean_clustering::segmentByDistance(
                                             #if defined (OPENCL_EPHOS)
                                             OCL_Struct* OCL_objs,
                                             #endif
                                             /*const PointCloud *in_cloud_ptr,*/
					     const PointCloud in_cloud_ptr,
					     int cloud_size,
                PointCloudRGB *out_cloud_ptr,
                BoundingboxArray *in_out_boundingbox_array,
                Centroid *in_out_centroids,
                #if defined (DOUBLE_FP)
                double in_max_cluster_distance=0.5
                #else
                float in_max_cluster_distance=0.5
                #endif
                )
{
        //cluster the pointcloud according to the distance of the points using different thresholds (not only one for the entire pc)
        //in this way, the points farther in the pc will also be clustered

        //0 => 0-15m d=0.5
        //1 => 15-30 d=1
        //2 => 30-45 d=1.6
        //3 => 45-60 d=2.1
        //4 => >60   d=2.6

	// -----------------------------------------------
	// Using same strategy as in the CUDA version
	// -----------------------------------------------
	// fs: in the cuda version just a big array is used and the pointers are sorted into
        // the segments and the segments are pointers to their first element of the segment
	/*
        PointCloud*   cloud_segments_array[5];
        double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};
	*/
        PointCloud   cloud_segments_array[5];
	int segment_size[5] = {0, 0, 0, 0, 0};
	int *segment_index = (int*) malloc(cloud_size * sizeof(int));

        #if defined (DOUBLE_FP)
	double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};
        #else
        float thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};
        #endif

	/*
        for(unsigned int i=0; i<5; i++)
        {
                PointCloud *tmp_cloud = new PointCloud;
                cloud_segments_array[i] = tmp_cloud;
        }
	*/

        for (unsigned int i=0; i</*in_cloud_ptr->size()*/cloud_size; i++)
        {
                Point current_point;
                current_point.x = (in_cloud_ptr)[i].x; /*(*in_cloud_ptr)[i].x;*/
                current_point.y = (in_cloud_ptr)[i].y; /*(*in_cloud_ptr)[i].y;*/
		current_point.z = (in_cloud_ptr)[i].z; /*(*in_cloud_ptr)[i].z;*/

                float origin_distance = sqrt( pow(current_point.x,2) + pow(current_point.y,2) );

		/*
                if     (origin_distance < 15)   {cloud_segments_array[0]->push_back (current_point);}
                else if(origin_distance < 30)   {cloud_segments_array[1]->push_back (current_point);}
                else if(origin_distance < 45)   {cloud_segments_array[2]->push_back (current_point);}
                else if(origin_distance < 60)   {cloud_segments_array[3]->push_back (current_point);}
                else                            {cloud_segments_array[4]->push_back (current_point);}
		*/
                if     (origin_distance < 15)   {segment_index[i] = 0; segment_size[0]++;}
                else if(origin_distance < 30)   {segment_index[i] = 1; segment_size[1]++;}
                else if(origin_distance < 45)   {segment_index[i] = 2; segment_size[2]++;}
                else if(origin_distance < 60)   {segment_index[i] = 3; segment_size[3]++;}
                else                            {segment_index[i] = 4; segment_size[4]++;}		
        }

	// -----------------------------------------------
	// Using same strategy as in the CUDA version
	// -----------------------------------------------

	// now resort the point cloud array and get rid of the segment_index array
	int current_segment_pos[5] = { 0, 
                                       segment_size[0], 
				       segment_size[0]+segment_size[1], 
                                       segment_size[0]+segment_size[1]+segment_size[2],
				       segment_size[0]+segment_size[1]+segment_size[2]+segment_size[3] 
                                     };

	//cloud_segments_array[4] = in_cloud_ptr + current_segment_pos[4];
	for (int segment = 0; segment < 5; segment++) // find points belonging into each segment
	  {
	    cloud_segments_array[segment] = in_cloud_ptr + current_segment_pos[segment];
	    for (int i = current_segment_pos[segment]; i < cloud_size; i++) // all in the segment before are already sorted in
	      {
		if (segment_index[i] == segment)
		  {
		    Point swap_tmp = in_cloud_ptr[current_segment_pos[segment]];
		    in_cloud_ptr[current_segment_pos[segment]] = in_cloud_ptr[i];
		    in_cloud_ptr[i] = swap_tmp;		   
		    
		    segment_index[i] = segment_index[current_segment_pos[segment]];
		    segment_index[current_segment_pos[segment]] = segment;

		    current_segment_pos[segment]++;
		  }
	      }
	  }
	
	free(segment_index);

        for(unsigned int i=0; i<5; i++)
        {

		/*
		#if defined (PRINTINFO)
		std::cout << std::endl;
		std::cout << "cluster type: " << i << std::endl;
		#endif
		*/
		/*
        	clusterAndColor(cloud_segments_array[i], out_cloud_ptr, in_out_boundingbox_array, in_out_centroids, thresholds[i]);
		*/
		clusterAndColor(
                                #if defined (OPENCL_EPHOS)
                                OCL_objs,
                                #endif
                                cloud_segments_array[i], 
				segment_size[i], 
				out_cloud_ptr,
				in_out_boundingbox_array,
				in_out_centroids,
				thresholds[i]
                                );
        }
}


#if 0
void parsePointCloud(std::ifstream& input_file, PointCloud *cloud)
{
    int size = 0;
    Point p;
    input_file.read((char*)&(size), sizeof(int));

    try {
	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&p.x, sizeof(float));
		input_file.read((char*)&p.y, sizeof(float));
		input_file.read((char*)&p.z, sizeof(float));
		cloud->push_back(p);
	    }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}
#endif
void parsePointCloud(std::ifstream& input_file, PointCloud *cloud, int *cloud_size)
{
    input_file.read((char*)(cloud_size), sizeof(int));

    /*
    __host__ ​cudaError_t cudaMallocManaged ( void** devPtr, 
                                             size_t size, 
                                             unsigned int  flags = cudaMemAttachGlobal )
    Allocates memory that will be automatically managed by the Unified Memory system.
    */
    // cudaMallocManaged(cloud, sizeof(Point) * (*cloud_size));

    *cloud = (Point*) malloc(sizeof(Point) * (*cloud_size));
    
    try {
	for (int i = 0; i < *cloud_size; i++)
	    {
		input_file.read((char*)&(*cloud)[i].x, sizeof(float));
		input_file.read((char*)&(*cloud)[i].y, sizeof(float));
		input_file.read((char*)&(*cloud)[i].z, sizeof(float));
	    }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void parseOutCloud(std::ifstream& input_file, PointCloudRGB *cloud)
{
    int size = 0;
    PointRGB p;
    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&p.x, sizeof(float));
		input_file.read((char*)&p.y, sizeof(float));
		input_file.read((char*)&p.z, sizeof(float));
		input_file.read((char*)&p.r, sizeof(uint8_t));
		input_file.read((char*)&p.g, sizeof(uint8_t));
		input_file.read((char*)&p.b, sizeof(uint8_t));				    
		cloud->push_back(p);
	    }
    }  catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    } 
}

void parseBoundingboxArray(std::ifstream& input_file, BoundingboxArray *bb_array)
{
    int size = 0;
    Boundingbox bba;

    #if defined (DOUBLE_FP)
    // Nothing required for double-precision version
    #else
    double temp;
    #endif

    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
                #if defined (DOUBLE_FP)
		input_file.read((char*)&bba.position.x, sizeof(double));
		input_file.read((char*)&bba.position.y, sizeof(double));
		input_file.read((char*)&bba.orientation.x, sizeof(double));
		input_file.read((char*)&bba.orientation.y, sizeof(double));
		input_file.read((char*)&bba.orientation.z, sizeof(double));
		input_file.read((char*)&bba.orientation.w, sizeof(double));
		input_file.read((char*)&bba.dimensions.x, sizeof(double));
		input_file.read((char*)&bba.dimensions.y, sizeof(double));
                #else
		input_file.read((char*)&temp, sizeof(double));
		bba.position.x=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.position.y=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.orientation.x=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.orientation.y=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.orientation.z=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.orientation.w=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.dimensions.x=temp;
		input_file.read((char*)&temp, sizeof(double));
		bba.dimensions.y=temp;
                #endif
		bb_array->boxes.push_back(bba);
	    }
    }  catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

void parseCentroids(std::ifstream& input_file, Centroid *centroids)
{
    int size = 0;
    PointDouble p;

    #if defined (DOUBLE_FP)
    // Nothing required for double-precision version
    #else
    double temp;
    #endif

    try {
	input_file.read((char*)&(size), sizeof(int));
	
	
	for (int i = 0; i < size; i++)
	    {
                #if defined (DOUBLE_FP)
		input_file.read((char*)&p.x, sizeof(double));
		input_file.read((char*)&p.y, sizeof(double));
		input_file.read((char*)&p.z, sizeof(double));
                #else
		input_file.read((char*)&temp, sizeof(double));
		p.x = temp;
		input_file.read((char*)&temp, sizeof(double));
		p.y = temp;
		input_file.read((char*)&temp, sizeof(double));
		p.z = temp;
                #endif
		centroids->points.push_back(p);
	    }
    } catch (std::ifstream::failure e) {
	std::cerr << "Error reading file\n";
	exit(-3);
    }
}

// return how many could be read
int euclidean_clustering::read_next_testcases(int count)
{
  int i;

  delete [] in_cloud_ptr;
  delete [] cloud_size;
  delete [] out_cloud_ptr;
  delete [] out_boundingbox_array;
  delete [] out_centroids;

  in_cloud_ptr = new PointCloud[count];
  cloud_size = new int [count];
  out_cloud_ptr = new PointCloudRGB[count];
  out_boundingbox_array = new BoundingboxArray[count];
  out_centroids = new Centroid[count];
  
  for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
    {
      /*parsePointCloud(input_file, in_cloud_ptr + i);*/
      parsePointCloud(input_file, &in_cloud_ptr[i], &cloud_size[i]);
    }
  
  return i;
}


void euclidean_clustering::init() {

  std::cout << "init\n";

  testcases = /*350*/ 350;
  
  input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  try {
      input_file.open("../../../data/ec_input.dat", std::ios::binary);
      output_file.open("../../../data/ec_output.dat", std::ios::binary);
  }  catch (std::ifstream::failure e) {
      std::cerr << "Error opening file\n";
      exit(-3);
  }
  error_so_far = false;
  max_delta = 0.0;

  testcases = read_number_testcases(input_file);
  
  in_cloud_ptr = NULL;
  out_cloud_ptr = NULL;
  out_boundingbox_array = NULL;
  out_centroids = NULL;
    
  std::cout << "done\n" << std::endl;
}

void euclidean_clustering::run(int p) {
  pause_func();

  #if defined (OPENCL_EPHOS)
  OCL_Struct OCL_objs;

  #if defined (OPENCL_CPP_WRAPPER)
 
  try {
    // Get list of OpenCL platforms.
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);
 
    if (platform.empty()) {
      std::cerr << "OpenCL platforms not found." << std::endl;
      exit(EXIT_FAILURE);
    }

    // Get first available GPU device which supports double precision.
    cl::Context context;
    std::vector<cl::Device> device;
    for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
      std::vector<cl::Device> pldev;

      try {
        p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

        for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
	  if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

          #if defined (DOUBLE_FP)
	  std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

	  if (
	      ext.find("cl_khr_fp64") == std::string::npos &&
	      ext.find("cl_amd_fp64") == std::string::npos
	     ) continue;
          #endif

	  device.push_back(*d);
	  context = cl::Context(device);
	}
      } catch(...) {
        device.clear();
      }
    }

    #if defined (DOUBLE_FP)
    if (device.empty()) {
      std::cerr << "GPUs with double precision not found." << std::endl;
      exit(EXIT_FAILURE);
    }
    #endif

    //#if defined (PRINTINFO)
    std::cout << "EPHoS OpenCL device: " << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    //#endif

    // Create command queue.
    cl::CommandQueue queue(context, device[0]);

    // Preparing data to pass to kernel functions
    //OCL_objs.platform = platform[0];
    OCL_objs.device   = device[0];
    OCL_objs.context  = context;
    OCL_objs.cmdqueue = queue;

  } catch (const cl::Error &err) {
    std::cerr
	 << "OpenCL error: "
	 << err.what() << "(" << err.err() << ")"
	 << std::endl;
    exit(EXIT_FAILURE);
  }

  // Kernel code was stringified, rather than read from file
  std::string sourceCode = initRadiusSearch_ocl_krnl;
  cl::Program::Sources sourcesCL = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size()));

  // Create program
  cl::Program program(OCL_objs.context, sourcesCL);

  try {   
    // Building, specifying options
    // Notice initial space within strings
    const std::string ocl_build_options = std::string(" -I ./ocl/device/") +
				 	  std::string(" -DNUMWORKITEMS_PER_WORKGROUP=") + NUMWORKITEMS_PER_WORKGROUP_STRING
                                          #if defined (DOUBLE_FP)
                                          + std::string(" -DDOUBLE_FP");
                                          #else
					  ;				
                                          #endif

    //#if defined (PRINTINFO)
    std::cout << "Kernel compilation flags passed to OpenCL device: " << std::endl << ocl_build_options << std::endl;
    //#endif

    // Conversion from std::string to char* is required:
    // https://stackoverflow.com/a/7352131/1616865
    program.build(ocl_build_options.c_str());

    // Use this in case previous build() is not be supported in other platforms
    /*
    program.build();
    */
		    
    } catch (const cl::Error&) {
      std::cerr
        << "OpenCL compilation error" << std::endl
	<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(OCL_objs.device)
	<< std::endl;
			    
      exit(EXIT_FAILURE);
    }

  cl::Kernel initRadiusSearch_kernel(program, "initRadiusSearch");
  cl::Kernel parallelRadiusSearch_kernel(program, "parallelRadiusSearch");

  OCL_objs.kernel_initRS = initRadiusSearch_kernel;
  OCL_objs.kernel_parallelRS = parallelRadiusSearch_kernel;

  #else // No OPENCL_CPP_WRAPPER

	// Define OpenCL platform and needed objects
	// DO NOT USE CPP BINDINGS
	cl_int err;
	cl_platform_id* local_platform_id;
	cl_uint         local_platformCount;

	err = clGetPlatformIDs(0, NULL, &local_platformCount);

	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
 		exit(EXIT_FAILURE);
  	}
	local_platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * local_platformCount);

	err = clGetPlatformIDs(local_platformCount, local_platform_id, NULL);
  	if (err != CL_SUCCESS){
		printf("Error: clGetPlatformIDs(): %d\n",err);
		fflush(stdout);
	 	exit(EXIT_FAILURE);
  	}

	OCL_objs.rcar_platform = local_platform_id;

	// retrieving the CVEngine device
	err = clGetDeviceIDs(OCL_objs.rcar_platform[0], CL_DEVICE_TYPE_GPU, 1, & OCL_objs.cvengine_device, NULL);

	// creating the context for RCar OpenCL devices
	OCL_objs.rcar_context =  clCreateContext(0, 1, &OCL_objs.cvengine_device, NULL, NULL, &err);

	// creating a command queueu for CVEngine accelerator
	OCL_objs.cvengine_command_queue = clCreateCommandQueue(OCL_objs.rcar_context, OCL_objs.cvengine_device, CL_QUEUE_PROFILING_ENABLE, &err);

	// constructing the OpenCL program for the points2image function
	cl_program program = clCreateProgramWithSource(OCL_objs.rcar_context, 1, (const char **)&initRadiusSearch_ocl_krnl, NULL, &err);

        const std::string ocl_build_options = std::string(" -I ./ocl/device/") +
				 	      std::string(" -DNUMWORKITEMS_PER_WORKGROUP=") + NUMWORKITEMS_PER_WORKGROUP_STRING
                                              #if defined (DOUBLE_FP)
                                              + std::string(" -DDOUBLE_FP");
                                              #else
					      ;				
                                              #endif

	// building the OpenCL program for all the objects
	err = clBuildProgram(program, 1, &OCL_objs.cvengine_device, /*NULL*/ ocl_build_options.c_str(), NULL, NULL);

	// kernel
	OCL_objs.kernel_initRS = clCreateKernel(program, "initRadiusSearch", &err);
	OCL_objs.kernel_parallelRS = clCreateKernel(program, "parallelRadiusSearch", &err);

  #endif // OPENCL_CPP_WRAPPER

  #endif // OPENCL_EPHOS

  while (read_testcases < testcases)
    {

      int count = read_next_testcases(p);

      #if defined (PRINTINFO)
      //std::cout << std::endl;
      std::cout << "# testcase: " << read_testcases << ", count:" << count << std::endl;
      #endif

if(read_testcases != 1) {
      unpause_func();
}
      for (int i = 0; i < count; i++)
	  {
	      // actual kernel invocation
	      segmentByDistance(
                                #if defined (OPENCL_EPHOS)
                                &OCL_objs,
                                #endif
                                /*&in_cloud_ptr[i],*/
				in_cloud_ptr[i],
				cloud_size[i],
				&out_cloud_ptr[i],
				&out_boundingbox_array[i],
				&out_centroids[i]
                               );
	  }
if(read_testcases != 1) {
      pause_func();
}
      check_next_outputs(count);
    }

  #if defined (OPENCL_EPHOS)
  
  #if defined (OPENCL_CPP_WRAPPER)

  #else // No OPENCL_CPP_WRAPPER
  err = clReleaseKernel(OCL_objs.kernel_initRS);
  err = clReleaseKernel(OCL_objs.kernel_parallelRS);
  err = clReleaseProgram(program);
  err = clReleaseCommandQueue(OCL_objs.cvengine_command_queue);
  err = clReleaseContext(OCL_objs.rcar_context);
  #endif

  #endif
}

// implemented for comparing, so we can make a canonical order, using plain lexicographical order
inline bool 
compareRGBPoints (const PointRGB &a, const PointRGB &b)
{
    if (a.x != b.x)
	return (a.x < b.x);
    else
	if (a.y != b.y)
	    return (a.y < b.y);
	else
	    return (a.z < b.z);
}

// implemented for comparing, so we can make a canonical order, using plain lexicographical order
inline bool 
comparePoints (const PointDouble &a, const PointDouble &b)
{
    if (a.x != b.x)
	return (a.x < b.x);
    else
	if (a.y != b.y)
	    return (a.y < b.y);
	else
	    return (a.z < b.z);
}


// implemented for comparing, so we can make a canonical order, using plain lexicographical order
inline bool 
compareBBs (const Boundingbox &a, const Boundingbox &b)
{
    if (a.position.x != b.position.x)
	return (a.position.x < b.position.x);
    else
	if (a.position.y != b.position.y)
	    return (a.position.y < b.position.y);
	else
	    if (a.dimensions.x != b.dimensions.x)
		return (a.dimensions.x < b.dimensions.x);
	    else
		return (a.dimensions.y < b.dimensions.y);
}


void euclidean_clustering::check_next_outputs(int count)
{
    PointCloudRGB reference_out_cloud;
    BoundingboxArray reference_bb_array;
    Centroid reference_centroids;
    
    for (int i = 0; i < count; i++)
	{
	    parseOutCloud(output_file, &reference_out_cloud);
	    parseBoundingboxArray(output_file, &reference_bb_array);
	    parseCentroids(output_file, &reference_centroids);

	    // as the result is still right when points/boxes/centroids are in different order,
	    // we sort the result and reference to normalize it and we can compare it
	    std::sort(reference_out_cloud.begin(), reference_out_cloud.end(), compareRGBPoints);
	    std::sort(out_cloud_ptr[i].begin(), out_cloud_ptr[i].end(), compareRGBPoints);
	    std::sort(reference_bb_array.boxes.begin(), reference_bb_array.boxes.end(), compareBBs);
	    std::sort(out_boundingbox_array[i].boxes.begin(), out_boundingbox_array[i].boxes.end(), compareBBs);
	    std::sort(reference_centroids.points.begin(), reference_centroids.points.end(), comparePoints);
	    std::sort(out_centroids[i].points.begin(), out_centroids[i].points.end(), comparePoints);
	    
	    //std::cout << "out_cloud size (actual/expected): " << out_cloud_ptr[i].size() << "/" << reference_out_cloud.size() << "\n";
	    //std::cout << "bounding boxes (actual/expected): " << out_boundingbox_array[i].boxes.size() << "/" << reference_bb_array.boxes.size() << "\n";
	    //std::cout << "centroids      (actual/expected): " << out_centroids[i].points.size() << "/" << reference_centroids.points.size() << "\n";
	    if (reference_out_cloud.size() != out_cloud_ptr[i].size())
		{
		    error_so_far = true;
		    continue;
		}
	    if (reference_bb_array.boxes.size() != out_boundingbox_array[i].boxes.size())
		{
		    error_so_far = true;
		    continue;
		}
	    if (reference_centroids.points.size() != out_centroids[i].points.size())
		{
		    error_so_far = true;
		    continue;
		}

	    for (int j = 0; j < reference_out_cloud.size(); j++)
		{
		    max_delta = MAX(abs(out_cloud_ptr[i][j].x - reference_out_cloud[j].x), max_delta);
		    max_delta = MAX(abs(out_cloud_ptr[i][j].y - reference_out_cloud[j].y), max_delta);
		    max_delta = MAX(abs(out_cloud_ptr[i][j].z - reference_out_cloud[j].z), max_delta);
		}
	    for (int j = 0; j < reference_bb_array.boxes.size(); j++)
		{	          
		    //std::cout << "ow (a/e) : " << out_boundingbox_array[i].boxes[j].orientation.w << "/" << reference_bb_array.boxes[j].orientation.w << "\n";
		    //std::cout << "oz (a/e) : " << out_boundingbox_array[i].boxes[j].orientation.z << "/" << reference_bb_array.boxes[j].orientation.z << "\n";	    
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].position.x - reference_bb_array.boxes[j].position.x), max_delta);		    
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].position.y - reference_bb_array.boxes[j].position.y), max_delta);
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].dimensions.x - reference_bb_array.boxes[j].dimensions.x), max_delta);		    
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].dimensions.y - reference_bb_array.boxes[j].dimensions.y), max_delta); 
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].orientation.x - reference_bb_array.boxes[j].orientation.x), max_delta);
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].orientation.y - reference_bb_array.boxes[j].orientation.y), max_delta);
		    continue;
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].orientation.z - reference_bb_array.boxes[j].orientation.z), max_delta);
		    max_delta = MAX(abs(out_boundingbox_array[i].boxes[j].orientation.w - reference_bb_array.boxes[j].orientation.w), max_delta);
		}
	    for (int j = 0; j < reference_centroids.points.size(); j++)
		{   
		    max_delta = MAX(abs(out_centroids[i].points[j].x - reference_centroids.points[j].x), max_delta);
		    max_delta = MAX(abs(out_centroids[i].points[j].y - reference_centroids.points[j].y), max_delta);
		    max_delta = MAX(abs(out_centroids[i].points[j].z - reference_centroids.points[j].z), max_delta);
		    //std::cout << "x (a/e) : " << out_centroids[i].points[j].x << "/" << reference_centroids.points[j].x << "\n";
		}
	    reference_bb_array.boxes.clear();
	    reference_out_cloud.clear();
	    reference_centroids.points.clear();
	}
}

bool euclidean_clustering::check_output() {
    std::cout << "checking output \n";

    input_file.close();
    output_file.close();
    
    std::cout << "max delta: " << max_delta << "\n";
    if ((max_delta > MAX_EPS) || error_so_far)
          return false;
    return true;
}

euclidean_clustering a = euclidean_clustering();
kernel& myKernel = a;
