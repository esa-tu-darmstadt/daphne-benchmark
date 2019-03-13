#include "benchmark.h"
#include <iostream>
#include <fstream>
#include "datatypes.h"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

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
    void clusterAndColor(const PointCloud *in_cloud_ptr,
		    PointCloudRGB *out_cloud_ptr,
		    BoundingboxArray *in_out_boundingbox_array,
		    Centroid *in_out_centroids,
		    double in_max_cluster_distance);
    void segmentByDistance(const PointCloud *in_cloud_ptr,
			   PointCloudRGB *out_cloud_ptr,
			   BoundingboxArray *in_out_boundingbox_array,
			   Centroid *in_out_centroids,
			   double in_max_cluster_distance);
    virtual int read_next_testcases(int count);
    virtual void check_next_outputs(int count);
    int read_number_testcases(std::ifstream& input_file);
    PointCloud *in_cloud_ptr;
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
        double dx, dy;

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
        double ax = vect[n-1].x;
        double ay = vect[n-1].y;

        for( i = 0; i < n; i++ )
        {
            double bx = vect[i].x;
            double by = vect[i].y;

            double convexity = ax * by - ay * bx;

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
        angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    }
    else if( n == 2 )
    {
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
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

// from pcl
inline bool 
comparePointClusters (const PointIndices &a, const PointIndices &b)
{
    return (a.indices.size () < b.indices.size ());
}


// from pcl (EuclideanCluster)
void
extract (const PointCloud *input_, std::vector<PointIndices> &clusters, double cluster_tolerance_)
{
    if (input_->empty())
	{
	    clusters.clear ();
	    return;
	}

    // Send the input dataset to the spatial locator
    extractEuclideanClusters (*input_, static_cast<float> (cluster_tolerance_), clusters, _cluster_size_min, _cluster_size_max );

    // Sort the clusters based on their size (largest one first)
    std::sort (clusters.rbegin (), clusters.rend (), comparePointClusters);

}


// clusterAndColor from Autoware (using PCL)
void euclidean_clustering::clusterAndColor(const PointCloud *in_cloud_ptr,
		     PointCloudRGB *out_cloud_ptr,
		     BoundingboxArray* in_out_boundingbox_array,
		     Centroid* in_out_centroids,
		     double in_max_cluster_distance=0.5)
{
    //pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    //tree->setInputCloud (in_cloud_ptr);

    std::vector<PointIndices> cluster_indices;

    extract (in_cloud_ptr, cluster_indices, in_max_cluster_distance);
    
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
		    p.x = (*in_cloud_ptr)[*pit].x;
		    p.y = (*in_cloud_ptr)[*pit].y;
		    p.z = (*in_cloud_ptr)[*pit].z;
		    p.r = 10; //_colors[k].val[0];
		    p.g = 20; //_colors[k].val[1];
		    p.b = 30; //_colors[k].val[2];

		    centroid.x += (*in_cloud_ptr)[*pit].x;
		    centroid.y += (*in_cloud_ptr)[*pit].y;
		    centroid.z += (*in_cloud_ptr)[*pit].z;

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

	    double rz = 0;
	    
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
	    double halfYaw = rz * 0.5;  
	    double cosYaw = cos(halfYaw);
	    double sinYaw = sin(halfYaw);

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
void euclidean_clustering::segmentByDistance(const PointCloud *in_cloud_ptr,
                PointCloudRGB *out_cloud_ptr,
                BoundingboxArray *in_out_boundingbox_array,
                Centroid *in_out_centroids,
                double in_max_cluster_distance=0.5)
{
        //cluster the pointcloud according to the distance of the points using different thresholds (not only one for the entire pc)
        //in this way, the points farther in the pc will also be clustered

        //0 => 0-15m d=0.5
        //1 => 15-30 d=1
        //2 => 30-45 d=1.6
        //3 => 45-60 d=2.1
        //4 => >60   d=2.6

        PointCloud*   cloud_segments_array[5];
        double thresholds[5] = {0.5, 1.1, 1.6, 2.3, 2.6f};

       for(unsigned int i=0; i<5; i++)
        {
                PointCloud *tmp_cloud = new PointCloud;
                cloud_segments_array[i] = tmp_cloud;
        }

        for (unsigned int i=0; i<in_cloud_ptr->size(); i++)
        {
                Point current_point;
                current_point.x = (*in_cloud_ptr)[i].x;
                current_point.y = (*in_cloud_ptr)[i].y;
		current_point.z = (*in_cloud_ptr)[i].z;

                float origin_distance = sqrt( pow(current_point.x,2) + pow(current_point.y,2) );

                if              (origin_distance < 15 ) {cloud_segments_array[0]->push_back (current_point);}
                else if(origin_distance < 30)   {cloud_segments_array[1]->push_back (current_point);}
                else if(origin_distance < 45)   {cloud_segments_array[2]->push_back (current_point);}
                else if(origin_distance < 60)   {cloud_segments_array[3]->push_back (current_point);}
                else                                                    {cloud_segments_array[4]->push_back (current_point);}
        }


        for(unsigned int i=0; i<5; i++)
        {
                clusterAndColor(cloud_segments_array[i], out_cloud_ptr, in_out_boundingbox_array, in_out_centroids, thresholds[i]);
        }
}



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
    try {
	input_file.read((char*)&(size), sizeof(int));

	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&bba.position.x, sizeof(double));
		input_file.read((char*)&bba.position.y, sizeof(double));
		input_file.read((char*)&bba.orientation.x, sizeof(double));
		input_file.read((char*)&bba.orientation.y, sizeof(double));
		input_file.read((char*)&bba.orientation.z, sizeof(double));
		input_file.read((char*)&bba.orientation.w, sizeof(double));
		input_file.read((char*)&bba.dimensions.x, sizeof(double));
		input_file.read((char*)&bba.dimensions.y, sizeof(double));
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

    try {
	input_file.read((char*)&(size), sizeof(int));
	
	
	for (int i = 0; i < size; i++)
	    {
		input_file.read((char*)&p.x, sizeof(double));
		input_file.read((char*)&p.y, sizeof(double));
		input_file.read((char*)&p.z, sizeof(double));
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
  delete [] out_cloud_ptr;
  delete [] out_boundingbox_array;
  delete [] out_centroids;

  in_cloud_ptr = new PointCloud[count];
  out_cloud_ptr = new PointCloudRGB[count];
  out_boundingbox_array = new BoundingboxArray[count];
  out_centroids = new Centroid[count];
  
  for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
    {
      parsePointCloud(input_file, in_cloud_ptr + i);
    }
  
  return i;
}


void euclidean_clustering::init() {

  std::cout << "init\n";

  testcases = 350;
  
  input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  try {
      input_file.open("../../../data/ec_input.dat", std::ios::binary);
      output_file.open("../../../data/ec_output.dat", std::ios::binary);
  }  catch (std::ifstream::failure e) {
      std::cerr << "Error opening file\n";
      exit(-3);
  }

  testcases = read_number_testcases(input_file);
  error_so_far = false;
  max_delta = 0.0;


  in_cloud_ptr = NULL;
  out_cloud_ptr = NULL;
  out_boundingbox_array = NULL;
  out_centroids = NULL;
    
  std::cout << "done\n" << std::endl;
}

void euclidean_clustering::run(int p) {
  pause_func();
  
  while (read_testcases < testcases)
    {
      int count = read_next_testcases(p);
      unpause_func();
      for (int i = 0; i < count; i++)
	  {
	      // actual kernel invocation
	      segmentByDistance(&in_cloud_ptr[i],
				&out_cloud_ptr[i],
				&out_boundingbox_array[i],
				&out_centroids[i]);
	  }
      pause_func();
      check_next_outputs(count);
    }
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
