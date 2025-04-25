#define _USE_MATH_DEFINES	// to use PI
#include <cstdlib>
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <ctime>			//for random number generation
#include <fstream>			//to write the sampling mask in a .txt-file
#include <time.h>			//to measure the runtime
#include <sstream>
#include <vector>
#include <omp.h>
#ifndef PYTHONCALL
    #define PYTHONCALL 0
#endif
//#include "boost/smart_ptr.hpp" // download from: http://sourceforge.net/projects/boost/files/boost/1.57.0/boost_1_57_0.zip/download

using namespace std;

#ifndef CLASSES
#define CLASSES
#define MAX_PATTERN_LINES			512
#define MAX_PATTERN_PARTITIONS		512
#define	MAX_PATTERN_MEASUREMENTS	64
#define PDF_GENERATION_PREC			10000

struct vd_options{ float fully_sampled; float ringHeight; float ringRadius; };

#include "Point.h"
#include "LinkedList.h"
#include "SomeFunctions.h"
#include "VariableDensity.h"
#include "VDSamplingUpper.h"
#include "Approx.h"
#endif

static int ***startPD(long lLines, long lPartitions, double dAccel, float fully_sampled, float pF_val, bool pFx, int lPhases, short int v_type, short int s_type, bool ellipticalMask, float p, float n, short int body_part, float iso_fac, bool lVerbose);
static int ***startGaussian(long lLines, long lPartitions, double dAccel, float fully_sampled, float pF_val, bool pFx, long lPhases, bool lVerbose);
static void run_subsampling(double* kSpacePolar_LinInd, double* kSpacePolar_ParInd, double* phaseInd, long nX, long nY, double accels, short int subsampleType, float fully_sampled, float pF_value, bool pF_x, int lPhases, short int vd_type, short int smpl_type, bool ellip_mask, float p, float n, short int body_region, float iso_fac, bool lVerbose);



