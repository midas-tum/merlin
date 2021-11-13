#define MATLABCALL 1
#include "sampling.cpp"
#include <mex.h>

/*****************************************************************************
 ** MEX FUNCTION
 *****************************************************************************/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double nbPhase, nbPartition, acc, center, nbSegment, nbRep, isGolden, isVariable, isInOut, isCenter, isSamePattern, isVerbose;
    double *outMatrix1, *outMatrix2;
    double *mask;
    double *n_SamplesInSpace, *n_sampled, *nb_spiral;
    
    /*	Get input parameters	*/
	mexPrintf("Get input");  
    nbPhase     = mxGetScalar(prhs[0]);
    nbPartition = mxGetScalar(prhs[1]);
    acc         = mxGetScalar(prhs[2]);
    center      = mxGetScalar(prhs[3]);
    nbSegment   = mxGetScalar(prhs[4]);
    nbRep       = mxGetScalar(prhs[5]);
    isGolden    = mxGetScalar(prhs[6]);
    isVariable  = mxGetScalar(prhs[7]);
    isInOut     = mxGetScalar(prhs[8]);
    isCenter    = mxGetScalar(prhs[9]);
    isSamePattern = mxGetScalar(prhs[10]);
    isVerbose   = mxGetScalar(prhs[11]);
    mexPrintf("...Done\n");     
	
    //size_t ncols = round_d(nbPhase*nbPartition/(acc));
	size_t ncols = round_d(nbRep*nbPhase*nbPartition/(1.0));
	//size_t ncols = round_d(nbPhase*nbPartition/(1.0));
    mexPrintf("Create output"); 
    plhs[0] = mxCreateDoubleMatrix((mwSize)ncols,1,mxREAL);
    outMatrix1 = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix((mwSize)ncols,1,mxREAL);
    outMatrix2 = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix((mwSize)nbPhase,(mwSize)nbPartition,mxREAL);
    mask = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
    n_SamplesInSpace = mxGetPr(plhs[3]);
	
	plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
    n_sampled = mxGetPr(plhs[4]);
    
    plhs[5] = mxCreateDoubleMatrix(1,1,mxREAL);
    nb_spiral = mxGetPr(plhs[5]);
    
    mexPrintf("...Done\n"); 
    /*	Call CPP-function Here */
    sampling_mex(nbPhase, nbPartition, acc, center, nbSegment, nbRep, isGolden, isVariable, isInOut, isCenter, isSamePattern, isVerbose, outMatrix1, outMatrix2, mask, n_SamplesInSpace, n_sampled, nb_spiral);

}




