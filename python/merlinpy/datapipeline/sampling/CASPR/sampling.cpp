#ifndef MATLABCALL
    #define MATLABCALL 0
#endif
#define _USE_MATH_DEFINES

#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>

#if MATLABCALL > 0
    #include <mex.h>
#endif
/*
%	ACCELERATED VARIABLE DENSITY SPIRAL WITH ARBITRARY (Golden, TinyGolden, linear, ...) ANGLE
%   AND SEVERAL PHASES/CONTRAST/... - INTERLEAVED
%   FOR USE ON SIEMENS MRI
%	-----------------------------------------------------
%
%	Function generates variable density spiral with golden angle
%   Acceleration factor and number of segment per spiral are provided
%   by the user
%
%
%	INPUTS:
%	-------
%	nbPhase     = number of phase to acquired
%	nbPartition = number of partition to acquired
%	acc         = acceleration factor
%	center      = size of the fully sampled center of k-space
%   nbSegment   = number of segments/rings
%   nbRepIn     = phases/repetitions/contrasts
%   isGolden    = select golden angle (>1) or incremental spirals (=0) (boolean), i.e. following options: 0 = linear-linear, 1=golden-golden, 2=tinyGolden-golden, 3=linear-golden, 4=noIncr-golden
%   isVariable  = variable-density on/off (boolean)
%   isInOut     = spiral inward-outward trajectory (boolean)
%   isCenter    = k-space central point acquired each spiral arm (1D self-navigation)
%   isSamePattern = only prepare same pattern for all phases/repetitions/contrasts
% 
%
%
%	OUTPUTS:
%	--------
%	Phas        = Acquired phase of k-space
%	Part        = Acquired partition of k-space
%	mask        = mask of the acquired k-space
%   nSamplesInSpace = sampling points in space
%   n_sampled   = sampled points
%	nb_spiral   = number of acquired spiral
%
%
%	Aurelien Bustin -- 07/04/2017.
%   aurelien.bustin@kcl.ac.uk
%
%	Thomas Kuestner -- 2019
% 	thomas.kuestner@kcl.ac.uk
% 	extended CMRA VD-CASPR sampling:
%   - non-interleaved mode with arbitrary angle increment between spirals
%   - several phases/contrasts/...
%   - inward-out trajectory
%   - self-navigation
*/



/**************************************************************
 ** ROUND DOUBLE
 *************************************************************/
int round_d(double number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

/**************************************************************
 ** MODULO FUNCTION
 *************************************************************/
double modulo(double a, double q)  // matlab style
{
   	if(q == 0.0)
	{
		return a;
	} else {
		double b = a / q;
		return (b - floor(b)) * q;
	}
}

long lmodulo(long a, long b)
{
	return (b == 0) ? b : long(modulo((double)a, (double)b));	
}

long lLimiter(long a, long b)
{
    while(a >= b)
    {
        a = a-b;
    }
    if(a >= b)
    {
        return a-b;
    } else if (a < 0) {
        return b-1;
    } else {
        return a;
    }
    
}


/**************************************************************
 ** GENERATE GOLDEN ANGLE
 *************************************************************/
std::vector<long> generate_golden_angle2(long nb_spiral, double gr, double lr) {
    
    //double gr = (sqrt(5.0)-1.0)/2.0;//0.6180;
    //double lr = 1.0/((1.0+sqrt(5.0))/2.0 + 12.0 -1.0); // 12.0 determines order of tiny angle
    double inc_gr = gr * nb_spiral;
    double inc_lr = lr * nb_spiral;
    double V = 0; 
    //double V = gr * (lRep-1);
    long N = 1;
    int I;
    //long lRep;
    std::vector<long> golden_vector(nb_spiral,0);
    std::vector<long> exist(nb_spiral,0);
    
	while (N<=nb_spiral) {

	    I = round_d(V) + 1;

	    if (I>nb_spiral) {
		I=1;
	    }

        if(lr == 0.0)
        {
            if (exist[I-1]<=1) {
            golden_vector[N-1] = I; // +lRep*nb_spiral
            exist[I-1] += 1;
            N = N+1;
            }
        } else {
            if (exist[I-1]==0) {
            golden_vector[N-1] = I; // +lRep*nb_spiral
            exist[I-1] = 1;
            N = N+1;
            }
        }

        if(N % 2 != 0)
        {
            V = V + inc_gr;
        } else {
            V = V + inc_lr;
        }
	    while (V > nb_spiral) {
		V = V - nb_spiral;
	    }


	}
    
    return golden_vector;
    
}

std::vector<double> generate_golden_angle2_list(long nb_spiral, double gr, double lr) {
    
    //double gr = (sqrt(5.0)-1.0)/2.0;//0.6180;
    //double lr = 1.0/((1.0+sqrt(5.0))/2.0 + 12.0 -1.0); // 12.0 determines order of tiny angle
    double inc_gr = gr * nb_spiral;
    double inc_lr = lr * nb_spiral;
    double V = 0; 
    //double V = gr * (lRep-1);
    long N = 1;
    int I;
    //long lRep;
    std::vector<double> golden_vector_list(nb_spiral,0);
    std::vector<long> exist(nb_spiral,0);
    
	while (N<=nb_spiral) {

	    I = round_d(V) + 1;

	    if (I>nb_spiral) {
		I=1;
	    }

        if(lr == 0.0)
        {
            if (exist[I-1]<=1) {
            golden_vector_list[N-1] = I; // +lRep*nb_spiral
            exist[I-1] += 1;
            N = N+1;
            }
        } else {
            if (exist[I-1]==0) {
            golden_vector_list[N-1] = V; // +lRep*nb_spiral
            exist[I-1] = 1;
            N = N+1;
            }
        }

        if(N % 2 != 0)
        {
            V = V + inc_gr;
        } else {
            V = V + inc_lr;
        }
	    while (V > nb_spiral) {
		V = V - nb_spiral;
	    }


	}
    
    return golden_vector_list;
    
}

double fFindTinyGolden(long nb_spiral)
{
    long N=3;
    bool lFound = false;
    long G1, G2, Gprev;
    while(N <= 20 && !lFound)
    {
        G1 = 1 + N;
        G2 = N;
        while(G1 != nb_spiral && G1 < nb_spiral)
        {
            Gprev = G1;
            G1 = G1 + G2;
            G2 = Gprev;
        }
        if(G1 == nb_spiral)
        {
            lFound = true;
            break;
        }
        N = N + 1;
    }
    if(!lFound)
        N = 12; // safety fall-back

    #if MATLABCALL > 0
        mexPrintf("Choosing N=%d\n", N);
    #endif
    return 1.0/((1.0+sqrt(5.0))/2.0 + double(N) -1.0);
}

long lFindShift(std::vector<long> GOLDEN_VECTOR, std::vector<double> GOLDEN_VECTOR_LIST, std::vector<long> lExist, long nb_spiral, long lAngleIdx, long lLastSpiral, bool isInOut, bool isSamePattern, double isGolden, double dInc_lr)
{
    double dInc_gr = (sqrt(5.0)-1.0)/2.0 * nb_spiral;
    //double dInc_lr = 1.0/((1.0+sqrt(5.0))/2.0 + 12.0 -1.0) * nb_spiral; // 12.0 determines order of tiny angle
    dInc_lr *= nb_spiral;
    double dInc_lin = 2 * M_PI / nb_spiral;
    double dIncAngle = 0.0;
    std::vector<double> minVal(GOLDEN_VECTOR_LIST.size(), 1000*M_PI);
    std::vector<double>::iterator dIter;
    std::vector< long >::iterator itvec; 
    long lShift;
    long idx;
    
    switch(long(isGolden))
    {
        case 1:
            dIncAngle = dInc_gr;
            break;
        case 2:
            if(lAngleIdx % 2 == 0)
            {
                dIncAngle = dInc_lr; // next one is tiny angle
            } else {
                dIncAngle = dInc_gr;
            }
            break;
        case 3:
            if(lAngleIdx % 2 == 0)
            {
                dIncAngle = dInc_lin; // next one is linear angle
            } else {
                dIncAngle = dInc_gr;
            }
            break;
        case 4:
            if(lAngleIdx % 2 == 0)
            {
                dIncAngle = 0.0; // next one is tiny angle
            } else {
                dIncAngle = dInc_gr;
            }
            break;
    }
        
    dIncAngle = GOLDEN_VECTOR_LIST[lLastSpiral] + dIncAngle;
    lShift = round_d(dIncAngle) + 1;
    while (lShift > nb_spiral) {
        lShift = lShift - nb_spiral;
	}
    itvec = std::find(GOLDEN_VECTOR.begin(), GOLDEN_VECTOR.end(), lShift); // find last used element
    idx = std::distance(GOLDEN_VECTOR.begin(), itvec);
    #if MATLABCALL > 0
        mexPrintf("--FindShift: lLastSpiral = %d, dIncAngle = %.2f, lShift = %d, idx = %d\n", lLastSpiral, dIncAngle, lShift, idx);
    #endif
    if (itvec != GOLDEN_VECTOR.end() && lExist[idx] == 0) // && idx != lLastSpiral)
    {
        #if MATLABCALL > 0
            mexPrintf("Element Found\n");
        #endif
        //lShift = std::distance(GOLDEN_VECTOR.begin(), itvec);
        lShift = idx;
    } else {
        #if MATLABCALL > 0
            mexPrintf("Element Not Found -> find closest\n");
        #endif
        for(long i=0; i<GOLDEN_VECTOR_LIST.size(); i++)
        {
            #if MATLABCALL > 0
                mexPrintf("%d, ", lExist[i]);
            #endif
            if(lExist[i] == 0)
                minVal[i] = fabs(GOLDEN_VECTOR_LIST[i] - dIncAngle);
        }
        #if MATLABCALL > 0
            mexPrintf("\nminVal: ");
            for(long i=0; i<GOLDEN_VECTOR_LIST.size(); i++)
            {
                mexPrintf("%.1f, ", minVal[i]);
            }
        #endif
        dIter = std::min_element(minVal.begin(), minVal.end());
        lShift = long(std::distance(minVal.begin(), dIter));
        #if MATLABCALL > 0
            mexPrintf("\n > Shift = %d\n", lShift);
        #endif
    }
    
    return lShift;
}

double dGetAddon(long lRep, double dGolden, double dTinyGolden, double dLin, long lGolden)
{
    double dReturn = 0.0;
    long i;
    switch(lGolden)
    {
        case 0: // linear - linear
            dReturn = fmod(dLin * lRep, 2.0*M_PI);
            break;
        case 1: // golden - golden
            dReturn = fmod(dGolden * lRep, 2.0*M_PI);
            break;

        case 2: // tinyGolden - golden
            for(i=0; i<=lRep; i++)
            {
                if(i % 2 == 0)
                {
                    dReturn += dTinyGolden;
                } else {
                    dReturn += dGolden;
                }
            }
            break;

        case 3: // linear - golden
            for(i=0; i<=lRep; i++)
            {
                if(i % 2 == 0)
                {
                    dReturn += dLin;
                } else {
                    dReturn += dGolden;
                }
            }
            break;

        case 4: // no-incr - golden
            for(i=0; i<=lRep; i++)
            {
                if(i % 2 == 0)
                {
                    dReturn += 0.0;
                } else {
                    dReturn += dGolden;
                }
            }
            break;                
    }
    return dReturn;
}

/**************************************************************
 ** SAVE MASK TO IMAGE
 *************************************************************/
void save2img(const double* mask, int sx, int sy) {
    
    FILE *fp;
    static unsigned char color[3];
    fp = fopen("mask.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", sy, sx);
    
    for (int i = 0; i < sx; ++ i) {
        for (int j = 0; j < sy; ++ j) {
            color[0] = (int)mask[j*sx+i]*255 & 255;
            color[1] = (int)mask[j*sx+i]*255 & 255;
            color[2] = (int)mask[j*sx+i]*255 & 255;
            fwrite(color, 1, 3, fp);
        }
    }
    fclose(fp);
}

/**************************************************************
 ** PRINT SAMPLING MASK
 *************************************************************/
 #if MATLABCALL > 0
void fPrintSampling(std::vector< std::vector<unsigned short int> > samplingMask, long numLin, long numPar)
{
	long lLine, lPartition, lSampled = 0;

    mexPrintf("*** Sampling Pattern ***\n");
    mexPrintf("     ");
	for (lPartition = 0; lPartition < numPar; lPartition++) mexPrintf("%03u ", lPartition);
	mexPrintf("\n");

	for (lLine = 0; lLine < numLin; lLine++) {
		mexPrintf("%03u  ", lLine);
		for (lPartition = 0; lPartition < numPar; lPartition++) {
				
			if (samplingMask[lLine][lPartition]) {
				mexPrintf("%03u ", samplingMask[lLine][lPartition]);
				lSampled++;
			} else {
				mexPrintf("--- ");
			}
		}
		mexPrintf("\n");
	}	
	mexPrintf("Sampled points: %d\n", lSampled);
}

/**************************************************************
 ** PRINT MASK for INDEXRINGALL
 *************************************************************/
void fPlotIndexRing(std::vector< std::vector< std::vector<long> > > kSpacePolar_Lin, std::vector< std::vector< std::vector<long> > > kSpacePolar_Par, long numLin, long numPar, long m_lNumConcentricRings, long nb_spiral, long nbRep)
{
	std::vector< std::vector<unsigned short int> > samplingMask(numLin, std::vector<unsigned short int>(numPar,0));

	long lRep, lRing, lSpiralInter, lLinSample, lParSample;
	
    for (lRep = 0 ; lRep < nbRep ; ++lRep) // repetitions
    {
        for(lSpiralInter = 0 ; lSpiralInter < nb_spiral ; ++lSpiralInter) //nb_spiral
        { 
            for(lRing = 0; lRing < m_lNumConcentricRings; ++lRing)
            {           
                lLinSample = kSpacePolar_Lin[lRep][lSpiralInter][lRing]; // index_ringAll[lShots][lRing][minInd-1][2]
                lParSample = kSpacePolar_Par[lRep][lSpiralInter][lRing];
                samplingMask[(unsigned short int)lLinSample][(unsigned short int)lParSample] += 1;
            }
        }
    }
	mexPrintf("*** Sampling Points ***\n");
	fPrintSampling(samplingMask, numLin, numPar);		
}
#endif

/**************************************************************
 ** PRINT INT
 *************************************************************/
void printi(const char* value, int i)
{
    printf("%s: %d \n", value, i);
}

/**************************************************************
 ** PRINT DOUBLE
 *************************************************************/
void printd(const char* value, double d)

{
    printf("%s: %f \n", value, d);
}

/**************************************************************
 ** SORT MATRIX W.R.T FIRST COLUMN
 *************************************************************/
bool sortColumn0(std::vector<double> rowA, std::vector<double> rowB) {
    return (rowA[0] < rowB[0]);
}

/**************************************************************
 ** SORT MATRIX W.R.T SECOND COLUMN
 *************************************************************/
bool sortColumn1(std::vector<double> rowA, std::vector<double> rowB) {
    return (rowA[1] < rowB[1]);
}

/**************************************************************
 ** ACCUMULATE VECTOR (SUM ELEMENTS FROM START TO END)
 *************************************************************/
long summup(std::vector<long> vect, long start, long end, long sum)
{
    if (start>end) {
        return sum;
    } else if (start==end) {
        return (sum+vect[start-1]);
    } else {
        for(long n=start-1;n<end;n++)
        {
            sum += vect[n];
        }
        return sum;
    }
}


/**************************************************************
 **
 ** MAIN FILE: VARIABLE DENSITY
 **
 *************************************************************/
void sampling_mex(double nbPhase, double nbPartition, double acc, double center,
                  double nbSegment, double nbRepIn, double isGolden, double isVariable, double isInOut, double isCenter, double isSamePattern, double isVerbose,
                  double *output1, double *output2,
                  double *mask, double *nSamplesInSpace, double *n_sampled, double *nb_spiral_real_out) {
    

    long m_lNumConcentricRings = (long)nbSegment;
    long numLin = (long)nbPhase;
    long numPar = (long)nbPartition;
    long nbRep  = (long)nbRepIn;
    long mm_lNumSpiralInter;
    long i, j, k, id = 0;
    int lIndex = 0;
    long lKSCenterLin = long(round_d((double) numLin/2))-1;
	long lKSCenterPar = long(round_d((double) numPar/2))-1;
    bool bIsCenter = isCenter > 0;
    bool bIsInOut = isInOut > 0;
    bool bIsGoldenAngle = isGolden > 0;
    bool bIsSamePattern = isSamePattern > 0;
    double m_dGoldenAngle = isGolden;
    
    if(bIsCenter)
    {
        m_lNumConcentricRings -= 1;
    }
        
    
    /**************************************************************
     ** GET NB_POINTS TO ACQUIRED IF FULLY ACQUIRED
     *************************************************************/
    long m_lNumSpiralInter = 0;
    for (j = 0 ; j<numLin ; ++j) {
        for (i = 0; i<numPar ; ++i) {
            if (((-1.0 + (double)2.0*i/((double)nbPartition - 1.0))*(-1.0 + (double)2.0*i/((double)nbPartition - 1.0)) + (-1.0 + (double)2.0*j/((double)nbPhase - 1.0))*(-1.0 + (double)2.0*j/((double)nbPhase - 1.0)))  <= sqrt(1 + (-1.0 + (2.0*(floor((double)nbPhase/2.0)-3.0))/(double)(nbPhase-1))*(-1.0 + (2.0*(floor((double)nbPhase/2.0)-3.0))/(double)(nbPhase-1)))) {
                m_lNumSpiralInter++;
            }
        }
    }
    
    /**************************************************************
     ** INITIAL ORDERING AND VECTORIZE
     *************************************************************/
    double r_center = (double)center/100.0 * sqrt(1 + (-1.0 + (2.0*(floor((double)numLin/2.0)-3.0))/(double)(numLin-1))*(-1.0 + (2.0*(floor((double)numLin/2.0)-3.0))/(double)(numLin-1)));
    double rs1 = 1.0, rs2 = 1.0; // resolution ratio
    long nbExtern, sum1, minInd, lRing, lSpiralInter, lRep, idx, sum;
    long nb_spiral, ring_center, ring_extern, nbCenter = 0;
    long nb_points_center = 0;
    double KY, KZ, variability, ky_idx, kz_idx;
    double lastSpiralPreviousRep, dShiftAngle, multipleGoldenAngle, dPower;
    
    std::vector<long> index_Lin_vec(m_lNumSpiralInter);
    std::vector<long> index_Par_vec(m_lNumSpiralInter);
    std::vector<double> ky(m_lNumSpiralInter);
    std::vector<double> kz(m_lNumSpiralInter);
    
    std::vector<long> ind_extern(m_lNumSpiralInter);
    std::vector<long> ind_center(m_lNumSpiralInter);
    
    
    /**************************************************************
     ** REMOVE CORNER (WE DON'T ACQUIRE CORNER POINTS)
     *************************************************************/
    double edge, radius;
    edge    = -1.0 + (2.0*(floor((double)nbPhase/2.0)-3.0))/(double)(nbPhase-1);
    radius  = sqrt(1 + edge*edge);
    
    std::vector< std::vector<double> > vec_rad_ang_ind(m_lNumSpiralInter, std::vector<double>(3));
    
    for(j = 0 ; j<nbPhase ; ++j) {
        for(i = 0; i<nbPartition ; ++i) {
            
            ky_idx = -1.0 + 2.0*j/(nbPhase - 1.0);
            kz_idx = -1.0 + 2.0*i/(nbPartition - 1.0);
            
            if ((ky_idx*ky_idx + kz_idx*kz_idx) <= radius) {
                ky[id] = ky_idx;
                kz[id] = kz_idx;
                index_Lin_vec[id] = j;
                index_Par_vec[id] = i;
                
                vec_rad_ang_ind[id][1] = sqrt( ky_idx*rs1*ky_idx*rs1 + kz_idx*rs2*kz_idx*rs2 ); // radius
                vec_rad_ang_ind[id][0] = atan2(ky_idx*rs1*sin(10*vec_rad_ang_ind[id][1]) + kz_idx*rs2*cos(10*vec_rad_ang_ind[id][1]), ky_idx*rs1*cos(10*vec_rad_ang_ind[id][1]) - kz_idx*rs2*sin(10*vec_rad_ang_ind[id][1])); // angle
                vec_rad_ang_ind[id][2] = id + 1;
                nb_points_center++;
                
                if (vec_rad_ang_ind[id][1] < r_center) {
                    ind_center[id] = 0;
                    ind_extern[id] = 1;
                    nbCenter++;
                } else {
                    ind_center[id] = 1;
                    ind_extern[id] = 0;
                }
                
                id++;
            }
        }
    }
    
    /**************************************************************
     ** SET ALL PARAMETERS
     *************************************************************/
    //long total_shots    = (long)floor((double)m_lNumSpiralInter / (double)m_lNumConcentricRings);
    //m_lNumSpiralInter   = total_shots*m_lNumConcentricRings;
    nb_spiral   = round_d(((double)m_lNumSpiralInter/(double)(acc*m_lNumConcentricRings)));//number of spiral per repetition/contrast
    // correct spiral rounding errors for extreme case: large number of segments, very asymetric 
    if(nb_spiral * m_lNumConcentricRings * acc > m_lNumSpiralInter)
        nb_spiral = std::max<long>(nb_spiral - 1, 1);
    
    std::vector<long> nb_points(m_lNumConcentricRings);
    std::vector<double> theta(m_lNumConcentricRings,0); // angle for current index for each ring
    std::vector< std::vector< std::vector<long> > > kSpacePolar_Lin(nbRep, std::vector< std::vector<long> > (nb_spiral, std::vector<long> (m_lNumConcentricRings)));
    std::vector< std::vector< std::vector<long> > > kSpacePolar_Par(nbRep, std::vector< std::vector<long> > (nb_spiral, std::vector<long> (m_lNumConcentricRings)));
    std::vector< std::vector<double> > index_ring(m_lNumConcentricRings, std::vector<double>(3));
    
    /**************************************************************
     ** DETERMINE K-SPACE RADIUS AND ANGLE
     *************************************************************/
    
    ring_center = (long)ceil((double)nbCenter/(double)nb_spiral);// number of ring in the center
    ring_extern = m_lNumConcentricRings - ring_center; // number of external ring
    nbExtern    = m_lNumSpiralInter - nbCenter;

    if(isVerbose > 0)
    {
        #if MATLABCALL > 0
            mexPrintf("nb_spiral = %d\n", nb_spiral);
            mexPrintf("ring_center = %d\n", ring_center);
            mexPrintf("ring_extern = %d\n", ring_extern);
            mexPrintf("nbCenter = %d\n", nbCenter);
            mexPrintf("nbExtern = %d\n", nbExtern);
        #else
            printf("nb_spiral = %d\n", nb_spiral);
            printf("ring_center = %d\n", ring_center);
            printf("ring_extern = %d\n", ring_extern);
            printf("nbCenter = %d\n", nbCenter);
            printf("nbExtern = %d\n", nbExtern);
        #endif
    }

    std::vector< std::vector<double> > v_ind_center(nbExtern, std::vector<double>(3));
    std::vector<long> ring_index_extern(ring_extern, 0); // ring index for the external rings
    std::vector<long> v_ind_extern(nbCenter);
    std::vector< std::vector<double> > tmp(m_lNumSpiralInter - nbCenter, std::vector<double>(3));
    
    center = center/10.0; // in percent
    
    sum = 0;
    for(i = ring_center-1 ; i < m_lNumConcentricRings-1 ; ++i) {
        ring_index_extern[i - ring_center + 1] = i + 1;
        sum = sum + pow(i + 1,center);
    }
    
    if(acc > 1.0 && isVariable >= 1.0)
    {
        variability = ((double)m_lNumSpiralInter - (double)nb_spiral*(double)m_lNumConcentricRings)/(double)sum; // Linear increase of acceleration for external rings | m_lNumSpiralInter is if fully acquired
    } else {
        variability = 0.0;
    }
        
    
    /**************************************************************
     ** GET NB_POINTS IN THE CENTER
     *************************************************************/
    
    std::vector<long> SUM(m_lNumConcentricRings);
    
    // 1st RING
    nb_points[0]  = nb_spiral;
    sum           = nb_spiral;
    SUM[0]        = 0;
    for(lRing = 1 ; lRing < ring_center ; ++lRing) {
        nb_points[lRing] = nb_spiral;
        sum             += nb_points[lRing];
        SUM[lRing]       = SUM[lRing-1] + nb_points[lRing-1];// summup(nb_points, 1, lRing, 0);
    }
            
    /**************************************************************
     ** GET NB_POINTS IN THE EACH PERIPHERY RING
     *************************************************************/
    
    // Center RING
    nb_points[ring_center]   = (long)round_d( nb_spiral + (double)variability*pow((double)ring_index_extern[0],center) );
    sum                     += nb_points[ring_center];
    SUM[ring_center]         = 0;
    for(lRing = ring_center + 1 ; lRing < m_lNumConcentricRings ; ++lRing) {
        nb_points[lRing] = (long)round_d( nb_spiral + (double)variability*pow((double)ring_index_extern[lRing - ring_center],center) );
        sum             += nb_points[lRing];
        SUM[lRing]       = SUM[lRing-1] + nb_points[lRing-1];//summup(nb_points, ring_center + 1, lRing, 0);
    }
    
    /**************************************************************
     ** GET NB_POINTS FOR THE LAST RING
     *************************************************************/
    
    nb_points[m_lNumConcentricRings - 1] = m_lNumSpiralInter - (sum - nb_points[m_lNumConcentricRings - 1]);
    
    idx = 0;
    for(i = 0 ; i < m_lNumSpiralInter ; ++i) {
        if (ind_center[i] == 1) {
            v_ind_center[idx] = vec_rad_ang_ind[i];
            idx++;
        }
    }
    
    for(i = 0 ; i < nbExtern ; ++i) {
        tmp[i] = v_ind_center[i];
    }

    #if MATLABCALL > 0
        if(isVerbose > 1.0)
        {
            for(i = 0; i< nb_points.size(); ++i)
            {
                mexPrintf("nb_points[%d] = %d\n", i, nb_points[i]);
            }
        }
    #endif
    
    /**************************************************************
     ** SORT BY RADIUS
     *************************************************************/
    
    sort(tmp.begin(), tmp.end(), &sortColumn1);
    
    /**************************************************************
     ** GET INDEXES IN THE PERIPHERY
     *************************************************************/
    
    idx = 0;
    for(i = 0 ; i < m_lNumSpiralInter ; ++i) {
        if (ind_extern[i] == 1) {
            v_ind_extern[idx] = (long)vec_rad_ang_ind[i][2];
            idx++;
        }
    }
    
    std::vector< std::vector<double> > value_center_sort(nb_spiral*ring_center, std::vector<double>(3));
    
    for(i = 0 ; i < nbCenter ; ++i) {
        value_center_sort[i] = vec_rad_ang_ind[v_ind_extern[i]-1];
    }
    
    for(i = nbCenter ; i < nb_spiral*ring_center ; ++i) {
        value_center_sort[i] = vec_rad_ang_ind[(long)tmp[i-nbCenter][2] - 1];
    }
    
    std::vector< std::vector<double> > value_extern_sort(m_lNumSpiralInter - nb_spiral*ring_center, std::vector<double>(3));
    
    for(i = (nb_spiral*ring_center - nbCenter) ; i < nbExtern ; ++i) {
        value_extern_sort[i-(nb_spiral*ring_center - nbCenter)] = tmp[i];
    }
    
    //sort(value_extern_sort.begin(), value_extern_sort.end(), &sortColumn1); // (DEBUG OK: VALUE_EXTERN_SORT IS SIMILAR TO MATLAB)  
    
    /**************************************************************
     ** BUILD TABLE FOR EACH RING
     *************************************************************/
    std::vector< long > GOLDEN_VECTOR(nb_spiral);
    std::vector< double > GOLDEN_VECTOR_LIST(nb_spiral);
    double gr = (sqrt(5.0)-1.0)/2.0;//0.6180;
    // find out optimal tiny golden angle
    double lr = fFindTinyGolden(nb_spiral);
    //double lr = 1.0/((1.0+sqrt(5.0))/2.0 + 12.0 -1.0);
    double linr = 2 * M_PI / nb_spiral;
    double dGoldenAngleIncr = 0.0;
    
    if (bIsGoldenAngle) {
        dGoldenAngleIncr = 2 * M_PI / nb_spiral;
        
        switch(long(m_dGoldenAngle))
        {
            case 1: // golden - golden
                GOLDEN_VECTOR = generate_golden_angle2((long)nb_spiral, gr, gr);
                GOLDEN_VECTOR_LIST = generate_golden_angle2_list((long)nb_spiral, gr, gr);
                break;
                    
            case 2: // tinyGolden - golden
                GOLDEN_VECTOR = generate_golden_angle2((long)nb_spiral, gr, lr);
                GOLDEN_VECTOR_LIST = generate_golden_angle2_list((long)nb_spiral, gr, lr);
                break;
                
            case 3: // linear - golden
                GOLDEN_VECTOR = generate_golden_angle2((long)nb_spiral, gr, linr);
                GOLDEN_VECTOR_LIST = generate_golden_angle2_list((long)nb_spiral, gr, linr);
                break;
                
            case 4: // no-incr - golden
                GOLDEN_VECTOR = generate_golden_angle2((long)nb_spiral, gr, gr);
                GOLDEN_VECTOR_LIST = generate_golden_angle2_list((long)nb_spiral, gr, gr);
                break;                
        }
                       
    } else {
        dGoldenAngleIncr = 2 * M_PI / nb_spiral;
    }
    
    sort(value_center_sort.begin(), value_center_sort.end(), &sortColumn0);
        
    std::vector< std::vector<std::vector<double> > > index_ring1(m_lNumConcentricRings);
    std::vector< std::vector<std::vector<double> > > index_ring2(m_lNumConcentricRings);
    
    for(lRing = 0 ; lRing < m_lNumConcentricRings ; ++lRing) {
          
            mm_lNumSpiralInter = nb_points[lRing];
            index_ring.resize(mm_lNumSpiralInter);
            
            if (lRing <= (ring_center-1)) {
                
                sum1 = SUM[lRing];
                
                for (k = 0 ; k < mm_lNumSpiralInter ; ++k) {
                    index_ring[k] = value_center_sort[k + sum1]; // here copy value_center_sort[k][0,1,2]
                }
                index_ring1[lRing]=index_ring;
            } else {
                
                sum1 = SUM[lRing];
                sort(value_extern_sort.begin() + sum1, value_extern_sort.begin() + mm_lNumSpiralInter + sum1, &sortColumn0);
                
                for (k = 0 ; k < mm_lNumSpiralInter ; ++k) {
                    index_ring[k] = value_extern_sort[k + sum1]; // here copy value_extern_sort[k][0,1,2]
                }
                index_ring2[lRing]=index_ring;
            } // endif
    }
    
    double dAddonAngle = 0;
    long lAngleIdx = 0;
    
    for (lRep = 0 ; lRep < nbRep ; ++lRep) // repetitions
    {
        theta.clear();
        theta.resize(m_lNumConcentricRings,0);
        for(lSpiralInter = 0 ; lSpiralInter < nb_spiral ; ++lSpiralInter) //nb_spiral
        {        
            for(lRing = 0 ; lRing < m_lNumConcentricRings ; ++lRing)//m_lNumConcentricRings
            {

                //dAddonAngle             =  fmod(dGoldenAngleIncr * lRep, 2.0*M_PI);
                dAddonAngle             =  dGetAddon(lRep, gr * M_PI/nb_spiral, lr * M_PI/nb_spiral, linr, long(m_dGoldenAngle));
                mm_lNumSpiralInter      =  nb_points[lRing];
                lastSpiralPreviousRep   =  theta[lRing];
                dShiftAngle             =  dGoldenAngleIncr + lastSpiralPreviousRep;// + dAddonAngle;
                multipleGoldenAngle     =  fmod(dShiftAngle , 2.0*M_PI); // rem = fmod
                minInd                  =  (long)round_d((double)multipleGoldenAngle/(double)(2.0*M_PI/(double)mm_lNumSpiralInter)) + round_d((double)dAddonAngle/(double)(2.0*M_PI/(double)mm_lNumSpiralInter)) + 1.0;

                if (minInd > mm_lNumSpiralInter) {
                    minInd = minInd - mm_lNumSpiralInter;
                }

                #if MATLABCALL > 0
                    if(isVerbose > 1.0)
                        mexPrintf("lRep = %d, lSpiralInter = %d, lRing = %d, mm_lNumSpiralInter = %d, lastSpiralPreviousRep = %.2f, dShiftAngle = %.2f, multipleGoldenAngle = %.2f, minInd = %d, dAddonAngle = %.2f, ", lRep, lSpiralInter, lRing, mm_lNumSpiralInter, lastSpiralPreviousRep, dShiftAngle, multipleGoldenAngle, minInd, dAddonAngle);
                #endif
                theta[lRing] = dShiftAngle;

                if (lRing <= (ring_center-1)) {
                    index_ring = index_ring1[lRing];
                } else {
                    index_ring = index_ring2[lRing];

                }

                                    
                    lAngleIdx = lSpiralInter + lRep;
                    if(lAngleIdx >= nb_spiral) {
                        lAngleIdx = lAngleIdx - nb_spiral;
                    }
                    #if MATLABCALL > 0
                        if(isVerbose > 1.0)
                            mexPrintf("GOLDEN_VECTOR[%d] = %d, ", lAngleIdx, GOLDEN_VECTOR[lAngleIdx]);
                    #endif

                    kSpacePolar_Lin[lRep][lSpiralInter][lRing] = (long)index_Lin_vec[(long)index_ring[minInd-1][2] - 1];
                    kSpacePolar_Par[lRep][lSpiralInter][lRing] = (long)index_Par_vec[(long)index_ring[minInd-1][2] - 1];

                    #if MATLABCALL > 0
                        if(isVerbose > 1.0)
                            mexPrintf("lin = %d, par = %d\n", kSpacePolar_Lin[lRep][lSpiralInter][lRing], kSpacePolar_Par[lRep][lSpiralInter][lRing]);
                    #endif
            } // end Ring
        } // end spiral      
    } // end repetition

    #if MATLABCALL > 0
        if(isVerbose > 1.0)
            fPlotIndexRing(kSpacePolar_Lin, kSpacePolar_Par, numLin, numPar, m_lNumConcentricRings, nb_spiral, nbRep);
    #endif
        
        //sort(INDEX.begin(), INDEX.end(), &sortColumn0);
        
        /**************************************************************
        ** FINAL ATTRIBUTION
        *************************************************************/
    lAngleIdx = 0;
    long lRepSel = 0;
    long lShift = 0;
    long lAdd = 0;
    long lContSpiralCnt = 0;
        
    std::vector<long> lCurrSpiral_Lin(m_lNumConcentricRings);
	std::vector<long> lCurrSpiral_Par(m_lNumConcentricRings);
    
    std::vector< std::vector < long > > lExist(nbRep, std::vector < long > (nb_spiral,0));
    std::vector<long> lLastSpiral(nbRep,0);
    std::vector<long> GOLDEN_VECTORori = GOLDEN_VECTOR;
    #if MATLABCALL > 0
        if(isVerbose > 1.0)
            mexPrintf("Angle order:\n");
    #endif
    for(lSpiralInter = 0 ; lSpiralInter < nb_spiral ; ++lSpiralInter) //nb_spiral
    { 
        for (lRep = 0 ; lRep < nbRep ; ++lRep)
        {       
            if( !bIsSamePattern )
                lRepSel = lRep;
            
            if (bIsGoldenAngle) {
                if(bIsSamePattern)
                {
                    lAngleIdx = lSpiralInter;
                } 
                if(lExist[lRep][lLimiter(lAngleIdx + lAdd, nb_spiral)] > 0)
                {
                    #if MATLABCALL > 0
                        if(isVerbose > 1.0)
                            mexPrintf("Element already existing!\n");
                    #endif
                    GOLDEN_VECTOR = GOLDEN_VECTORori;
                    lShift = lFindShift(GOLDEN_VECTOR, GOLDEN_VECTOR_LIST, lExist[lLimiter(lRep,nbRep)], nb_spiral, lAngleIdx, lLastSpiral[lLimiter(lRep-1, nbRep)], bIsInOut, bIsSamePattern, m_dGoldenAngle, lr);
                    std::rotate(GOLDEN_VECTOR.begin(), GOLDEN_VECTOR.begin()+lShift, GOLDEN_VECTOR.end());
                    lAdd = lShift;
                    lShift = 0;
                        
                    lAngleIdx = 0; // reset to zero
                }
                
                lCurrSpiral_Lin = kSpacePolar_Lin[lRepSel][GOLDEN_VECTOR[lAngleIdx]-1];
                lCurrSpiral_Par = kSpacePolar_Par[lRepSel][GOLDEN_VECTOR[lAngleIdx]-1];
                #if MATLABCALL > 0
                    if(isVerbose > 1.0)
                        mexPrintf("lRep = %d, lSpiralInter = %d, GOLDEN_VECTOR[%d] = %d, lAdd = %d, lOriAngleIdx = %d,%d\n", lRep, lSpiralInter, lAngleIdx, GOLDEN_VECTOR[lAngleIdx], lAdd, lAngleIdx + lAdd, lLimiter(lAngleIdx + lAdd, nb_spiral));
                #endif
                lExist[lRep][lLimiter(lAngleIdx + lAdd, nb_spiral)] += 1;
                lLastSpiral[lRep] = lLimiter(lAngleIdx + lAdd, nb_spiral);
                //if(lRep == 0)
                //    lLastSpiral = lAngleIdx; // interleaved mode!!! only remember the last spiral which determines the shift
                if(!bIsSamePattern)
                {
                    if(m_dGoldenAngle == 4) // no-incr - golden
                    {
                        if(lContSpiralCnt % 2 != 0)
                            lAngleIdx++;
                    } else {
                        lAngleIdx++;
                    }
                }
                
                if(lAngleIdx >= nb_spiral) { // || (lRep == 0 && lAngleIdx+nbRep >= nb_spiral)) {
                    GOLDEN_VECTOR = GOLDEN_VECTORori;
                    //if(acc > 1.0) { //&& acc >= nbRep) {
                    //if(acc > 1.0 && acc < nbRep) {
                    if(acc > 1.0) { // && nb_spiral % long(nbRep) != 0) { // bugfix for high accelerations: comment out modulo division
                        lShift = lFindShift(GOLDEN_VECTOR, GOLDEN_VECTOR_LIST, lExist[lLimiter(lRep+1,nbRep)], nb_spiral, lAngleIdx, lLastSpiral[lRep], bIsInOut, bIsSamePattern, m_dGoldenAngle, lr);
                        std::rotate(GOLDEN_VECTOR.begin(), GOLDEN_VECTOR.begin()+lShift, GOLDEN_VECTOR.end());
                        lAdd = lShift;
                        lShift = 0;
                        
                        lAngleIdx = 0; // reset to zero
                    } else {
                        lShift = lAngleIdx - nb_spiral;
                        while(lShift < nb_spiral && lExist[lLimiter(lRep+1,nbRep)][lShift] > 0)
                        {
                            lShift++;
                        }
                        if(lShift == nb_spiral)
                            lShift = 0;
                        
                        lAngleIdx = lAngleIdx - nb_spiral + lShift;
                        /*if(acc > 1.0 && acc > nbRep && lExist[lLimiter(lRep+1,nbRep)][lAngleIdx] > 0)
                        {
                            mexPrintf("Element already existing\n");
                            lShift = lFindShift(GOLDEN_VECTOR, GOLDEN_VECTOR_LIST, lExist[lLimiter(lRep+1,nbRep)], nb_spiral, lAngleIdx, lLastSpiral[lRep], isInOut == 1, isSamePattern ==1, isGolden, lr);
                            std::rotate(GOLDEN_VECTOR.begin(), GOLDEN_VECTOR.begin()+lShift, GOLDEN_VECTOR.end());
                            lAdd = lShift;
                            lShift = 0;

                            lAngleIdx = 0; // reset to zero
                        }*/
                    }    
                        /*if(lRep == nbRep-1) {
                            lShift = lmodulo(lShift+1, nbRep);
                        } else {
                            lShift = lAngleIdx - nb_spiral;
                            while(lExist[lRep][lShift] > 0 && lShift <= nb_spiral)
                            {
                                lShift++;
                            }
                            if(lShift == nb_spiral)
                                lShift = 0;
                        }
                    }*/
        
                        //lShift = lmodulo(lShift+1, nbRep);
                    //lAngleIdx = lAngleIdx - nb_spiral + lShift; // +nbRep-1 ensures not the same spiral is selected
                        //lShift++;
                }
            } else {
                lCurrSpiral_Lin = kSpacePolar_Lin[lRepSel][lSpiralInter];
                lCurrSpiral_Par = kSpacePolar_Par[lRepSel][lSpiralInter];
                lExist[lRep][lSpiralInter] = 1;
            }
            if(bIsInOut)
            {
				if (lContSpiralCnt % 2 != 0 )
				{
					std::reverse(lCurrSpiral_Lin.begin(), lCurrSpiral_Lin.end());
					std::reverse(lCurrSpiral_Par.begin(), lCurrSpiral_Par.end());
				}
			}
            if(bIsCenter)
            {
                if(bIsInOut)
                {
                    if(lContSpiralCnt % 2 == 0)
                    {
                        output1[lIndex] = (double)lKSCenterLin;
                        output2[lIndex] = (double)lKSCenterPar;
                        mask[(int)(output1[lIndex] + nbPhase*output2[lIndex])] += 1.0;
                        lIndex++;
                    } 
                } else {
                    output1[lIndex] = (double)lKSCenterLin;
                    output2[lIndex] = (double)lKSCenterPar;
                    mask[(int)(output1[lIndex] + nbPhase*output2[lIndex])] += 1.0;
                    lIndex++;
                }
            }
            for (lRing = 0; lRing<m_lNumConcentricRings; ++lRing)
            {                            
                output1[lIndex] = (double)lCurrSpiral_Lin[lRing];
                output2[lIndex] = (double)lCurrSpiral_Par[lRing];

                mask[(int)(output1[lIndex] + nbPhase*output2[lIndex])] += 1.0;
                lIndex++;
            }
            if(bIsCenter)
            {
                if(bIsInOut)
                {
                    if(lContSpiralCnt % 2 != 0)
                    {
                         // just for debugging purpose to see spirals
                        output1[lIndex] = lKSCenterLin;
                        output2[lIndex] = -1.0;
                        lIndex++;
                    }
                }
            }
            //lAngleIdx++;
            lContSpiralCnt++;
        }
    } // end Spiral

    // show debug output
    #if MATLABCALL > 0
        mexPrintf("Vector sampled\n");
        for (lRep = -1 ; lRep < nbRep ; ++lRep)
        {
            for(lSpiralInter = 0 ; lSpiralInter < nb_spiral ; ++lSpiralInter) //nb_spiral
            {
                if(lRep == -1)
                {
                    mexPrintf("%03d ", lSpiralInter);
                } else {
                    if(lExist[lRep][lSpiralInter] > 0) {
                        mexPrintf(" %02d ", lExist[lRep][lSpiralInter]);
                    } else {
                        mexPrintf(" -  ");
                    }
                }
            }
            mexPrintf("\n");
        }

        // debug outputs
        nSamplesInSpace[0]=(double)nb_points_center;
        nb_spiral_real_out[0] = double(nb_spiral); // return nb_spiral
        n_sampled[0] = double(lIndex);
        //save2img(mask, 120, 72);

        /**************************************************************
         ** DISPLAY SAMPLING PARAMETERS
         *************************************************************/
        if(isVerbose > 0)
        {
            mexPrintf("Number of phase: %d\n", (long)nbPhase);
            mexPrintf("Number of partition: %d\n", (long)nbPartition);
            mexPrintf("Number of segment: %d\n", (long)nbSegment);
            mexPrintf("Number of rep: %d\n", nbRep);
            mexPrintf("Number of spiral: %d\n", nb_spiral);
            mexPrintf("Number of ring: %d\n", m_lNumConcentricRings);
            mexPrintf("Acceleration: %.1f\n", acc);
            mexPrintf("Number of point in the center: %d\n", nbCenter);
            mexPrintf("Number of point in the periphery: %d\n", nbExtern);
            mexPrintf("Total number of point acquired: %d\n", lIndex);
        }
    #else
        nSamplesInSpace[0]=(double)nb_points_center;
        *nb_spiral_real_out = double(nb_spiral); // return nb_spiral
        *n_sampled = double(lIndex);
         if(isVerbose > 0)
        {
            printf("Number of phase: %d\n", (long)nbPhase);
            printf("Number of partition: %d\n", (long)nbPartition);
            printf("Number of segment: %d\n", (long)nbSegment);
            printf("Number of rep: %d\n", nbRep);
            printf("Number of spiral: %d\n", nb_spiral);
            printf("Number of ring: %d\n", m_lNumConcentricRings);
            printf("Acceleration: %.1f\n", acc);
            printf("Number of point in the center: %d\n", nbCenter);
            printf("Number of point in the periphery: %d\n", nbExtern);
            printf("Total number of point acquired: %d\n", lIndex);
        }
    #endif
    /*printf("------------------------------------------------\n");
    printf("---    Variable density sampling Launched    ---\n");
    printf("------------------------------------------------\n");
    printi("Number of phase: ", nbPhase);
    printi("Number of partition: ", nbPartition);
    printi("Number of segment: ", nbSegment);
    printi("Number of spiral: ", nb_spiral);
    printi("Number of ring: ", m_lNumConcentricRings);
    printi("Number of point in the center: ", nbCenter);
    printi("Number of point in the periphery: ", nbExtern);
    printi("Number of point if fully acquired: ", m_lNumSpiralInter);
    printi("Total number of point acquired: ", Total_Acquired);
    printd("Acceleration reached: ", (double)m_lNumSpiralInter/(double)Total_Acquired);*/

}

