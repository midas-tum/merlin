#define MATLABCALL 0

#include <stdio.h>
#include <Python.h>
#include <thread>
#include "sampling.cpp"

static PyObject *GenError;


/* This is the main function */
PyObject* py_VD_CASPR(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3, *arg4, *arg5;
	Py_buffer parameter_list, mask, outMatrix1, outMatrix2, out_parameter; //  , n_SamplesInSpace, n_sampled, nb_spiral;

    //int max_thread_num = -1;

    /* get arguments */
	if (!PyArg_ParseTuple(args, "OOOOO|d", &arg1, &arg2, &arg3, &arg4, &arg5 ))
		return NULL;

    /* copy input and output in buffer */
	if (PyObject_GetBuffer(arg1, &parameter_list, PyBUF_FULL_RO) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg2, &mask, PyBUF_FULL) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg3, &outMatrix1, PyBUF_FULL) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg4, &outMatrix2, PyBUF_FULL) < 0)
		return NULL;
    if (PyObject_GetBuffer(arg5, &out_parameter, PyBUF_FULL) < 0)
		return NULL;

    float* param_array = (float*) parameter_list.buf;
    //std::cout << "param[0]=" << (int)param_array[0] << std::endl;
    double nbPhase = (double) param_array[0];
	double nbPartition = (double) param_array[1];
	double acc = (double) param_array[2];
	double center = (double) param_array[3];
	double nbSegment = (double) param_array[4];
	double nbRep = (double) param_array[5];
	double isGolden = (double) param_array[6];
	double isVariable = (double) param_array[7];
    double isInOut = (double) param_array[8];
    double isCenter = (double) param_array[9];
    double isSamePattern = (double) param_array[10];
    double isVerbose = (double) param_array[11];
    double dinita, dinitb, dinitc = 0.0;
    double* n_SamplesInSpace = &dinita;
    double* n_sampled = &dinitb;
    double* nb_spiral = &dinitc;

    /* Call Main Sampling Function */
    sampling_mex(nbPhase, nbPartition, acc, center, nbSegment, nbRep, isGolden, isVariable, isInOut, isCenter, isSamePattern, isVerbose,
                    (double *) outMatrix1.buf, (double *) outMatrix2.buf, (double *) mask.buf, (double *) n_SamplesInSpace, (double *) n_sampled, (double *) nb_spiral);

	//int results = VD_CASPR((double *)nbPhase.buf, (double *)b_imgin.buf, Nx, Ny);
	//PyObject* res = PyLong_FromLong(results);

	float* param_out_array = (float*) out_parameter.buf;
	param_out_array[0] = (float) *n_SamplesInSpace;
	param_out_array[1] = (float) *n_sampled;
	param_out_array[2] = (float) *nb_spiral;
	int results = 0;
	PyObject* res = PyLong_FromLong(results);

	/* Release buffers*/
	//PyBuffer_Release(&parameter_list);
	//PyBuffer_Release(&mask);
	//PyBuffer_Release(&outMatrix1);
	//PyBuffer_Release(&outMatrix2);

	return res;
}


// Here VD_CASPR is how you will call the function in python
PyMethodDef VD_CASPR_Methods[] = {
	{ "run", (PyCFunction)py_VD_CASPR, METH_VARARGS, "run(parameter_list, lMask, kSpacePolar_LinInd, kSpacePolar_ParInd)" },
	{ NULL, NULL, 0, NULL }
};


/* Module definition structure */
static struct PyModuleDef VD_CASPR_module = {
	PyModuleDef_HEAD_INIT,
	"VD_CASPR",            /* module name */
	"VD_CASPR Module C++", /* documentation */
	-1,
	VD_CASPR_Methods
};


/* Initialization function */
PyMODINIT_FUNC PyInit_VD_CASPR_CINE(void)
{
	PyObject *m = PyModule_Create(&VD_CASPR_module);

	if (m == NULL)
		return NULL;

	GenError = PyErr_NewException("VD_CASPR.error", NULL, NULL);
	Py_INCREF(GenError);
	PyModule_AddObject(m, "error", GenError);

	return m;

}


