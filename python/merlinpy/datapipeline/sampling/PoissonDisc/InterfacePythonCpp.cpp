#define PYTHONCALL 1

#include <stdio.h>
#include <Python.h>
#include <thread>
//#include "SubsampleMain.h"
#include "Point.cpp"
#include "LinkedList.cpp"
#include "SomeFunctions.cpp"
#include "VariableDensity.cpp"
#include "VDSamplingUpper.cpp"
#include "Approx.cpp"
#include "SubsampleMain.cpp"

static PyObject *GenError;


/* This is the main function */
PyObject* py_VDPD(PyObject *self, PyObject *args)
{
	PyObject *arg1, *arg2, *arg3, *arg4, *arg5;
	Py_buffer parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd, out_parameter; //  , n_SamplesInSpace, n_sampled, nb_spiral;

    //int max_thread_num = -1;

    /* get arguments */
	if (!PyArg_ParseTuple(args, "OOOOO|d", &arg1, &arg2, &arg3, &arg4, &arg5 ))
		return NULL;

    /* copy input and output in buffer */
	if (PyObject_GetBuffer(arg1, &parameter_list, PyBUF_FULL_RO) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg2, &kSpacePolar_LinInd, PyBUF_FULL) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg3, &kSpacePolar_ParInd, PyBUF_FULL) < 0)
		return NULL;
	if (PyObject_GetBuffer(arg4, &phaseInd, PyBUF_FULL) < 0)
		return NULL;
    if (PyObject_GetBuffer(arg5, &out_parameter, PyBUF_FULL) < 0)
		return NULL;

    float* param_array = (float*) parameter_list.buf;
    //std::cout << "param[0]=" << (int)param_array[0] << std::endl;
    long nX = (long) param_array[0];
    long nY = (long) param_array[1];
    double accels = (double) param_array[2];
    short int subsampleType = (short int) param_array[3];
    float fully_sampled = (float) param_array[4];
    float pF_value = (float) param_array[5];
    bool pF_x = (bool) param_array[6];
    int lPhases = (int) param_array[7];
    short int vd_type = (short int) param_array[8];
    short int smpl_type = (short int) param_array[9];
    bool ellip_mask = (bool) param_array[10];
    float p = (float) param_array[11];
    float n = (float) param_array[12];
    short int body_region = (short int) param_array[13];
    float iso_fac = (float) param_array[14];
    bool lVerbose = (bool) param_array[15];

    /* Call Main Sampling Function */
    run_subsampling((double *) kSpacePolar_LinInd.buf, (double *) kSpacePolar_ParInd.buf, (double *) phaseInd.buf,
                  nX, nY, accels, subsampleType, fully_sampled, pF_value, pF_x, lPhases, vd_type, smpl_type, ellip_mask, p, n, body_region, iso_fac, lVerbose);

	float* param_out_array = (float*) out_parameter.buf;
	param_out_array[0] = (float) 0;
	param_out_array[1] = (float) 0;
	param_out_array[2] = (float) 0;
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
PyMethodDef VDPD_Methods[] = {
	{ "run", (PyCFunction)py_VDPD, METH_VARARGS, "run(parameter_list, kSpacePolar_LinInd, kSpacePolar_ParInd, phaseInd)" },
	{ NULL, NULL, 0, NULL }
};


/* Module definition structure */
static struct PyModuleDef VDPD_module = {
	PyModuleDef_HEAD_INIT,
	"VDPD",            /* module name */
	"VDPD Module C++", /* documentation */
	-1,
	VDPD_Methods
};


/* Initialization function */
PyMODINIT_FUNC PyInit_VDPD(void)
{
	PyObject *m = PyModule_Create(&VDPD_module);

	if (m == NULL)
		return NULL;

	GenError = PyErr_NewException("VDPD.error", NULL, NULL);
	Py_INCREF(GenError);
	PyModule_AddObject(m, "error", GenError);

	return m;

}


