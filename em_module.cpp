#include <Python.h>
#include <numpy/arrayobject.h>

#define PY_ARRAY_UNIQUE_SYMBOL 

#include "EM.h"


#include <algorithm>
#include <vector>
#include <iostream>

using std::vector;

vector<vector<vector<double> > > get_X(PyObject* X) {
    double*** X_ptr;
    npy_intp dims[3];
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_AsCArray(&X, (void ***)&X_ptr, dims, 3, descr);
    vector<vector<vector<double> > > result(dims[0], vector<vector<double> >(dims[1], vector<double>(dims[2])));
    std::cout << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                result[i][j][k] = X_ptr[i][j][k];
            }
        }
    }
    return result;
};

vector<vector<double> > get_2d_arr(PyObject* arr) {
    double** arr_ptr;
    npy_intp dims[2];
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_AsCArray(&arr, &arr_ptr, dims, 2, descr);
    vector<vector<double> > result(dims[0], vector<double>(dims[1]));
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            result[i][j] = arr_ptr[i][j];
        }
    }
    return result;
};

PyArrayObject* convert_to_np_array(vector<vector<double> > A) {
    npy_intp dims[2];
    dims[0] = A.size();
    dims[1] = A[0].size();
    double* flat_A = new double[dims[0] * dims[1]];
    for (int i = 0; i < dims[0]; ++i) {
       for (int j = 0; j < dims[1]; ++j) {
           flat_A[i*dims[1] + j] = A[i][j];
       }
    }

    PyArrayObject* res = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, flat_A);
    res->flags |= NPY_ARRAY_OWNDATA;
    // delete[] flat_A;
    return res;
}


static PyObject* em_run_em(PyObject* self, PyObject* args) {

    std::cout << "init!" << std::endl;

    PyObject *arg_X = NULL, *arg_F = NULL, *arg_B = NULL, *arg_A = NULL;
    double s;
    int n;
    int h, w;

    if (!PyArg_ParseTuple(args, "OiiOOdOi", &arg_X, &h, &w, &arg_F,
        &arg_B, &s, &arg_A, &n)){
        std::cout << "parsing error!" << std::endl;
        return NULL;
    }

    vector<vector<vector<double> > > X_ptr = get_X(arg_X);
    vector<vector<double> > F_ptr = get_2d_arr(arg_F);
    vector<vector<double> > B_ptr = get_2d_arr(arg_B);
    vector<vector<double> > A_ptr = get_2d_arr(arg_A);

    std::cout << "running EM!" << std::endl;
    run_EM(X_ptr, h, w, &F_ptr, &B_ptr, &s, &A_ptr, n);

    PyArrayObject* np_F = convert_to_np_array(F_ptr);
    PyArrayObject* np_B = convert_to_np_array(B_ptr);
    PyArrayObject* np_A = convert_to_np_array(A_ptr);
    
    return Py_BuildValue("OOdO", np_F, np_B, s, np_A);
};

static PyMethodDef EMMethods[] = {
    { "run_em", em_run_em, METH_VARARGS, "Restore picture and background." },
    { NULL, NULL, 0, NULL }        /* Sentinel */
};


/*
static bool dfs(PyObject* graph, Py_ssize_t current, vector<bool>& visited) {
    visited[current] = true;
    PyObject* all_neighbours = PyList_GetItem(graph, current);

    if (!PyList_Check(all_neighbours)) {
        PyErr_SetString(PyExc_TypeError, "Invalid graph.");
        return false;
    }

    for (Py_ssize_t i = 0; i < PyList_Size(all_neighbours); ++i) {
        PyObject* neighbour_obj = PyList_GetItem(all_neighbours, i);
        if (!PyInt_Check(neighbour_obj)) {
            PyErr_SetString(PyExc_TypeError, "Invalid graph.");
            return false;
        }
        Py_ssize_t neighbour = PyInt_AsSsize_t(neighbour_obj);

        if (!visited[neighbour]) {
            if (!dfs(graph, neighbour, visited)) {
                return false;
            }
        }
    }

    return true;
}

static PyObject* dfs_is_connected(PyObject* self, PyObject* args) {
    PyObject* graph;
    int start;
    if (!PyArg_ParseTuple(args, "Oi", &graph, &start)) {
        return NULL;
    }

    if (!PyList_Check(graph)) {
        PyErr_SetString(PyExc_TypeError, "Invalid graph.");
        return NULL;
    }

    Py_ssize_t vertexes_count = PyList_Size(graph);
    vector<bool> visited(vertexes_count, false);
    if (!dfs(graph, start, visited)) {
        return NULL;
    }

    for (size_t i = 0; i < visited.size(); ++i) {
        if (!visited[i]) {
            return Py_False;
        }
    }
    return Py_True;
}

static PyMethodDef DFSMethods[] = {
    { "is_connected", dfs_is_connected, METH_VARARGS,
    "Check graph connectivity." },
    { NULL, NULL, 0, NULL }        
};
*/
PyMODINIT_FUNC initem(void) {
    import_array();
    Py_InitModule("em", EMMethods);
}
