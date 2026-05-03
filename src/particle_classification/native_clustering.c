#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int dx;
    int dy;
    double dt;
    double spatial2;
} NeighborOffset;

static int32_t find_root(int32_t *parent, int32_t idx) {
    int32_t root = idx;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[idx] != idx) {
        int32_t next = parent[idx];
        parent[idx] = root;
        idx = next;
    }
    return root;
}

static void union_roots(int32_t *parent, int8_t *rank, int32_t a, int32_t b) {
    int32_t root_a = find_root(parent, a);
    int32_t root_b = find_root(parent, b);
    if (root_a == root_b) {
        return;
    }
    if (rank[root_a] < rank[root_b]) {
        int32_t tmp = root_a;
        root_a = root_b;
        root_b = tmp;
    }
    parent[root_b] = root_a;
    if (rank[root_a] == rank[root_b]) {
        rank[root_a] += 1;
    }
}

static int64_t lower_bound_bucket(
    const int32_t *indices,
    int64_t start,
    int64_t end,
    const double *time,
    double value
) {
    int64_t lo = start;
    int64_t hi = end;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (time[indices[mid]] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static int64_t upper_bound_bucket(
    const int32_t *indices,
    int64_t start,
    int64_t end,
    const double *time,
    double value
) {
    int64_t lo = start;
    int64_t hi = end;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (time[indices[mid]] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static NeighborOffset *make_offsets(double eps, int *n_offsets) {
    int radius = (int)ceil(eps);
    int max_offsets = (2 * radius + 1) * (2 * radius + 1);
    NeighborOffset *offsets = (NeighborOffset *)malloc((size_t)max_offsets * sizeof(NeighborOffset));
    if (offsets == NULL) {
        return NULL;
    }
    double eps2 = eps * eps;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            double spatial2 = (double)(dx * dx + dy * dy);
            if (spatial2 <= eps2 + 1e-12) {
                offsets[count].dx = dx;
                offsets[count].dy = dy;
                offsets[count].spatial2 = spatial2;
                offsets[count].dt = sqrt(fmax(0.0, eps2 - spatial2));
                count++;
            }
        }
    }
    *n_offsets = count;
    return offsets;
}

static int build_pixel_buckets(
    const uint16_t *x,
    const uint16_t *y,
    const int64_t *order,
    int64_t n,
    int64_t *offsets,
    int32_t *bucket_indices
) {
    int64_t *cursor = (int64_t *)calloc(65536, sizeof(int64_t));
    if (cursor == NULL) {
        return -1;
    }
    memset(offsets, 0, (size_t)65537 * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) {
        if (x[i] > 255 || y[i] > 255) {
            free(cursor);
            return -2;
        }
        int pixel = (int)y[i] * 256 + (int)x[i];
        offsets[pixel + 1] += 1;
    }
    for (int pixel = 0; pixel < 65536; pixel++) {
        offsets[pixel + 1] += offsets[pixel];
        cursor[pixel] = offsets[pixel];
    }
    for (int64_t pos = 0; pos < n; pos++) {
        int64_t idx64 = order[pos];
        if (idx64 < 0 || idx64 >= n) {
            free(cursor);
            return -3;
        }
        int32_t idx = (int32_t)idx64;
        int pixel = (int)y[idx] * 256 + (int)x[idx];
        bucket_indices[cursor[pixel]++] = idx;
    }
    free(cursor);
    return 0;
}

static void mark_core_points(
    const uint16_t *x,
    const uint16_t *y,
    const double *time,
    int64_t n,
    const int64_t *bucket_offsets,
    const int32_t *bucket_indices,
    const NeighborOffset *offsets,
    int n_offsets,
    double eps2,
    int min_samples,
    uint8_t *core
) {
    const double tol = 1e-12;
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int64_t i = 0; i < n; i++) {
        int xi = (int)x[i];
        int yi = (int)y[i];
        double ti = time[i];
        int count = 1;
        uint8_t is_core = (min_samples <= 1) ? 1 : 0;
        for (int o = 0; o < n_offsets && !is_core; o++) {
            int nx = xi + offsets[o].dx;
            int ny = yi + offsets[o].dy;
            if (nx < 0 || nx >= 256 || ny < 0 || ny >= 256) {
                continue;
            }
            int pixel = ny * 256 + nx;
            int64_t start = bucket_offsets[pixel];
            int64_t end = bucket_offsets[pixel + 1];
            if (start == end) {
                continue;
            }
            int64_t lo = lower_bound_bucket(bucket_indices, start, end, time, ti - offsets[o].dt - tol);
            int64_t hi = upper_bound_bucket(bucket_indices, start, end, time, ti + offsets[o].dt + tol);
            for (int64_t pos = lo; pos < hi; pos++) {
                int32_t j = bucket_indices[pos];
                if (j == i) {
                    continue;
                }
                double dt = time[j] - ti;
                if (offsets[o].spatial2 + dt * dt <= eps2 + tol) {
                    count++;
                    if (count >= min_samples) {
                        is_core = 1;
                        break;
                    }
                }
            }
        }
        core[i] = is_core;
    }
}

static void init_union_find(int32_t *parent, int8_t *rank, int64_t n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        parent[i] = (int32_t)i;
        rank[i] = 0;
    }
}

static void union_core_pairs(
    const uint16_t *x,
    const uint16_t *y,
    const double *time,
    int64_t n,
    const int64_t *bucket_offsets,
    const int32_t *bucket_indices,
    const NeighborOffset *offsets,
    int n_offsets,
    double eps2,
    const uint8_t *core,
    int32_t *parent,
    int8_t *rank
) {
    const double tol = 1e-12;
    for (int64_t i = 0; i < n; i++) {
        if (!core[i]) {
            continue;
        }
        int xi = (int)x[i];
        int yi = (int)y[i];
        double ti = time[i];
        for (int o = 0; o < n_offsets; o++) {
            int nx = xi + offsets[o].dx;
            int ny = yi + offsets[o].dy;
            if (nx < 0 || nx >= 256 || ny < 0 || ny >= 256) {
                continue;
            }
            int pixel = ny * 256 + nx;
            int64_t lo = lower_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti - offsets[o].dt - tol
            );
            int64_t hi = upper_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti + offsets[o].dt + tol
            );
            for (int64_t pos = lo; pos < hi; pos++) {
                int32_t j = bucket_indices[pos];
                if (j <= i || !core[j]) {
                    continue;
                }
                double dt = time[j] - ti;
                if (offsets[o].spatial2 + dt * dt <= eps2 + tol) {
                    union_roots(parent, rank, (int32_t)i, j);
                }
            }
        }
    }
}

static void assign_dbscan_labels(
    const uint16_t *x,
    const uint16_t *y,
    const double *time,
    int64_t n,
    const int64_t *bucket_offsets,
    const int32_t *bucket_indices,
    const NeighborOffset *offsets,
    int n_offsets,
    double eps2,
    const uint8_t *core,
    int32_t *parent,
    int32_t *labels
) {
    const double tol = 1e-12;
    int32_t *root_label = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (root_label == NULL) {
        for (int64_t i = 0; i < n; i++) {
            labels[i] = -1;
        }
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        root_label[i] = -1;
        labels[i] = -1;
    }
    int32_t next_label = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!core[i]) {
            continue;
        }
        int32_t root = find_root(parent, (int32_t)i);
        if (root_label[root] < 0) {
            root_label[root] = next_label++;
        }
        labels[i] = root_label[root];
    }
    for (int64_t i = 0; i < n; i++) {
        if (core[i]) {
            continue;
        }
        int xi = (int)x[i];
        int yi = (int)y[i];
        double ti = time[i];
        int done = 0;
        for (int o = 0; o < n_offsets && !done; o++) {
            int nx = xi + offsets[o].dx;
            int ny = yi + offsets[o].dy;
            if (nx < 0 || nx >= 256 || ny < 0 || ny >= 256) {
                continue;
            }
            int pixel = ny * 256 + nx;
            int64_t lo = lower_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti - offsets[o].dt - tol
            );
            int64_t hi = upper_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti + offsets[o].dt + tol
            );
            for (int64_t pos = lo; pos < hi; pos++) {
                int32_t j = bucket_indices[pos];
                if (!core[j]) {
                    continue;
                }
                double dt = time[j] - ti;
                if (offsets[o].spatial2 + dt * dt <= eps2 + tol) {
                    labels[i] = labels[j];
                    done = 1;
                    break;
                }
            }
        }
    }
    free(root_label);
}

static void union_all_radius_pairs(
    const uint16_t *x,
    const uint16_t *y,
    const double *time,
    int64_t n,
    const int64_t *bucket_offsets,
    const int32_t *bucket_indices,
    const NeighborOffset *offsets,
    int n_offsets,
    double eps2,
    int32_t *parent,
    int8_t *rank
) {
    const double tol = 1e-12;
    for (int64_t i = 0; i < n; i++) {
        int xi = (int)x[i];
        int yi = (int)y[i];
        double ti = time[i];
        for (int o = 0; o < n_offsets; o++) {
            int nx = xi + offsets[o].dx;
            int ny = yi + offsets[o].dy;
            if (nx < 0 || nx >= 256 || ny < 0 || ny >= 256) {
                continue;
            }
            int pixel = ny * 256 + nx;
            int64_t lo = lower_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti - offsets[o].dt - tol
            );
            int64_t hi = upper_bound_bucket(
                bucket_indices, bucket_offsets[pixel], bucket_offsets[pixel + 1], time, ti + offsets[o].dt + tol
            );
            for (int64_t pos = lo; pos < hi; pos++) {
                int32_t j = bucket_indices[pos];
                if (j <= i) {
                    continue;
                }
                double dt = time[j] - ti;
                if (offsets[o].spatial2 + dt * dt <= eps2 + tol) {
                    union_roots(parent, rank, (int32_t)i, j);
                }
            }
        }
    }
}

static void assign_stream_labels(int64_t n, int min_samples, int32_t *parent, int32_t *labels) {
    int32_t *sizes = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    int32_t *root_label = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (sizes == NULL || root_label == NULL) {
        for (int64_t i = 0; i < n; i++) {
            labels[i] = -1;
        }
        free(sizes);
        free(root_label);
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        root_label[i] = -1;
        labels[i] = -1;
    }
    for (int64_t i = 0; i < n; i++) {
        int32_t root = find_root(parent, (int32_t)i);
        sizes[root] += 1;
    }
    int32_t next_label = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t root = find_root(parent, (int32_t)i);
        if (sizes[root] < min_samples) {
            continue;
        }
        if (root_label[root] < 0) {
            root_label[root] = next_label++;
        }
        labels[i] = root_label[root];
    }
    free(sizes);
    free(root_label);
}

static int run_kernel(
    PyArrayObject *x_arr,
    PyArrayObject *y_arr,
    PyArrayObject *t_arr,
    PyArrayObject *order_arr,
    double eps,
    int min_samples,
    int threads,
    int stream_mode,
    int32_t *labels
) {
    int64_t n = (int64_t)PyArray_DIM(x_arr, 0);
    const uint16_t *x = (const uint16_t *)PyArray_DATA(x_arr);
    const uint16_t *y = (const uint16_t *)PyArray_DATA(y_arr);
    const double *time = (const double *)PyArray_DATA(t_arr);
    const int64_t *order = (const int64_t *)PyArray_DATA(order_arr);
    int status = 0;

    int64_t *bucket_offsets = (int64_t *)malloc((size_t)65537 * sizeof(int64_t));
    int32_t *bucket_indices = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int8_t *rank = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    uint8_t *core = stream_mode ? NULL : (uint8_t *)malloc((size_t)n * sizeof(uint8_t));
    int n_offsets = 0;
    NeighborOffset *neighbor_offsets = make_offsets(eps, &n_offsets);

    if (bucket_offsets == NULL || bucket_indices == NULL || parent == NULL || rank == NULL || neighbor_offsets == NULL || (!stream_mode && core == NULL)) {
        status = -1;
        goto cleanup;
    }

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif

    status = build_pixel_buckets(x, y, order, n, bucket_offsets, bucket_indices);
    if (status != 0) {
        goto cleanup;
    }
    init_union_find(parent, rank, n);
    if (stream_mode) {
        union_all_radius_pairs(x, y, time, n, bucket_offsets, bucket_indices, neighbor_offsets, n_offsets, eps * eps, parent, rank);
        assign_stream_labels(n, min_samples, parent, labels);
    } else {
        mark_core_points(x, y, time, n, bucket_offsets, bucket_indices, neighbor_offsets, n_offsets, eps * eps, min_samples, core);
        union_core_pairs(x, y, time, n, bucket_offsets, bucket_indices, neighbor_offsets, n_offsets, eps * eps, core, parent, rank);
        assign_dbscan_labels(x, y, time, n, bucket_offsets, bucket_indices, neighbor_offsets, n_offsets, eps * eps, core, parent, labels);
    }

cleanup:
    free(bucket_offsets);
    free(bucket_indices);
    free(parent);
    free(rank);
    free(core);
    free(neighbor_offsets);
    return status;
}

static PyObject *run_ordered(PyObject *self, PyObject *args, PyObject *kwargs, int stream_mode) {
    (void)self;
    PyObject *x_obj = NULL;
    PyObject *y_obj = NULL;
    PyObject *t_obj = NULL;
    PyObject *order_obj = NULL;
    double eps = 0.0;
    int min_samples = 0;
    int threads = 0;
    static char *kwlist[] = {"x", "y", "time_scaled", "order", "eps", "min_samples", "threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOdii", kwlist, &x_obj, &y_obj, &t_obj, &order_obj, &eps, &min_samples, &threads)) {
        return NULL;
    }
    PyArrayObject *x_arr = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *t_arr = (PyArrayObject *)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *order_arr = (PyArrayObject *)PyArray_FROM_OTF(order_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (x_arr == NULL || y_arr == NULL || t_arr == NULL || order_arr == NULL) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(t_arr);
        Py_XDECREF(order_arr);
        return NULL;
    }
    if (PyArray_NDIM(x_arr) != 1 || PyArray_NDIM(y_arr) != 1 || PyArray_NDIM(t_arr) != 1 || PyArray_NDIM(order_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "native clustering inputs must be one-dimensional arrays");
        goto error;
    }
    npy_intp n = PyArray_DIM(x_arr, 0);
    if (PyArray_DIM(y_arr, 0) != n || PyArray_DIM(t_arr, 0) != n || PyArray_DIM(order_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "native clustering inputs must have matching lengths");
        goto error;
    }
    if (n > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "native clustering supports at most INT32_MAX hits");
        goto error;
    }
    npy_intp dims[1] = {n};
    PyArrayObject *labels_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (labels_arr == NULL) {
        goto error;
    }
    int32_t *labels = (int32_t *)PyArray_DATA(labels_arr);
    int status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = run_kernel(x_arr, y_arr, t_arr, order_arr, eps, min_samples, threads, stream_mode, labels);
    Py_END_ALLOW_THREADS
    if (status != 0) {
        Py_DECREF(labels_arr);
        if (status == -2) {
            PyErr_SetString(PyExc_ValueError, "native grid clustering requires x/y coordinates in 0..255");
        } else if (status == -3) {
            PyErr_SetString(PyExc_ValueError, "native grid clustering received invalid order indices");
        } else {
            PyErr_SetString(PyExc_MemoryError, "native grid clustering allocation failed");
        }
        goto error;
    }
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(t_arr);
    Py_DECREF(order_arr);
    return (PyObject *)labels_arr;

error:
    Py_XDECREF(x_arr);
    Py_XDECREF(y_arr);
    Py_XDECREF(t_arr);
    Py_XDECREF(order_arr);
    return NULL;
}

static PyObject *grid_dbscan_ordered(PyObject *self, PyObject *args, PyObject *kwargs) {
    return run_ordered(self, args, kwargs, 0);
}

static PyObject *stream_grid_linker_ordered(PyObject *self, PyObject *args, PyObject *kwargs) {
    return run_ordered(self, args, kwargs, 1);
}

static PyMethodDef methods[] = {
    {"grid_dbscan_ordered", (PyCFunction)grid_dbscan_ordered, METH_VARARGS | METH_KEYWORDS, "Exact grid-aware DBSCAN over ordered Timepix hits."},
    {"stream_grid_linker_ordered", (PyCFunction)stream_grid_linker_ordered, METH_VARARGS | METH_KEYWORDS, "Approximate stream/grid radius linker over ordered Timepix hits."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native_clustering",
    "Native OpenMP clustering kernels for particle_classification.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__native_clustering(void) {
    import_array();
    return PyModule_Create(&module);
}
