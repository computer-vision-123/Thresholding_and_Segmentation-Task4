/**
 * bindings.cpp – PyBind11 module definition for cv_backend.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "thresholding.hpp"
#include "segmentation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cv_backend, m)
{
    m.doc() = "cv_backend – C++ image processing kernels for CV Task 4.";

    // ── Part A: Thresholding ─────────────────────────────────────────────────

    m.def(
        "threshold_optimal",
        &thresholding::optimal,
        py::arg("image"),
        py::arg("tol") = 1.0f / 255.0f,
        R"doc(
Optimal (iterative) thresholding.

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].
tol : float, optional
    Convergence tolerance (default 1/255).

Returns
-------
tuple[np.ndarray, float]
    (binary_image, threshold_value)
        )doc"
    );

    m.def(
        "threshold_otsu",
        &thresholding::otsu,
        py::arg("image"),
        R"doc(
Otsu's global thresholding (maximises between-class variance).

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].

Returns
-------
tuple[np.ndarray, float]
    (binary_image, threshold_value)
        )doc"
    );

    m.def(
        "threshold_spectral",
        &thresholding::spectral,
        py::arg("image"),
        py::arg("n_classes") = 3,
        R"doc(
Spectral (multi-Otsu) thresholding for n_classes >= 3.

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].
n_classes : int, optional
    Number of output intensity classes (default 3).

Returns
-------
tuple[np.ndarray, list[float]]
    (label_image, list_of_thresholds)
        )doc"
    );

    m.def(
        "threshold_local",
        &thresholding::local,
        py::arg("image"),
        py::arg("block_size") = 35,
        py::arg("offset")     = 10.0f / 255.0f,
        R"doc(
Local (adaptive) thresholding using local neighbourhood mean.

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].
block_size : int, optional
    Odd window size (default 35).
offset : float, optional
    Offset subtracted from local mean (default 10/255).

Returns
-------
np.ndarray
    Binary image (float32, 0.0 / 1.0), shape (H, W).
        )doc"
    );

    // ── Part B: Segmentation ─────────────────────────────────────────────────

    m.def(
        "segment_kmeans",
        &segmentation::kmeans,
        py::arg("image"),
        py::arg("k")        = 4,
        py::arg("max_iter") = 100,
        R"doc(
K-Means clustering segmentation.

Parameters
----------
image : np.ndarray
    Float32 image, shape (H, W) or (H, W, 3), values in [0, 1].
k : int, optional
    Number of clusters (default 4).
max_iter : int, optional
    Maximum EM iterations (default 100).

Returns
-------
np.ndarray
    Float32 label image, shape (H, W), values in {0 … k−1}.
        )doc"
    );

    m.def(
        "segment_region_growing",
        &segmentation::region_growing,
        py::arg("image"),
        py::arg("seed_rows"),
        py::arg("seed_cols"),
        py::arg("tolerance") = 15.0f / 255.0f,
        R"doc(
Multi-seed region growing segmentation.

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].
seed_rows : list[int]
    Row indices of seed pixels.
seed_cols : list[int]
    Column indices of seed pixels.
tolerance : float, optional
    Max intensity difference from each seed's value (default 15/255).

Returns
-------
np.ndarray
    Float32 label image, shape (H, W).
    0 = background, 1..N = region for seed N.
        )doc"
    );

    m.def(
        "segment_agglomerative",
        &segmentation::agglomerative,
        py::arg("image"),
        py::arg("k") = 4,
        R"doc(
Agglomerative (hierarchical) clustering segmentation.

Parameters
----------
image : np.ndarray
    Float32 image, shape (H, W) or (H, W, 3), values in [0, 1].
k : int, optional
    Target number of clusters (default 4).

Returns
-------
np.ndarray
    Float32 label image, shape (H, W), values in {0 … k−1}.
        )doc"
    );

    m.def(
        "segment_mean_shift",
        &segmentation::mean_shift,
        py::arg("image"),
        py::arg("bandwidth") = 30.0f / 255.0f,
        R"doc(
Mean Shift segmentation.

Parameters
----------
image : np.ndarray
    Float32 image, shape (H, W) or (H, W, 3), values in [0, 1].
bandwidth : float, optional
    Kernel bandwidth (default 30/255).

Returns
-------
np.ndarray
    Float32 label image, shape (H, W).
        )doc"
    );
}