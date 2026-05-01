/**
 * bindings.cpp – PyBind11 module definition for cv_backend.
 *
 * Exposes all thresholding and segmentation C++ functions to Python.
 *
 * Python-side usage (after `pip install -e .`):
 *
 *   import cv_backend
 *
 *   # Thresholding (returns py::tuple – consumed by core/thresholding.py)
 *   img_bin, thresh = cv_backend.threshold_optimal(float32_image, tol=1/255)
 *   img_bin, thresh = cv_backend.threshold_otsu(float32_image)
 *   img_lab, threshs = cv_backend.threshold_spectral(float32_image, n_classes=3)
 *   img_bin         = cv_backend.threshold_local(float32_image, block_size=35, offset=10/255)
 *
 *   # Segmentation (returns float32 label image)
 *   labels = cv_backend.segment_kmeans(float32_image, k=4, max_iter=100)
 *   labels = cv_backend.segment_region_growing(float32_image, seed_row, seed_col, tol)
 *   labels = cv_backend.segment_agglomerative(float32_image, k=4)
 *   labels = cv_backend.segment_mean_shift(float32_image, bandwidth=30/255)
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
        py::arg("seed_row"),
        py::arg("seed_col"),
        py::arg("tolerance") = 15.0f / 255.0f,
        R"doc(
Region growing segmentation from a seed pixel.

Parameters
----------
image : np.ndarray
    Float32 grayscale image, shape (H, W), values in [0, 1].
seed_row : int
    Row index of the seed pixel.
seed_col : int
    Column index of the seed pixel.
tolerance : float, optional
    Max intensity difference from seed (default 15/255).

Returns
-------
np.ndarray
    Float32 binary mask, shape (H, W).  1.0 = region, 0.0 = background.
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
