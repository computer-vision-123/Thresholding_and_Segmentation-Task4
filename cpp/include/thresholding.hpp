#pragma once

/**
 * thresholding.hpp – Declarations for Part A algorithms.
 *
 * Convention
 * ----------
 * All functions accept images as float32 numpy arrays (values in [0, 1]).
 * The Python binding layer (core/thresholding.py) is responsible for
 * converting uint8 input to float32 before calling into C++.
 *
 * Return values are py::tuple; the Python wrappers convert them to dataclasses.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Convenience alias
using ImageF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace thresholding {

/**
 * Optimal (Iterative) Thresholding.
 *
 * Algorithm:
 *   1. T₀ = mean(image)
 *   2. G1 = pixels ≤ T,  G2 = pixels > T
 *   3. T_new = (mean(G1) + mean(G2)) / 2
 *   4. Repeat until |T_new − T_old| < tol
 *
 * @param image      Float32 grayscale image, values in [0, 1], shape (H, W).
 * @param tol        Convergence tolerance (in normalised [0,1] units).
 * @return           py::tuple { binary_image (float32, 0.0/1.0), threshold (float) }
 */
py::tuple optimal(ImageF32 image, float tol = 1.0f / 255.0f);

/**
 * Otsu's Global Thresholding.
 *
 * Algorithm:
 *   Sweep all candidate thresholds t in [0, 1]; select t that maximises
 *   between-class variance: σ²_B = w₀·w₁·(μ₀−μ₁)²
 *
 * @param image      Float32 grayscale image, values in [0, 1], shape (H, W).
 * @return           py::tuple { binary_image (float32, 0.0/1.0), threshold (float) }
 */
py::tuple otsu(ImageF32 image);

/**
 * Spectral (Multi-modal / Multi-Otsu) Thresholding.
 *
 * Extends Otsu to n_classes > 2 by finding (n_classes−1) thresholds that
 * jointly maximise total between-class variance.
 *
 * @param image      Float32 grayscale image, values in [0, 1], shape (H, W).
 * @param n_classes  Number of output intensity classes (must be ≥ 3).
 * @return           py::tuple { label_image (float32, class indices), thresholds (vector<float>) }
 */
py::tuple spectral(ImageF32 image, int n_classes = 3);

/**
 * Local (Adaptive) Thresholding.
 *
 * Per-pixel threshold = local neighbourhood mean − offset.
 *
 * @param image       Float32 grayscale image, values in [0, 1], shape (H, W).
 * @param block_size  Odd integer; size of the local neighbourhood window.
 * @param offset      Constant subtracted from the local mean (normalised).
 * @return            Binary image (float32, 0.0/1.0), shape (H, W).
 */
ImageF32 local(ImageF32 image, int block_size = 35, float offset = 10.0f / 255.0f);

}  // namespace thresholding
