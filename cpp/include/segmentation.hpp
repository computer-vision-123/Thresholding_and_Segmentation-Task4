#pragma once

/**
 * segmentation.hpp – Declarations for Part B algorithms.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

using ImageF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace segmentation {

/**
 * K-Means Clustering Segmentation.
 */
ImageF32 kmeans(ImageF32 image, int k = 4, int max_iter = 100);

/**
 * Multi-Seed Region Growing Segmentation.
 *
 * Algorithm (parallel BFS):
 *   1. Enqueue all seed pixels simultaneously with their respective labels.
 *   2. BFS expands each seed's region; a pixel is claimed by the FIRST seed
 *      that reaches it whose intensity difference from that seed's value is
 *      within tolerance.
 *   3. Pixels unreachable by any seed are assigned label 0 (background).
 *
 * @param image      Float32 grayscale image, shape (H, W), values in [0, 1].
 * @param seed_rows  Row indices of seed pixels (one per seed).
 * @param seed_cols  Column indices of seed pixels (one per seed).
 * @param tolerance  Max allowed intensity difference from a seed's value.
 * @return           Float32 label image, shape (H, W).
 *                   0 = background, 1..N = region for seed N.
 */
ImageF32 region_growing(ImageF32 image,
                        std::vector<int> seed_rows,
                        std::vector<int> seed_cols,
                        float tolerance = 15.0f / 255.0f);

/**
 * Agglomerative (Hierarchical) Clustering Segmentation.
 */
ImageF32 agglomerative(ImageF32 image, int k = 4);

/**
 * Mean Shift Segmentation.
 */
ImageF32 mean_shift(ImageF32 image, float bandwidth = 30.0f / 255.0f);

}  // namespace segmentation