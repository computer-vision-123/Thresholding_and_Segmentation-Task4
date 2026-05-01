#pragma once

/**
 * segmentation.hpp – Declarations for Part B algorithms.
 *
 * Convention
 * ----------
 * All functions accept images as float32 numpy arrays (values in [0, 1]).
 * Colour images have shape (H, W, 3); grayscale images have shape (H, W).
 * The Python binding layer (core/segmentation.py) handles uint8→float32.
 *
 * All functions return a float32 label image of shape (H, W) where each
 * unique value identifies a distinct segment (0, 1, 2, …, k-1).
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using ImageF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace segmentation {

/**
 * K-Means Clustering Segmentation.
 *
 * Algorithm:
 *   1. Reshape image to (N, C) feature matrix (C=1 gray, C=3 colour).
 *   2. Initialise k centroids (k-means++ strategy).
 *   3. Assign each pixel to nearest centroid (Euclidean distance).
 *   4. Recompute centroids; repeat until convergence or max_iter.
 *
 * @param image    Float32 image, shape (H, W) or (H, W, 3).
 * @param k        Number of clusters.
 * @param max_iter Maximum EM iterations.
 * @return         Float32 label image, shape (H, W), values in {0 … k−1}.
 */
ImageF32 kmeans(ImageF32 image, int k = 4, int max_iter = 100);

/**
 * Region Growing Segmentation.
 *
 * Algorithm (BFS):
 *   1. Initialise a visited mask and a FIFO queue.
 *   2. Enqueue the seed pixel; mark as visited.
 *   3. Dequeue a pixel → add to region; for each 4-connected neighbour:
 *      if |pixel − seed_intensity| ≤ tolerance and not visited → enqueue.
 *   4. Return binary mask (1.0 = region, 0.0 = background).
 *
 * @param image      Float32 grayscale image, shape (H, W).
 * @param seed_row   Row index of the seed pixel.
 * @param seed_col   Column index of the seed pixel.
 * @param tolerance  Maximum allowed intensity difference from seed.
 * @return           Float32 binary mask, shape (H, W).
 */
ImageF32 region_growing(ImageF32 image,
                        int seed_row,
                        int seed_col,
                        float tolerance = 15.0f / 255.0f);

/**
 * Agglomerative (Hierarchical) Clustering Segmentation.
 *
 * Algorithm:
 *   1. Build superpixels (SLIC) to reduce number of initial nodes.
 *   2. Start with each superpixel as its own cluster.
 *   3. Iteratively merge the two closest clusters (Ward linkage) until k
 *      clusters remain.
 *
 * @param image  Float32 image, shape (H, W) or (H, W, 3).
 * @param k      Target number of clusters.
 * @return       Float32 label image, shape (H, W), values in {0 … k−1}.
 */
ImageF32 agglomerative(ImageF32 image, int k = 4);

/**
 * Mean Shift Segmentation.
 *
 * Algorithm:
 *   1. Represent each pixel as a feature vector.
 *   2. For each pixel, centre a kernel window and shift it to the local mean
 *      iteratively until convergence.
 *   3. Pixels whose windows converge to the same mode share a label.
 *
 * @param image      Float32 image, shape (H, W) or (H, W, 3).
 * @param bandwidth  Kernel bandwidth (radius in feature space).
 * @return           Float32 label image, shape (H, W).
 */
ImageF32 mean_shift(ImageF32 image, float bandwidth = 30.0f / 255.0f);

}  // namespace segmentation
