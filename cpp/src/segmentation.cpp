#include "segmentation.hpp"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <algorithm>
#include <numeric>

namespace segmentation {

static float squared_distance(const float* a, const float* b, int C)
{
    float dist = 0.0f;
    for (int c = 0; c < C; ++c) {
        float diff = a[c] - b[c];
        dist += diff * diff;
    }
    return dist;
}


static std::vector<float> flatten_image(const py::buffer_info& info,
                                        int& N, int& C)
{
    const int H = static_cast<int>(info.shape[0]);
    const int W = static_cast<int>(info.shape[1]);
    C = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
    N = H * W;
 
    const float* src = static_cast<const float*>(info.ptr);
    std::vector<float> features(static_cast<size_t>(N) * C);
 
    if (C == 1) {
        // Grayscale: direct copy (already contiguous)
        std::copy(src, src + N, features.begin());
    } else {
        // Colour: src is already HxWx3 C-contiguous → layout matches target
        std::copy(src, src + N * C, features.begin());
    }
 
    return features;
}

ImageF32 kmeans(ImageF32 image, int k, int max_iter)
{
     if (k < 2)
        throw std::invalid_argument("k must be >= 2");
 
    // ── 1. Flatten image to feature matrix (N x C) 
    py::buffer_info info = image.request();
    int N, C;
    std::vector<float> features = flatten_image(info, N, C);
 
    const int H = static_cast<int>(info.shape[0]);
    const int W = static_cast<int>(info.shape[1]);
 
    if (N < k)
        throw std::invalid_argument("k must be <= number of pixels");
 
    // ── 2. K-Means++ initialisation ───────────────────────────────────────────
 
    std::mt19937 rng(42); // fixed seed → reproducible results
 
    // centroids[j] = feature vector of centroid j, length C
    std::vector<std::vector<float>> centroids(k, std::vector<float>(C, 0.0f));
 
    // Pick first centroid uniformly at random
    {
        std::uniform_int_distribution<int> uni(0, N - 1);
        int first_idx = uni(rng);
        for (int c = 0; c < C; ++c)
            centroids[0][c] = features[static_cast<size_t>(first_idx) * C + c];
    }
 
    // Squared-distance buffer (one entry per pixel)
    std::vector<float> min_dist_sq(N, std::numeric_limits<float>::max());
 
    for (int j = 1; j < k; ++j) {
        // Update min_dist_sq using the centroid added in the previous iteration
        const float* prev = centroids[j - 1].data();
        for (int i = 0; i < N; ++i) {
            float d = squared_distance(&features[static_cast<size_t>(i) * C], prev, C);
            if (d < min_dist_sq[i])
                min_dist_sq[i] = d;
        }
 
        // Sample next centroid proportional to min_dist_sq
        std::discrete_distribution<int> weighted(min_dist_sq.begin(),
                                                  min_dist_sq.end());
        int next_idx = weighted(rng);
        for (int c = 0; c < C; ++c)
            centroids[j][c] = features[static_cast<size_t>(next_idx) * C + c];
    }
 
    // ── 3. EM loop ────────────────────────────────────────────────────────────
    std::vector<int> labels(N, 0);
 
    for (int iter = 0; iter < max_iter; ++iter) {

        bool any_changed = false;
 
        for (int i = 0; i < N; ++i) {
            float  best_dist  = std::numeric_limits<float>::max();
            int    best_label = 0;
 
            for (int j = 0; j < k; ++j) {
                float d = squared_distance(
                    &features[static_cast<size_t>(i) * C],
                    centroids[j].data(),
                    C
                );
                if (d < best_dist) {
                    best_dist  = d;
                    best_label = j;
                }
            }
 
            if (best_label != labels[i]) {
                labels[i] = best_label;
                any_changed = true;
            }
        }
 
        // ── 3b. Early stopping ────────────────────────────────────────────────
        if (!any_changed)
            break;
 
        // ── 3c. Update step ───────────────────────────────────────────────────
        std::vector<std::vector<double>> sums(k, std::vector<double>(C, 0.0));
        std::vector<int>                 counts(k, 0);
 
        for (int i = 0; i < N; ++i) {
            int j = labels[i];
            counts[j]++;
            for (int c = 0; c < C; ++c)
                sums[j][c] += features[static_cast<size_t>(i) * C + c];
        }
 
        std::uniform_int_distribution<int> uni(0, N - 1);
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int c = 0; c < C; ++c)
                    centroids[j][c] = static_cast<float>(sums[j][c] / counts[j]);
            } else {
                // Reinitialise empty cluster to a random pixel
                int rand_idx = uni(rng);
                for (int c = 0; c < C; ++c)
                    centroids[j][c] = features[static_cast<size_t>(rand_idx) * C + c];
            }
        }
    }
 
    // ── 4. Build output label image (H x W), float32 ─────────────────────────
    ImageF32 label_img(std::vector<ssize_t>{H, W});
    auto out = label_img.mutable_unchecked<2>();
 
    for (int r = 0; r < H; ++r)
        for (int w = 0; w < W; ++w)
            out(r, w) = static_cast<float>(labels[r * W + w]);
 
    return label_img;
}

// ─────────────────────────────────────────────────────────────────────────────
// Region Growing
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 region_growing(ImageF32 image, int seed_row, int seed_col, float tolerance)
{
    // TODO: implement BFS region growing
    //
    // Suggested implementation sketch:
    //   auto buf = image.unchecked<2>();
    //   ssize_t H = buf.shape(0), W = buf.shape(1);
    //   py::array_t<float> result({H, W});   // zero-initialised
    //   std::vector<std::vector<bool>> visited(H, std::vector<bool>(W, false));
    //   float seed_val = buf(seed_row, seed_col);
    //
    //   std::queue<std::pair<int,int>> q;
    //   q.push({seed_row, seed_col});
    //   visited[seed_row][seed_col] = true;
    //
    //   const int dr[] = {-1, 1, 0, 0};
    //   const int dc[] = { 0, 0,-1, 1};
    //   while (!q.empty()) {
    //       auto [r, c] = q.front(); q.pop();
    //       result.mutable_at(r, c) = 1.0f;
    //       for (int d = 0; d < 4; ++d) {
    //           int nr = r+dr[d], nc = c+dc[d];
    //           if (nr>=0 && nr<H && nc>=0 && nc<W && !visited[nr][nc]
    //               && std::abs(buf(nr,nc) - seed_val) <= tolerance) {
    //               visited[nr][nc] = true;
    //               q.push({nr, nc});
    //           }
    //       }
    //   }
    //   return result;

    throw std::runtime_error("segmentation::region_growing – Not implemented");
}

// ─────────────────────────────────────────────────────────────────────────────
// Agglomerative Clustering
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 agglomerative(ImageF32 image, int k)
{
    if (k < 2)
        throw std::invalid_argument("k must be >= 2");

    // TODO: implement agglomerative clustering
    //
    // Suggested implementation sketch (with superpixel pre-pass):
    //   1. (Optional) Run SLIC superpixel segmentation to reduce N from H*W to
    //      ~200-500 superpixels. Each superpixel is represented by its mean
    //      feature vector.
    //   2. Build a complete distance matrix (Ward linkage).
    //   3. Greedily merge the closest pair of clusters until k remain.
    //   4. Map each pixel to its final cluster label.
    //
    // For small images or if SLIC is skipped, work directly on pixels but be
    // aware of O(N²) memory cost.

    throw std::runtime_error("segmentation::agglomerative – Not implemented");
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean Shift
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 mean_shift(ImageF32 image, float bandwidth)
{
    if (bandwidth <= 0.0f)
        throw std::invalid_argument("bandwidth must be > 0");

    // TODO: implement Mean Shift
    //
    // Suggested implementation sketch:
    //   1. Flatten image to (N, C) feature matrix.
    //   2. For each pixel i, start a window centred at feature[i]:
    //        while (shift_magnitude > epsilon):
    //            neighbours = all points within bandwidth of current centre
    //            new_centre  = mean(neighbours)
    //            shift       = new_centre - current_centre
    //            current_centre = new_centre
    //   3. Converged centres that are within bandwidth of each other → same label.
    //   4. Assign each pixel the label of its converged centre.
    //
    // Optimisation: use a spatial index (grid or k-d tree) to speed up
    // neighbour lookups from O(N) to O(log N) per iteration.

    throw std::runtime_error("segmentation::mean_shift – Not implemented");
}

}  // namespace segmentation
