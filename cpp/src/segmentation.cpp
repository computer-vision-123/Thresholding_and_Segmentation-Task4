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

ImageF32 region_growing(ImageF32 image,
                        std::vector<int> seed_rows,
                        std::vector<int> seed_cols,
                        float tolerance)
{
    if (seed_rows.empty() || seed_rows.size() != seed_cols.size())
        throw std::invalid_argument("seed_rows and seed_cols must be non-empty and equal length");
 
    auto buf = image.unchecked<2>();
    ssize_t H = buf.shape(0);
    ssize_t W = buf.shape(1);
 
    // Validate seeds
    int num_seeds = static_cast<int>(seed_rows.size());
    for (int s = 0; s < num_seeds; ++s) {
        if (seed_rows[s] < 0 || seed_rows[s] >= H ||
            seed_cols[s] < 0 || seed_cols[s] >= W)
            throw std::invalid_argument("Seed pixel out of image bounds");
    }
 
    // Label image: -1 = unvisited, 0 = background (unreachable), 1..N = seed labels
    // We store label as float starting from 1 for seed 0, 2 for seed 1, etc.
    std::vector<int> label_map(H * W, -1);
 
    // BFS queue holds (row, col, seed_index)
    std::queue<std::tuple<int,int,int>> q;
 
    // Seed values for each seed
    std::vector<float> seed_vals(num_seeds);
 
    for (int s = 0; s < num_seeds; ++s) {
        int r = seed_rows[s];
        int c = seed_cols[s];
        seed_vals[s] = buf(r, c);
 
        // Only enqueue if not already claimed by an earlier seed
        if (label_map[r * W + c] == -1) {
            label_map[r * W + c] = s + 1;  // labels start at 1
            q.push({r, c, s});
        }
    }
 
    const int dr[] = {-1, 1, 0, 0};
    const int dc[] = { 0, 0,-1, 1};
 
    while (!q.empty()) {
        auto [r, c, s] = q.front();
        q.pop();
 
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d];
            int nc = c + dc[d];
 
            if (nr < 0 || nr >= H || nc < 0 || nc >= W)
                continue;
            if (label_map[nr * W + nc] != -1)
                continue;
            if (std::abs(buf(nr, nc) - seed_vals[s]) <= tolerance) {
                label_map[nr * W + nc] = s + 1;
                q.push({nr, nc, s});
            }
        }
    }
 
    // Build output: unvisited pixels get label 0 (background)
    ImageF32 result(std::vector<ssize_t>{H, W});
    auto out = result.mutable_unchecked<2>();
 
    for (ssize_t r = 0; r < H; ++r) {
        for (ssize_t c = 0; c < W; ++c) {
            int lbl = label_map[r * W + c];
            out(r, c) = static_cast<float>(lbl < 0 ? 0 : lbl);
        }
    }
 
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Agglomerative Clustering
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 agglomerative(ImageF32 image, int k)
{
    if (k < 2)
        throw std::invalid_argument("k must be >= 2");
 
    // ── 0. Unpack image dimensions ────────────────────────────────────────────
    py::buffer_info info = image.request();
    const int H = static_cast<int>(info.shape[0]);
    const int W = static_cast<int>(info.shape[1]);
    const int C = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
    const float* src = static_cast<const float*>(info.ptr);
 
    // ── 1. Choose cell size so the grid has at most ~600 cells total ──────────
    const int max_cells = 600;
    int cell = std::max(1, static_cast<int>(std::floor(
        std::sqrt(static_cast<double>(H * W) / max_cells)
    )));
 
    const int grid_h = (H + cell - 1) / cell;   // number of cell rows
    const int grid_w = (W + cell - 1) / cell;   // number of cell cols
    const int N      = grid_h * grid_w;          // total super-pixels
 
    if (N < k)
        throw std::invalid_argument(
            "k is larger than the number of super-pixels; "
            "use a smaller k or a larger image."
        );
 
    // ── 2. Compute mean feature vector for each cell ──────────────────────────
    std::vector<double> means(static_cast<size_t>(N) * C, 0.0);
    std::vector<int>    sizes(N, 0);
 
    for (int gr = 0; gr < grid_h; ++gr) {
        for (int gc = 0; gc < grid_w; ++gc) {
            const int cell_id = gr * grid_w + gc;
 
            // Pixel rows/cols covered by this cell (clamp to image bounds)
            const int r_start = gr * cell;
            const int r_end   = std::min(r_start + cell, H);
            const int c_start = gc * cell;
            const int c_end   = std::min(c_start + cell, W);
 
            for (int r = r_start; r < r_end; ++r) {
                for (int c = c_start; c < c_end; ++c) {
                    for (int ch = 0; ch < C; ++ch) {
                        // src layout: (r, c, ch) in C-contiguous order
                        means[static_cast<size_t>(cell_id) * C + ch]
                            += src[(r * W + c) * C + ch];
                    }
                    sizes[cell_id]++;
                }
            }
 
            // Normalise to get the actual mean
            for (int ch = 0; ch < C; ++ch)
                means[static_cast<size_t>(cell_id) * C + ch]
                    /= static_cast<double>(sizes[cell_id]);
        }
    }
 
    // ── 3. Build the upper-triangle distance matrix ───────────────────────────
    std::vector<float> D(static_cast<size_t>(N) * N,
                          std::numeric_limits<float>::max());
 
    for (int i = 0; i < N; ++i) {
        D[static_cast<size_t>(i) * N + i] = std::numeric_limits<float>::max(); // diagonal = ∞
 
        for (int j = i + 1; j < N; ++j) {
            float dist = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                float diff = static_cast<float>(
                    means[static_cast<size_t>(i) * C + ch] -
                    means[static_cast<size_t>(j) * C + ch]
                );
                dist += diff * diff;
            }
            D[static_cast<size_t>(i) * N + j] = dist;
            D[static_cast<size_t>(j) * N + i] = dist; // keep symmetric
        }
    }
 
    // ── 4. Agglomerative merging (average linkage) ────────────────────────────
    std::vector<int>  cluster_id(N);
    std::iota(cluster_id.begin(), cluster_id.end(), 0); // cluster_id[i] = i initially
 
    std::vector<bool> active(N, true);
    std::vector<int>  cluster_size(sizes.begin(), sizes.end()); // pixel counts per cluster
    int n_active = N;
 
    while (n_active > k) {
 
        // 4a. Find the closest pair of active clusters  (brute-force O(N²))
        float best_dist = std::numeric_limits<float>::max();
        int   best_i = -1, best_j = -1;
 
        for (int i = 0; i < N; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < N; ++j) {
                if (!active[j]) continue;
                float d = D[static_cast<size_t>(i) * N + j];
                if (d < best_dist) {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
 
        // Safety check – should never trigger for valid input
        if (best_i < 0)
            break;
 
        // 4b. Merge cluster j into cluster i  (i is the survivor)
        const int sz_i = cluster_size[best_i];
        const int sz_j = cluster_size[best_j];
        const int sz_new = sz_i + sz_j;
 
        // Update the mean of cluster i (weighted average)
        for (int ch = 0; ch < C; ++ch) {
            means[static_cast<size_t>(best_i) * C + ch] =
                (means[static_cast<size_t>(best_i) * C + ch] * sz_i +
                 means[static_cast<size_t>(best_j) * C + ch] * sz_j)
                / static_cast<double>(sz_new);
        }
        cluster_size[best_i] = sz_new;
 
        // 4c. Update row/column best_i using average-linkage formula
        for (int m = 0; m < N; ++m) {
            if (!active[m] || m == best_i || m == best_j) continue;
 
            float d_im = D[static_cast<size_t>(best_i) * N + m];
            float d_jm = D[static_cast<size_t>(best_j) * N + m];
 
            float d_new = (static_cast<float>(sz_i) * d_im +
                           static_cast<float>(sz_j) * d_jm)
                          / static_cast<float>(sz_new);
 
            D[static_cast<size_t>(best_i) * N + m] = d_new;
            D[static_cast<size_t>(m)      * N + best_i] = d_new;
        }
 
        // 4d. Deactivate cluster j; redirect its cells to cluster i
        active[best_j] = false;
        for (int i = 0; i < N; ++i)
            if (cluster_id[i] == best_j)
                cluster_id[i] = best_i;
 
        --n_active;
    }
 
    // ── 5. Relabel active clusters to consecutive ids 0, 1, …, k-1 ───────────
    std::unordered_map<int, int> label_map;
    int next_label = 0;
    for (int i = 0; i < N; ++i) {
        int cid = cluster_id[i];
        if (label_map.find(cid) == label_map.end())
            label_map[cid] = next_label++;
        cluster_id[i] = label_map[cid];
    }
 
    // ── 6. Project cell labels back to individual pixels ─────────────────────
    ImageF32 label_img(std::vector<ssize_t>{H, W});
    auto out = label_img.mutable_unchecked<2>();
 
    for (int gr = 0; gr < grid_h; ++gr) {
        for (int gc = 0; gc < grid_w; ++gc) {
            const int cell_id    = gr * grid_w + gc;
            const float lbl      = static_cast<float>(cluster_id[cell_id]);
 
            const int r_start = gr * cell;
            const int r_end   = std::min(r_start + cell, H);
            const int c_start = gc * cell;
            const int c_end   = std::min(c_start + cell, W);
 
            for (int r = r_start; r < r_end; ++r)
                for (int c = c_start; c < c_end; ++c)
                    out(r, c) = lbl;
        }
    }
 
    return label_img;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean Shift
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 mean_shift(ImageF32 image, float bandwidth)
{
    if (bandwidth <= 0.0f)
        throw std::invalid_argument("bandwidth must be > 0");
 
    // ── 0. Unpack image ───────────────────────────────────────────────────────
    py::buffer_info info = image.request();
    const int H = static_cast<int>(info.shape[0]);
    const int W = static_cast<int>(info.shape[1]);
    const int C = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
    const float* src = static_cast<const float*>(info.ptr);
 
    // ── 1. Build super-pixel grid ─────────────────────────────────────────────
    const int MAX_POINTS = 1200;
    const int cell = std::max(1, static_cast<int>(
        std::floor(std::sqrt(static_cast<double>(H * W) / MAX_POINTS))
    ));
 
    const int grid_h = (H + cell - 1) / cell;
    const int grid_w = (W + cell - 1) / cell;
    const int N      = grid_h * grid_w;
 
    // features[i*C + c] = mean of channel c for super-pixel i
    std::vector<float> features(static_cast<size_t>(N) * C, 0.0f);
    std::vector<int>   pix_count(N, 0);
 
    for (int gr = 0; gr < grid_h; ++gr) {
        for (int gc = 0; gc < grid_w; ++gc) {
            const int id      = gr * grid_w + gc;
            const int r_start = gr * cell;
            const int r_end   = std::min(r_start + cell, H);
            const int c_start = gc * cell;
            const int c_end   = std::min(c_start + cell, W);
 
            for (int r = r_start; r < r_end; ++r)
                for (int c = c_start; c < c_end; ++c)
                    for (int ch = 0; ch < C; ++ch)
                        features[static_cast<size_t>(id) * C + ch]
                            += src[(r * W + c) * C + ch];
 
            pix_count[id] = (r_end - r_start) * (c_end - c_start);
            for (int ch = 0; ch < C; ++ch)
                features[static_cast<size_t>(id) * C + ch]
                    /= static_cast<float>(pix_count[id]);
        }
    }
 
    // ── 2. Mean-shift iteration for every super-pixel ────────────────────────
    const float bw_sq  = bandwidth * bandwidth;
    const float epsilon = 1e-5f;
    const int max_iter  = 100;
 
    std::vector<float> modes(static_cast<size_t>(N) * C);
    std::vector<float> centre(C);
 
    for (int i = 0; i < N; ++i) {
 
        // Initialise the moving centre at this super-pixel's feature vector
        for (int ch = 0; ch < C; ++ch)
            centre[ch] = features[static_cast<size_t>(i) * C + ch];
 
        for (int iter = 0; iter < max_iter; ++iter) {
 
            // Accumulate the mean of all points within bandwidth
            std::vector<double> sum(C, 0.0);
            int count = 0;
 
            for (int j = 0; j < N; ++j) {
                float dist_sq = 0.0f;
                for (int ch = 0; ch < C; ++ch) {
                    float diff = features[static_cast<size_t>(j) * C + ch]
                                 - centre[ch];
                    dist_sq += diff * diff;
                }
                if (dist_sq <= bw_sq) {
                    for (int ch = 0; ch < C; ++ch)
                        sum[ch] += features[static_cast<size_t>(j) * C + ch];
                    ++count;
                }
            }
 
            // Shift centre to the mean; check convergence
            float shift_sq = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                float next = (count > 0)
                             ? static_cast<float>(sum[ch] / count)
                             : centre[ch];
                float diff = next - centre[ch];
                shift_sq  += diff * diff;
                centre[ch] = next;
            }
 
            if (shift_sq < epsilon * epsilon)
                break;
        }
 
        // Store converged centre as the mode for this super-pixel
        for (int ch = 0; ch < C; ++ch)
            modes[static_cast<size_t>(i) * C + ch] = centre[ch];
    }
 
    // ── 3. Merge nearby modes (union-find) ────────────────────────────────────
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0);
 
    std::function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x)
            parent[x] = find(parent[x]);   // path compression
        return parent[x];
    };
 
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[b] = a;
    };
 
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float dist_sq = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                float diff = modes[static_cast<size_t>(i) * C + ch]
                             - modes[static_cast<size_t>(j) * C + ch];
                dist_sq += diff * diff;
            }
            if (dist_sq <= bw_sq)
                unite(i, j);
        }
    }
 
    // ── 4. Relabel roots to consecutive 0, 1, … ──────────────────────────────
    std::unordered_map<int, int> label_map;
    int next_label = 0;
 
    std::vector<int> cell_label(N);
    for (int i = 0; i < N; ++i) {
        int root = find(i);
        if (label_map.find(root) == label_map.end())
            label_map[root] = next_label++;
        cell_label[i] = label_map[root];
    }
 
    // ── 5. Back-project cell labels to every original pixel ──────────────────
    ImageF32 label_img(std::vector<ssize_t>{H, W});
    auto out = label_img.mutable_unchecked<2>();
 
    for (int gr = 0; gr < grid_h; ++gr) {
        for (int gc = 0; gc < grid_w; ++gc) {
            const float lbl   = static_cast<float>(cell_label[gr * grid_w + gc]);
            const int r_start = gr * cell;
            const int r_end   = std::min(r_start + cell, H);
            const int c_start = gc * cell;
            const int c_end   = std::min(c_start + cell, W);
            for (int r = r_start; r < r_end; ++r)
                for (int c = c_start; c < c_end; ++c)
                    out(r, c) = lbl;
        }
    }
 
    return label_img;
}

}  // namespace segmentation
