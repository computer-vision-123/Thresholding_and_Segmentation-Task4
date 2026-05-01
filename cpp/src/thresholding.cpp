/**
 * thresholding.cpp – Part A algorithm stubs.
 *
 * Each function currently raises std::runtime_error("Not implemented").
 * Replace the throw with the actual algorithm body during implementation.
 *
 * Input contract (enforced by the Python wrapper in core/thresholding.py):
 *   - image is a 2-D float32 C-contiguous array, values in [0, 1].
 */

#include "thresholding.hpp"

#include <stdexcept>
#include <vector>

#include "thresholding.hpp"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace thresholding {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> compute_histogram(const py::detail::unchecked_reference<float, 2>& buf, int bins = 256) {
    std::vector<float> hist(bins, 0.0f);
    ssize_t H = buf.shape(0);
    ssize_t W = buf.shape(1);
    float scale = static_cast<float>(bins - 1);
    for (ssize_t r = 0; r < H; ++r) {
        for (ssize_t c = 0; c < W; ++c) {
            int bin = static_cast<int>(std::clamp(buf(r, c) * scale, 0.0f, scale));
            hist[bin] += 1.0f;
        }
    }
    float total = static_cast<float>(H * W);
    for (float& val : hist) val /= total;
    return hist;
}

static ImageF32 binarize(const py::detail::unchecked_reference<float, 2>& buf, float threshold) {
    ssize_t H = buf.shape(0);
    ssize_t W = buf.shape(1);
    ImageF32 result(std::vector<ssize_t>{H, W});
    auto out = result.mutable_unchecked<2>();
    for (ssize_t r = 0; r < H; ++r) {
        for (ssize_t c = 0; c < W; ++c) {
            out(r, c) = (buf(r, c) > threshold) ? 1.0f : 0.0f;
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimal (Iterative) Thresholding
// ─────────────────────────────────────────────────────────────────────────────

py::tuple optimal(ImageF32 image, float tol)
{
    auto buf = image.unchecked<2>();
    ssize_t H = buf.shape(0);
    ssize_t W = buf.shape(1);

    if (H == 0 || W == 0) return py::make_tuple(ImageF32(std::vector<ssize_t>{0, 0}), 0.0f);

    float T = 0.5f; // Initial guess
    float T_old;

    const int max_iters = 100;
    int iter = 0;

    do {
        T_old = T;
        double sum0 = 0, sum1 = 0;
        size_t n0 = 0, n1 = 0;

        for (ssize_t r = 0; r < H; ++r) {
            for (ssize_t c = 0; c < W; ++c) {
                float v = buf(r, c);
                if (v <= T) {
                    sum0 += v;
                    n0++;
                } else {
                    sum1 += v;
                    n1++;
                }
            }
        }

        float mu0 = (n0 > 0) ? static_cast<float>(sum0 / n0) : 0.0f;
        float mu1 = (n1 > 0) ? static_cast<float>(sum1 / n1) : 1.0f;
        T = 0.5f * (mu0 + mu1);
        iter++;
    } while (std::abs(T - T_old) > tol && iter < max_iters);

    return py::make_tuple(binarize(buf, T), T);
}

// ─────────────────────────────────────────────────────────────────────────────
// Otsu's Method
// ─────────────────────────────────────────────────────────────────────────────

py::tuple otsu(ImageF32 image)
{
    auto buf = image.unchecked<2>();
    auto hist = compute_histogram(buf);

    float best_sigma = -1.0f;
    int best_t = 0;

    float m_global = 0.0f;
    for (int i = 0; i < 256; ++i) m_global += i * hist[i];

    float w0 = 0.0f;
    float m0 = 0.0f;

    for (int t = 0; t < 256; ++t) {
        w0 += hist[t];
        if (w0 <= 0.0f) continue;
        float w1 = 1.0f - w0;
        if (w1 <= 0.0f) break;

        m0 += t * hist[t];
        float mu0 = m0 / w0;
        float mu1 = (m_global - m0) / w1;

        float sigma_b = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if (sigma_b > best_sigma) {
            best_sigma = sigma_b;
            best_t = t;
        }
    }

    float final_threshold = static_cast<float>(best_t) / 255.0f;
    return py::make_tuple(binarize(buf, final_threshold), final_threshold);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral (Multi-Otsu) Thresholding
// ─────────────────────────────────────────────────────────────────────────────

py::tuple spectral(ImageF32 image, int n_classes)
{
    if (n_classes < 3)
        throw std::invalid_argument("spectral thresholding requires n_classes >= 3");

    auto buf = image.unchecked<2>();
    auto hist = compute_histogram(buf);

    // Dynamic Programming for Multi-Otsu
    // Maximize Sum(w_i * mu_i^2)
    
    std::vector<double> P(256), S(256);
    P[0] = hist[0]; S[0] = 0;
    for (int i = 1; i < 256; ++i) {
        P[i] = P[i-1] + hist[i];
        S[i] = S[i-1] + i * hist[i];
    }

    auto get_P = [&](int i, int j) {
        if (i > j) return 0.0;
        return (i == 0) ? P[j] : P[j] - P[i-1];
    };
    auto get_S = [&](int i, int j) {
        if (i > j) return 0.0;
        return (i == 0) ? S[j] : S[j] - S[i-1];
    };
    auto get_var_term = [&](int i, int j) {
        double p = get_P(i, j);
        if (p <= 0) return 0.0;
        double s = get_S(i, j);
        return (s * s) / p;
    };

    // dp[k][m] = max variance using k classes for bins [0...m]
    // backtrace[k][m] = the start of the k-th class
    std::vector<std::vector<double>> dp(n_classes, std::vector<double>(256, -1.0));
    std::vector<std::vector<int>> backtrace(n_classes, std::vector<int>(256, 0));

    // Base case: 1 class
    for (int m = 0; m < 256; ++m) {
        dp[0][m] = get_var_term(0, m);
    }

    // DP transitions
    for (int k = 1; k < n_classes; ++k) {
        for (int m = k; m < 256; ++m) {
            for (int j = k - 1; j < m; ++j) {
                double val = dp[k-1][j] + get_var_term(j + 1, m);
                if (val > dp[k][m]) {
                    dp[k][m] = val;
                    backtrace[k][m] = j + 1;
                }
            }
        }
    }

    // Backtrack to find thresholds
    std::vector<int> t_indices(n_classes - 1);
    int curr_m = 255;
    for (int k = n_classes - 1; k > 0; --k) {
        int start = backtrace[k][curr_m];
        t_indices[k-1] = start - 1;
        curr_m = start - 1;
    }

    std::vector<float> thresholds;
    for (int idx : t_indices) {
        thresholds.push_back(static_cast<float>(idx) / 255.0f);
    }

    ssize_t H = buf.shape(0), W = buf.shape(1);
    ImageF32 label_img(std::vector<ssize_t>{H, W});
    auto out = label_img.mutable_unchecked<2>();
    for (ssize_t r = 0; r < H; ++r) {
        for (ssize_t c = 0; c < W; ++c) {
            float v = buf(r, c);
            int label = n_classes - 1;
            for (int i = 0; i < n_classes - 1; ++i) {
                if (v <= thresholds[i]) {
                    label = i;
                    break;
                }
            }
            out(r, c) = static_cast<float>(label);
        }
    }

    return py::make_tuple(label_img, thresholds);
}


// ─────────────────────────────────────────────────────────────────────────────
// Local (Adaptive) Thresholding
// ─────────────────────────────────────────────────────────────────────────────

ImageF32 local(ImageF32 image, int block_size, float offset)
{
    if (block_size % 2 == 0)
        throw std::invalid_argument("block_size must be odd");

    auto buf = image.unchecked<2>();
    ssize_t H = buf.shape(0), W = buf.shape(1);

    // Integral image for fast box sum
    std::vector<double> integral((H + 1) * (W + 1), 0.0);
    for (ssize_t r = 0; r < H; ++r) {
        double row_sum = 0;
        for (ssize_t c = 0; c < W; ++c) {
            row_sum += buf(r, c);
            integral[(r + 1) * (W + 1) + (c + 1)] = integral[r * (W + 1) + (c + 1)] + row_sum;
        }
    }

    ImageF32 result(std::vector<ssize_t>{H, W});
    auto out = result.mutable_unchecked<2>();
    int half = block_size / 2;

    for (ssize_t r = 0; r < H; ++r) {
        for (ssize_t c = 0; c < W; ++c) {
            ssize_t r1 = std::max<ssize_t>(0, r - half);
            ssize_t r2 = std::min<ssize_t>(H - 1, r + half);
            ssize_t c1 = std::max<ssize_t>(0, c - half);
            ssize_t c2 = std::min<ssize_t>(W - 1, c + half);

            double sum = integral[(r2 + 1) * (W + 1) + (c2 + 1)] -
                         integral[r1 * (W + 1) + (c2 + 1)] -
                         integral[(r2 + 1) * (W + 1) + c1] +
                         integral[r1 * (W + 1) + c1];

            double count = (r2 - r1 + 1) * (c2 - c1 + 1);
            float local_mean = static_cast<float>(sum / count);

            out(r, c) = (buf(r, c) > (local_mean - offset)) ? 1.0f : 0.0f;
        }
    }

    return result;
}

}  // namespace thresholding

