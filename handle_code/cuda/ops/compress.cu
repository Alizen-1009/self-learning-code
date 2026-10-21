// out[i] = sum_j softmax(score[:, i])[j] * kv[j, i]
// kv and score are contiguous row-major float32 tensors of shape [128, 512].
// One CTA owns 32 independent output columns. Its four warps partition the
// 128 rows, then combine four stable (max, denominator, numerator) triples.
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr int kRows = 128;
constexpr int kCols = 512;
constexpr int kColsPerBlock = 32;
constexpr int kRowParts = 4;
}  // namespace

__global__ void compress_kernel(const float* __restrict__ kv,
                                const float* __restrict__ score,
                                float* __restrict__ out) {
  __shared__ float partial_max[kRowParts][kColsPerBlock];
  __shared__ float partial_den[kRowParts][kColsPerBlock];
  __shared__ float partial_num[kRowParts][kColsPerBlock];

  const int col_in_block = threadIdx.x;
  const int row_part = threadIdx.y;
  const int col = blockIdx.x * kColsPerBlock + col_in_block;

  float m = -CUDART_INF_F;
  float den = 0.0f;
  float num = 0.0f;
  for (int row = row_part; row < kRows; row += kRowParts) {
    const int offset = row * kCols + col;
    const float s = score[offset];
    if (s == -CUDART_INF_F) continue;  // Optional masked row.
    const float next_m = fmaxf(m, s);
    const float scale = __expf(m - next_m);
    const float weight = __expf(s - next_m);
    den = den * scale + weight;
    num = num * scale + weight * kv[offset];
    m = next_m;
  }

  partial_max[row_part][col_in_block] = m;
  partial_den[row_part][col_in_block] = den;
  partial_num[row_part][col_in_block] = num;
  __syncthreads();

  if (row_part == 0) {
    float merged_m = -CUDART_INF_F;
#pragma unroll
    for (int part = 0; part < kRowParts; ++part)
      merged_m = fmaxf(merged_m, partial_max[part][col_in_block]);

    if (merged_m == -CUDART_INF_F) {
      out[col] = CUDART_NAN_F;  // Matches softmax of an all-masked column.
      return;
    }

    float merged_den = 0.0f;
    float merged_num = 0.0f;
#pragma unroll
    for (int part = 0; part < kRowParts; ++part) {
      const float scale = __expf(partial_max[part][col_in_block] - merged_m);
      merged_den += partial_den[part][col_in_block] * scale;
      merged_num += partial_num[part][col_in_block] * scale;
    }
    out[col] = merged_num / merged_den;
  }
}

// Call with device pointers; the caller owns allocation and synchronization.
cudaError_t compress(const float* kv, const float* score, float* out,
                     cudaStream_t stream = nullptr) {
  static_assert(kCols % kColsPerBlock == 0);
  const dim3 threads(kColsPerBlock, kRowParts);
  const dim3 blocks(kCols / kColsPerBlock);
  compress_kernel<<<blocks, threads, 0, stream>>>(kv, score, out);
  return cudaGetLastError();
}
