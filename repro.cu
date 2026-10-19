// ptxas miscompiles a 128-bit Karatsuba carry-less product when one operand is the constant one.
//
// mul() forms the 256-bit carry-less product of two 128-bit values with three 64x64 products
// (Karatsuba) on clmad.lo/hi.u64, then folds it to 128 bits. The kernel computes the running
// product e = 1, e = mul(e, a[t]) over N inputs and stores every step. Since the product's high half
// is zero when one factor is one, mul(1, a[0]) must equal a[0]. The host recomputes every step.
#include <cstdint>
#include <cstdio>

#ifndef N
#define N 14
#endif

struct u128 { uint64_t lo, hi; };

__device__ __forceinline__ void clmul(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) {
  asm("clmad.lo.u64 %0, %1, %2, 0;" : "=l"(lo) : "l"(a), "l"(b));
  asm("clmad.hi.u64 %0, %1, %2, 0;" : "=l"(hi) : "l"(a), "l"(b));
}

__device__ __forceinline__ u128 mul(u128 a, u128 b) {
  uint64_t l0, l1, h0, h1, m0, m1;
  clmul(a.lo, b.lo, l0, l1);
  clmul(a.hi, b.hi, h0, h1);
  clmul(a.lo ^ a.hi, b.lo ^ b.hi, m0, m1);
  m0 ^= l0 ^ h0;
  m1 ^= l1 ^ h1;
  // The 256-bit product is r3:r2:r1:r0.
  uint64_t r0 = l0, r1 = l1 ^ m0, r2 = h0 ^ m1, r3 = h1;
  return {r0 ^ (r2 << 1), r1 ^ (r3 << 1) ^ (r2 >> 63)};
}

__global__ void k(const u128 *a, u128 *out) {
  u128 e = {1, 0};
  for (int t = 0; t < N; t++) {
    e = mul(e, a[t]);
    out[t] = e;
  }
}

// The same computation on the host, with a bitwise carry-less multiply.
static void clmul_ref(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) {
  lo = hi = 0;
  for (int i = 0; i < 64; i++)
    if ((b >> i) & 1) {
      lo ^= a << i;
      if (i) hi ^= a >> (64 - i);
    }
}
static u128 mul_ref(u128 a, u128 b) {
  uint64_t l0, l1, h0, h1, x0, x1, y0, y1;
  clmul_ref(a.lo, b.lo, l0, l1);
  clmul_ref(a.hi, b.hi, h0, h1);
  clmul_ref(a.lo, b.hi, x0, x1);
  clmul_ref(a.hi, b.lo, y0, y1);
  uint64_t r0 = l0, r1 = l1 ^ x0 ^ y0, r2 = h0 ^ x1 ^ y1, r3 = h1;
  return {r0 ^ (r2 << 1), r1 ^ (r3 << 1) ^ (r2 >> 63)};
}

#define CHECK(call)                                                                                \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err));                                 \
      return 2;                                                                                    \
    }                                                                                              \
  } while (0)

// Exits 0 when every step matches the host, 1 when one doesn't and 2 on a CUDA error.
int main() {
  u128 h[N], r[N];
  h[0] = {0x7d49bf6b24b0c5ecull, 0x8b5868123167fa23ull};
  uint64_t z = 0;
  for (int t = 1; t < N; t++) {
    uint64_t w[2];
    for (int j = 0; j < 2; j++) {  // splitmix64
      uint64_t v = z += 0x9e3779b97f4a7c15ull;
      v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
      v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
      w[j] = v ^ (v >> 31);
    }
    h[t] = {w[0], w[1]};
  }
  u128 *a, *o;
  CHECK(cudaMalloc(&a, sizeof h));
  CHECK(cudaMalloc(&o, sizeof r));
  CHECK(cudaMemcpy(a, h, sizeof h, cudaMemcpyHostToDevice));
  k<<<1, 1>>>(a, o);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(r, o, sizeof r, cudaMemcpyDeviceToHost));
  int wrong = 0;
  u128 e = {1, 0};
  for (int t = 0; t < N; t++) {
    e = mul_ref(e, h[t]);
    if (r[t].lo != e.lo || r[t].hi != e.hi) {
      printf("step %2d: got %016llx %016llx, want %016llx %016llx\n", t,
             (unsigned long long)r[t].hi, (unsigned long long)r[t].lo, (unsigned long long)e.hi,
             (unsigned long long)e.lo);
      wrong++;
    }
  }
  printf("%d of %d steps wrong\n", wrong, N);
  return wrong ? 1 : 0;
}
