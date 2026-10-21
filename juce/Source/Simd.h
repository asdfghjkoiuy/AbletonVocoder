#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define VOC_NEON 1
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define VOC_SSE2 1
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define VOC_INLINE inline __attribute__((always_inline))
#else
#define VOC_INLINE inline
#endif

// IEEE-754 log2 / exp2. Mantissa is reduced to [1/sqrt(2), sqrt(2)] and a
// degree-7 Taylor series finishes it. Absolute error is a few ulps, which
// is inaudible on the Depth power curve (~1e-4 dB).
VOC_INLINE float vocFastLog2(float x) noexcept
{
    if (!(x > 0.f))
        return -80.f;
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    int exp = (int) ((u >> 23) & 255) - 127;
    u = (u & 0x007fffffu) | 0x3f800000u;
    float m = 0.f;
    std::memcpy(&m, &u, 4);
    if (m > 1.41421356237f)
    {
        m *= 0.5f;
        ++exp;
    }
    const float p = m - 1.f;
    float y = 1.f / 8.f;
    y = 1.f / 7.f - p * y;
    y = 1.f / 6.f - p * y;
    y = 0.2f - p * y;
    y = 0.25f - p * y;
    y = (1.f / 3.f) - p * y;
    y = 0.5f - p * y;
    y = 1.f - p * y;
    return (float) exp + (p * y) * 1.4426950408889634f;
}

VOC_INLINE float vocFastExp2(float x) noexcept
{
    if (x < -126.f) return 0.f;
    if (x > 127.f) return 1.0e30f;
    const int i = (int) std::floor(x);
    const float f = x - (float) i;
    float y = 1.52527338e-5f;
    y = y * f + 0.000154035304f;
    y = y * f + 0.00133335581f;
    y = y * f + 0.00961812911f;
    y = y * f + 0.05550410866f;
    y = y * f + 0.24022650696f;
    y = y * f + 0.69314718246f;
    y = y * f + 1.f;
    uint32_t u = 0;
    std::memcpy(&u, &y, 4);
    u += (uint32_t) i << 23;
    float out = 0.f;
    std::memcpy(&out, &u, 4);
    return out;
}

VOC_INLINE float vocFastExp(float x) noexcept
{
    return vocFastExp2(x * 1.4426950408889634f);
}

#if VOC_NEON

struct Mask
{
    uint32x4_t m;
};

struct V4
{
    float32x4_t v = vdupq_n_f32(0.f);

    V4() noexcept = default;
    explicit V4(float32x4_t x) noexcept : v(x) {}

    static V4 splat(float x) noexcept { return V4(vdupq_n_f32(x)); }

    static V4 load(const float* p) noexcept { return V4(vld1q_f32(p)); }

    static V4 from(float a, float b, float c, float d) noexcept
    {
        alignas(16) float t[4] = { a, b, c, d };
        return load(t);
    }

    void store(float* p) const noexcept { vst1q_f32(p, v); }

    float hsum() const noexcept { return vaddvq_f32(v); }
    float hmax() const noexcept { return vmaxvq_f32(v); }

    bool allFinite() const noexcept
    {
        const uint32x4_t u = vreinterpretq_u32_f32(v);
        const uint32x4_t expMask = vdupq_n_u32(0x7f800000u);
        const uint32x4_t bad = vceqq_u32(vandq_u32(u, expMask), expMask);
        return vmaxvq_u32(bad) == 0u;
    }
};

VOC_INLINE V4 operator+(V4 a, V4 b) noexcept { return V4(vaddq_f32(a.v, b.v)); }
VOC_INLINE V4 operator-(V4 a, V4 b) noexcept { return V4(vsubq_f32(a.v, b.v)); }
VOC_INLINE V4 operator*(V4 a, V4 b) noexcept { return V4(vmulq_f32(a.v, b.v)); }
VOC_INLINE V4 operator/(V4 a, V4 b) noexcept { return V4(vdivq_f32(a.v, b.v)); }

VOC_INLINE V4 vmax(V4 a, V4 b) noexcept { return V4(vmaxq_f32(a.v, b.v)); }
VOC_INLINE V4 vmin(V4 a, V4 b) noexcept { return V4(vminq_f32(a.v, b.v)); }
VOC_INLINE V4 vsqrt(V4 a) noexcept { return V4(vsqrtq_f32(a.v)); }

VOC_INLINE Mask vgt(V4 a, V4 b) noexcept { return Mask { vcgtq_f32(a.v, b.v) }; }
VOC_INLINE Mask vlt(V4 a, V4 b) noexcept { return Mask { vcltq_f32(a.v, b.v) }; }
VOC_INLINE Mask vge(V4 a, V4 b) noexcept { return Mask { vcgeq_f32(a.v, b.v) }; }

VOC_INLINE V4 vsel(Mask m, V4 a, V4 b) noexcept { return V4(vbslq_f32(m.m, a.v, b.v)); }

VOC_INLINE V4 fastLog2(V4 x) noexcept
{
    x = vmax(x, V4::splat(1.0e-20f));
    int32x4_t bits = vreinterpretq_s32_f32(x.v);
    int32x4_t exp = vsubq_s32(vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(255)), vdupq_n_s32(127));
    int32x4_t mant = vorrq_s32(vandq_s32(bits, vdupq_n_s32(0x007fffff)), vdupq_n_s32(0x3f800000));
    V4 m(vreinterpretq_f32_s32(mant));
    const uint32x4_t hi = vcgtq_f32(m.v, vdupq_n_f32(1.41421356237f));
    m.v = vbslq_f32(hi, vmulq_f32(m.v, vdupq_n_f32(0.5f)), m.v);
    exp = vaddq_s32(exp, vandq_s32(vreinterpretq_s32_u32(hi), vdupq_n_s32(1)));
    const V4 p = m - V4::splat(1.f);
    V4 y = V4::splat(1.f / 8.f);
    y = V4::splat(1.f / 7.f) - p * y;
    y = V4::splat(1.f / 6.f) - p * y;
    y = V4::splat(0.2f) - p * y;
    y = V4::splat(0.25f) - p * y;
    y = V4::splat(1.f / 3.f) - p * y;
    y = V4::splat(0.5f) - p * y;
    y = V4::splat(1.f) - p * y;
    const V4 log2m = (p * y) * V4::splat(1.4426950408889634f);
    return V4(vaddq_f32(vcvtq_f32_s32(exp), log2m.v));
}

VOC_INLINE V4 fastExp2(V4 x) noexcept
{
    x = vmin(vmax(x, V4::splat(-126.f)), V4::splat(127.f));
    const int32x4_t i = vcvtq_s32_f32(vrndmq_f32(x.v));
    const V4 f = x - V4(vcvtq_f32_s32(i));
    V4 y = V4::splat(1.52527338e-5f);
    y = y * f + V4::splat(0.000154035304f);
    y = y * f + V4::splat(0.00133335581f);
    y = y * f + V4::splat(0.00961812911f);
    y = y * f + V4::splat(0.05550410866f);
    y = y * f + V4::splat(0.24022650696f);
    y = y * f + V4::splat(0.69314718246f);
    y = y * f + V4::splat(1.f);
    int32x4_t bits = vreinterpretq_s32_f32(y.v);
    bits = vaddq_s32(bits, vshlq_n_s32(i, 23));
    return V4(vreinterpretq_f32_s32(bits));
}

#elif VOC_SSE2

struct Mask
{
    __m128 m;
};

struct V4
{
    __m128 v = _mm_setzero_ps();

    V4() noexcept = default;
    explicit V4(__m128 x) noexcept : v(x) {}

    static V4 splat(float x) noexcept { return V4(_mm_set1_ps(x)); }

    static V4 load(const float* p) noexcept { return V4(_mm_loadu_ps(p)); }

    static V4 from(float a, float b, float c, float d) noexcept
    {
        return V4(_mm_set_ps(d, c, b, a));
    }

    void store(float* p) const noexcept { _mm_storeu_ps(p, v); }

    float hsum() const noexcept
    {
        __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    float hmax() const noexcept
    {
        __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 maxs = _mm_max_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, maxs);
        maxs = _mm_max_ss(maxs, shuf);
        return _mm_cvtss_f32(maxs);
    }

    bool allFinite() const noexcept
    {
        const __m128i u = _mm_castps_si128(v);
        const __m128i expMask = _mm_set1_epi32(0x7f800000);
        const __m128i bad = _mm_cmpeq_epi32(_mm_and_si128(u, expMask), expMask);
        return _mm_movemask_epi8(bad) == 0;
    }
};

VOC_INLINE V4 operator+(V4 a, V4 b) noexcept { return V4(_mm_add_ps(a.v, b.v)); }
VOC_INLINE V4 operator-(V4 a, V4 b) noexcept { return V4(_mm_sub_ps(a.v, b.v)); }
VOC_INLINE V4 operator*(V4 a, V4 b) noexcept { return V4(_mm_mul_ps(a.v, b.v)); }
VOC_INLINE V4 operator/(V4 a, V4 b) noexcept { return V4(_mm_div_ps(a.v, b.v)); }

VOC_INLINE V4 vmax(V4 a, V4 b) noexcept { return V4(_mm_max_ps(a.v, b.v)); }
VOC_INLINE V4 vmin(V4 a, V4 b) noexcept { return V4(_mm_min_ps(a.v, b.v)); }
VOC_INLINE V4 vsqrt(V4 a) noexcept { return V4(_mm_sqrt_ps(a.v)); }

VOC_INLINE Mask vgt(V4 a, V4 b) noexcept { return Mask { _mm_cmpgt_ps(a.v, b.v) }; }
VOC_INLINE Mask vlt(V4 a, V4 b) noexcept { return Mask { _mm_cmplt_ps(a.v, b.v) }; }
VOC_INLINE Mask vge(V4 a, V4 b) noexcept { return Mask { _mm_cmpge_ps(a.v, b.v) }; }

VOC_INLINE V4 vsel(Mask m, V4 a, V4 b) noexcept
{
    return V4(_mm_or_ps(_mm_and_ps(m.m, a.v), _mm_andnot_ps(m.m, b.v)));
}

VOC_INLINE V4 fastLog2(V4 x) noexcept
{
    alignas(16) float t[4];
    x.store(t);
    return V4::from(vocFastLog2(t[0]), vocFastLog2(t[1]), vocFastLog2(t[2]), vocFastLog2(t[3]));
}

VOC_INLINE V4 fastExp2(V4 x) noexcept
{
    alignas(16) float t[4];
    x.store(t);
    return V4::from(vocFastExp2(t[0]), vocFastExp2(t[1]), vocFastExp2(t[2]), vocFastExp2(t[3]));
}

#else

struct Mask
{
    float m[4] = { 0.f, 0.f, 0.f, 0.f };
};

struct V4
{
    float v[4] = { 0.f, 0.f, 0.f, 0.f };

    static V4 splat(float x) noexcept { return V4 { { x, x, x, x } }; }

    static V4 load(const float* p) noexcept { return V4 { { p[0], p[1], p[2], p[3] } }; }

    static V4 from(float a, float b, float c, float d) noexcept { return V4 { { a, b, c, d } }; }

    void store(float* p) const noexcept
    {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }

    float hsum() const noexcept { return (v[0] + v[1]) + (v[2] + v[3]); }
    float hmax() const noexcept { return std::max(std::max(v[0], v[1]), std::max(v[2], v[3])); }

    bool allFinite() const noexcept
    {
        for (int i = 0; i < 4; ++i)
        {
            uint32_t u = 0;
            std::memcpy(&u, &v[i], 4);
            if ((u & 0x7f800000u) == 0x7f800000u)
                return false;
        }
        return true;
    }
};

VOC_INLINE V4 operator+(V4 a, V4 b) noexcept
{
    return V4 { { a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3] } };
}
VOC_INLINE V4 operator-(V4 a, V4 b) noexcept
{
    return V4 { { a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2], a.v[3] - b.v[3] } };
}
VOC_INLINE V4 operator*(V4 a, V4 b) noexcept
{
    return V4 { { a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3] } };
}
VOC_INLINE V4 operator/(V4 a, V4 b) noexcept
{
    return V4 { { a.v[0] / b.v[0], a.v[1] / b.v[1], a.v[2] / b.v[2], a.v[3] / b.v[3] } };
}
VOC_INLINE V4 vmax(V4 a, V4 b) noexcept
{
    return V4 { { std::max(a.v[0], b.v[0]), std::max(a.v[1], b.v[1]),
                  std::max(a.v[2], b.v[2]), std::max(a.v[3], b.v[3]) } };
}
VOC_INLINE V4 vmin(V4 a, V4 b) noexcept
{
    return V4 { { std::min(a.v[0], b.v[0]), std::min(a.v[1], b.v[1]),
                  std::min(a.v[2], b.v[2]), std::min(a.v[3], b.v[3]) } };
}
VOC_INLINE V4 vsqrt(V4 a) noexcept
{
    return V4 { { std::sqrt(a.v[0]), std::sqrt(a.v[1]), std::sqrt(a.v[2]), std::sqrt(a.v[3]) } };
}
VOC_INLINE Mask vgt(V4 a, V4 b) noexcept
{
    return Mask { { a.v[0] > b.v[0] ? 1.f : 0.f, a.v[1] > b.v[1] ? 1.f : 0.f,
                    a.v[2] > b.v[2] ? 1.f : 0.f, a.v[3] > b.v[3] ? 1.f : 0.f } };
}
VOC_INLINE Mask vlt(V4 a, V4 b) noexcept { return vgt(b, a); }
VOC_INLINE Mask vge(V4 a, V4 b) noexcept
{
    return Mask { { a.v[0] >= b.v[0] ? 1.f : 0.f, a.v[1] >= b.v[1] ? 1.f : 0.f,
                    a.v[2] >= b.v[2] ? 1.f : 0.f, a.v[3] >= b.v[3] ? 1.f : 0.f } };
}
VOC_INLINE V4 vsel(Mask m, V4 a, V4 b) noexcept
{
    V4 o;
    for (int i = 0; i < 4; ++i)
        o.v[i] = m.m[i] > 0.5f ? a.v[i] : b.v[i];
    return o;
}
VOC_INLINE V4 fastLog2(V4 x) noexcept
{
    return V4::from(vocFastLog2(x.v[0]), vocFastLog2(x.v[1]), vocFastLog2(x.v[2]), vocFastLog2(x.v[3]));
}
VOC_INLINE V4 fastExp2(V4 x) noexcept
{
    return V4::from(vocFastExp2(x.v[0]), vocFastExp2(x.v[1]), vocFastExp2(x.v[2]), vocFastExp2(x.v[3]));
}

#endif

// Exponent-bit test. Immune to -ffast-math rewriting of (x == x) / std::isfinite.
VOC_INLINE bool vocIsFinite(float x) noexcept
{
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    return (u & 0x7f800000u) != 0x7f800000u;
}

VOC_INLINE float vocSanitize(float x) noexcept
{
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u)
        return 0.f;
    if ((u & 0x7fffffffu) < 0x26800000u)
        return 0.f;
    if ((u & 0x7fffffffu) > 0x4cbebc20u) // |x| > 1.0e8f
        return 0.f;
    return x;
}
