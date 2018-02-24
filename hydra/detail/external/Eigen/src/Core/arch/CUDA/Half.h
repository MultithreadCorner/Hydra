// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// The conversion routines are Copyright (c) Fabian Giesen, 2016.
// The original license follows:
//
// Copyright (c) Fabian Giesen, 2016
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// Standard 16-bit float type, mostly useful for GPUs. Defines a new
// type HYDRA_EXTERNAL_NS::Eigen::half (inheriting from CUDA's __half struct) with
// operator overloads such that it behaves basically as an arithmetic
// type. It will be quite slow on CPUs (so it is recommended to stay
// in fp32 for CPUs, except for simple parameter conversions, I/O
// to disk and the likes), but fast on GPUs.


#ifndef HYDRA_EIGEN_HALF_CUDA_H
#define HYDRA_EIGEN_HALF_CUDA_H

#if __cplusplus > 199711L
#define HYDRA_EIGEN_EXPLICIT_CAST(tgt_type) explicit operator tgt_type()
#else
#define HYDRA_EIGEN_EXPLICIT_CAST(tgt_type) operator tgt_type()
#endif


HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {

struct half;

namespace half_impl {

#if !defined(HYDRA_EIGEN_HAS_CUDA_FP16)

// Make our own __half_raw  definition that is similar to CUDA's.
struct __half_raw  {
  HYDRA_EIGEN_DEVICE_FUNC __half_raw () {}
  explicit HYDRA_EIGEN_DEVICE_FUNC __half_raw (unsigned short raw) : x(raw) {}
  unsigned short x;
};
#elif defined(__CUDACC_VER__)&&__CUDACC_VER__  < 90000

typedef __half __half_raw;

#endif



HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC __half_raw raw_uint16_to_half(unsigned short x);
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC __half_raw float_to_half_rtne(float ff);
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC float half_to_float(__half_raw h);

struct half_base : public __half_raw {
  HYDRA_EIGEN_DEVICE_FUNC half_base() {}
  HYDRA_EIGEN_DEVICE_FUNC half_base(const half_base& h) : __half_raw(h) {}
  HYDRA_EIGEN_DEVICE_FUNC half_base(const __half_raw& h) : __half_raw(h) {}
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && __CUDACC_VER__ >= 90000
  HYDRA_EIGEN_DEVICE_FUNC half_base(const __half& h) : __half_raw(*(__half_raw*)&h) {}
#endif
};

} // namespace half_impl

// Class definition.
struct half : public half_impl::half_base {
  #if !defined(HYDRA_EIGEN_HAS_CUDA_FP16) || __CUDACC_VER__  < 90000
    typedef half_impl::__half_raw __half_raw;
  #endif

  HYDRA_EIGEN_DEVICE_FUNC half() {}

  HYDRA_EIGEN_DEVICE_FUNC half(const __half_raw& h) : half_impl::half_base(h) {}
  HYDRA_EIGEN_DEVICE_FUNC half(const half& h) : half_impl::half_base(h) {}
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) &&  __CUDACC_VER__ >= 90000
  HYDRA_EIGEN_DEVICE_FUNC half(const __half& h) : half_impl::half_base(h) {}
#endif

  explicit HYDRA_EIGEN_DEVICE_FUNC half(bool b)
      : half_impl::half_base(half_impl::raw_uint16_to_half(b ? 0x3c00 : 0)) {}
  template<class T>
  explicit HYDRA_EIGEN_DEVICE_FUNC half(const T& val)
      : half_impl::half_base(half_impl::float_to_half_rtne(static_cast<float>(val))) {}
  explicit HYDRA_EIGEN_DEVICE_FUNC half(float f)
      : half_impl::half_base(half_impl::float_to_half_rtne(f)) {}

  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(bool) const {
    // +0.0 and -0.0 become false, everything else becomes true.
    return (x & 0x7fff) != 0;
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(signed char) const {
    return static_cast<signed char>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(unsigned char) const {
    return static_cast<unsigned char>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(short) const {
    return static_cast<short>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(unsigned short) const {
    return static_cast<unsigned short>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(int) const {
    return static_cast<int>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(unsigned int) const {
    return static_cast<unsigned int>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(long) const {
    return static_cast<long>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(unsigned long) const {
    return static_cast<unsigned long>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(long long) const {
    return static_cast<long long>(half_impl::half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(unsigned long long) const {
    return static_cast<unsigned long long>(half_to_float(*this));
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(float) const {
    return half_impl::half_to_float(*this);
  }
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_EXPLICIT_CAST(double) const {
    return static_cast<double>(half_impl::half_to_float(*this));
  }

  HYDRA_EIGEN_DEVICE_FUNC half& operator=(const half& other) {
    x = other.x;
    return *this;
  }
};

namespace half_impl {

#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

// Intrinsics for native fp16 support. Note that on current hardware,
// these are no faster than fp32 arithmetic (you need to use the half2
// versions to get the ALU speed increased), but you do save the
// conversion steps back and forth.

__device__ half operator + (const half& a, const half& b) {
  return __hadd(a, b);
}
__device__ half operator * (const half& a, const half& b) {
  return __hmul(a, b);
}
__device__ half operator - (const half& a, const half& b) {
  return __hsub(a, b);
}
__device__ half operator / (const half& a, const half& b) {
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
}
__device__ half operator - (const half& a) {
  return __hneg(a);
}
__device__ half& operator += (half& a, const half& b) {
  a = a + b;
  return a;
}
__device__ half& operator *= (half& a, const half& b) {
  a = a * b;
  return a;
}
__device__ half& operator -= (half& a, const half& b) {
  a = a - b;
  return a;
}
__device__ half& operator /= (half& a, const half& b) {
  a = a / b;
  return a;
}
__device__ bool operator == (const half& a, const half& b) {
  return __heq(a, b);
}
__device__ bool operator != (const half& a, const half& b) {
  return __hne(a, b);
}
__device__ bool operator < (const half& a, const half& b) {
  return __hlt(a, b);
}
__device__ bool operator <= (const half& a, const half& b) {
  return __hle(a, b);
}
__device__ bool operator > (const half& a, const half& b) {
  return __hgt(a, b);
}
__device__ bool operator >= (const half& a, const half& b) {
  return __hge(a, b);
}

#else  // Emulate support for half floats

// Definitions for CPUs and older CUDA, mostly working through conversion
// to/from fp32.

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator + (const half& a, const half& b) {
  return half(float(a) + float(b));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator * (const half& a, const half& b) {
  return half(float(a) * float(b));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator - (const half& a, const half& b) {
  return half(float(a) - float(b));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator / (const half& a, const half& b) {
  return half(float(a) / float(b));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator - (const half& a) {
  half result;
  result.x = a.x ^ 0x8000;
  return result;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half& operator += (half& a, const half& b) {
  a = half(float(a) + float(b));
  return a;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half& operator *= (half& a, const half& b) {
  a = half(float(a) * float(b));
  return a;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half& operator -= (half& a, const half& b) {
  a = half(float(a) - float(b));
  return a;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half& operator /= (half& a, const half& b) {
  a = half(float(a) / float(b));
  return a;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator == (const half& a, const half& b) {
  return float(a) == float(b);
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator != (const half& a, const half& b) {
  return float(a) != float(b);
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator < (const half& a, const half& b) {
  return float(a) < float(b);
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator <= (const half& a, const half& b) {
  return float(a) <= float(b);
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator > (const half& a, const half& b) {
  return float(a) > float(b);
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool operator >= (const half& a, const half& b) {
  return float(a) >= float(b);
}

#endif  // Emulate support for half floats

// Division by an index. Do it in full float precision to avoid accuracy
// issues in converting the denominator to half.
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half operator / (const half& a, Index b) {
  return half(static_cast<float>(a) / static_cast<float>(b));
}

// Conversion routines, including fallbacks for the host or older CUDA.
// Note that newer Intel CPUs (Haswell or newer) have vectorized versions of
// these in hardware. If we need more performance on older/other CPUs, they are
// also possible to vectorize directly.

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC __half_raw  raw_uint16_to_half(unsigned short x) {
  __half_raw  h;
  h.x = x;
  return h;
}

union FP32 {
  unsigned int u;
  float f;
};

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC __half_raw  float_to_half_rtne(float ff) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
	__half tmp_ff = __float2half(ff);
	  return *(__half_raw*)&tmp_ff;

#elif defined(HYDRA_EIGEN_HAS_FP16_C)
  __half_raw h;
  h.x = _cvtss_sh(ff, 0);
  return h;

#else
  FP32 f; f.f = ff;

  const FP32 f32infty = { 255 << 23 };
  const FP32 f16max = { (127 + 16) << 23 };
  const FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  unsigned int sign_mask = 0x80000000u;
  __half_raw o;
  o.x = static_cast<unsigned short>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {  // result is Inf or NaN (all exponent bits set)
    o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  } else {  // (De)normalized number or zero
    if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      o.x = static_cast<unsigned short>(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o.x = static_cast<unsigned short>(f.u >> 13);
    }
  }

  o.x |= static_cast<unsigned short>(sign >> 16);
  return o;
#endif
}

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC float half_to_float(__half_raw h) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __half2float(h);

#elif defined(HYDRA_EIGEN_HAS_FP16_C)
  return _cvtsh_ss(h.x);

#else
  const FP32 magic = { 113 << 23 };
  const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
  FP32 o;

  o.u = (h.x & 0x7fff) << 13;             // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;   // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {     // Inf/NaN?
    o.u += (128 - 16) << 23;    // extra exp adjust
  } else if (exp == 0) {        // Zero/Denormal?
    o.u += 1 << 23;             // extra exp adjust
    o.f -= magic.f;             // renormalize
  }

  o.u |= (h.x & 0x8000) << 16;    // sign bit
  return o.f;
#endif
}

// --- standard functions ---

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool (isinf)(const half& a) {
  return (a.x & 0x7fff) == 0x7c00;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool (isnan)(const half& a) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(a);
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC bool (isfinite)(const half& a) {
  return !(isinf HYDRA_EIGEN_NOT_A_MACRO (a)) && !(isnan HYDRA_EIGEN_NOT_A_MACRO (a));
}

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half abs(const half& a) {
  half result;
  result.x = a.x & 0x7FFF;
  return result;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half exp(const half& a) {
  return half(::expf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half log(const half& a) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return HYDRA_EXTERNAL_NS::Eigen::half(::hlog(a));
#else
  return half(::logf(float(a)));
#endif
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half log1p(const half& a) {
  return half(numext::log1p(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half log10(const half& a) {
  return half(::log10f(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half sqrt(const half& a) {
  return half(::sqrtf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half pow(const half& a, const half& b) {
  return half(::powf(float(a), float(b)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half sin(const half& a) {
  return half(::sinf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half cos(const half& a) {
  return half(::cosf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half tan(const half& a) {
  return half(::tanf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half tanh(const half& a) {
  return half(::tanhf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half floor(const half& a) {
  return half(::floorf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half ceil(const half& a) {
  return half(::ceilf(float(a)));
}

HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half (min)(const half& a, const half& b) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(b, a) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f2 < f1 ? b : a;
#endif
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC half (max)(const half& a, const half& b) {
#if defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f1 < f2 ? b : a;
#endif
}

HYDRA_EIGEN_ALWAYS_INLINE std::ostream& operator << (std::ostream& os, const half& v) {
  os << static_cast<float>(v);
  return os;
}

} // end namespace half_impl

// import HYDRA_EXTERNAL_NS::Eigen::half_impl::half into Eigen namespace
// using half_impl::half;

namespace internal {

template<>
struct random_default_impl<half, false, false>
{
  static inline half run(const half& x, const half& y)
  {
    return x + (y-x) * half(float(std::rand()) / float(RAND_MAX));
  }
  static inline half run()
  {
    return run(half(-1.f), half(1.f));
  }
};

template<> struct is_arithmetic<half> { enum { value = true }; };

} // end namespace internal

}  /* end namespace Eigen */  HYDRA_EXTERNAL_NAMESPACE_END

namespace std {
template<>
struct numeric_limits<HYDRA_EXTERNAL_NS::Eigen::half> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 11;
  static const int digits10 = 2;
  //static const int max_digits10 = ;
  static const int radix = 2;
  static const int min_exponent = -13;
  static const int min_exponent10 = -4;
  static const int max_exponent = 16;
  static const int max_exponent10 = 4;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static HYDRA_EXTERNAL_NS::Eigen::half (min)() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x400); }
  static HYDRA_EXTERNAL_NS::Eigen::half lowest() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0xfbff); }
  static HYDRA_EXTERNAL_NS::Eigen::half (max)() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x7bff); }
  static HYDRA_EXTERNAL_NS::Eigen::half epsilon() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x0800); }
  static HYDRA_EXTERNAL_NS::Eigen::half round_error() { return HYDRA_EXTERNAL_NS::Eigen::half(0.5); }
  static HYDRA_EXTERNAL_NS::Eigen::half infinity() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x7c00); }
  static HYDRA_EXTERNAL_NS::Eigen::half quiet_NaN() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x7e00); }
  static HYDRA_EXTERNAL_NS::Eigen::half signaling_NaN() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x7e00); }
  static HYDRA_EXTERNAL_NS::Eigen::half denorm_min() { return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(0x1); }
};
}

HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {

template<> struct NumTraits<HYDRA_EXTERNAL_NS::Eigen::half>
    : GenericNumTraits<HYDRA_EXTERNAL_NS::Eigen::half>
{
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half epsilon() {
    return half_impl::raw_uint16_to_half(0x0800);
  }
  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half dummy_precision() { return HYDRA_EXTERNAL_NS::Eigen::half(1e-2f); }
  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half highest() {
    return half_impl::raw_uint16_to_half(0x7bff);
  }
  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half lowest() {
    return half_impl::raw_uint16_to_half(0xfbff);
  }
  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half infinity() {
    return half_impl::raw_uint16_to_half(0x7c00);
  }
  HYDRA_EIGEN_DEVICE_FUNC static HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half quiet_NaN() {
    return half_impl::raw_uint16_to_half(0x7c01);
  }
};

} /* end namespace Eigen */  HYDRA_EXTERNAL_NAMESPACE_END

// C-like standard mathematical functions and trancendentals.
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half fabsh(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
  HYDRA_EXTERNAL_NS::Eigen::half result;
  result.x = a.x & 0x7FFF;
  return result;
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half exph(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
  return HYDRA_EXTERNAL_NS::Eigen::half(::expf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half logh(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return HYDRA_EXTERNAL_NS::Eigen::half(::hlog(a));
#else
  return HYDRA_EXTERNAL_NS::Eigen::half(::logf(float(a)));
#endif
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half sqrth(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
  return HYDRA_EXTERNAL_NS::Eigen::half(::sqrtf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half powh(const HYDRA_EXTERNAL_NS::Eigen::half& a, const HYDRA_EXTERNAL_NS::Eigen::half& b) {
  return HYDRA_EXTERNAL_NS::Eigen::half(::powf(float(a), float(b)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half floorh(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
  return HYDRA_EXTERNAL_NS::Eigen::half(::floorf(float(a)));
}
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half ceilh(const HYDRA_EXTERNAL_NS::Eigen::half& a) {
  return HYDRA_EXTERNAL_NS::Eigen::half(::ceilf(float(a)));
}

namespace std {

#if __cplusplus > 199711L
template <>
struct hash<HYDRA_EXTERNAL_NS::Eigen::half> {
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE std::size_t operator()(const HYDRA_EXTERNAL_NS::Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
#endif

} // end namespace std


// Add the missing shfl_xor intrinsic
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__device__ HYDRA_EIGEN_STRONG_INLINE HYDRA_EXTERNAL_NS::Eigen::half __shfl_xor(HYDRA_EXTERNAL_NS::Eigen::half var, int laneMask, int width=warpSize) {
#if defined __CUDACC_VER__ && __CUDACC_VER__ < 90000
  return static_cast<HYDRA_EXTERNAL_NS::Eigen::half>(__shfl_xor(static_cast<float>(var), laneMask, width));
#else
return static_cast<HYDRA_EXTERNAL_NS::Eigen::half>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(var), laneMask, width));
#endif
}
#endif

// ldg() has an overload for __half_raw, but we also need one for HYDRA_EXTERNAL_NS::Eigen::half.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_DEVICE_FUNC HYDRA_EXTERNAL_NS::Eigen::half __ldg(const HYDRA_EXTERNAL_NS::Eigen::half* ptr) {
  return HYDRA_EXTERNAL_NS::Eigen::half_impl::raw_uint16_to_half(
      __ldg(reinterpret_cast<const unsigned short*>(ptr)));
}
#endif


#if defined(__CUDA_ARCH__)
HYDRA_EXTERNAL_NAMESPACE_BEGIN namespace Eigen {
namespace numext {

template<>
HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_ALWAYS_INLINE
bool (isnan)(const HYDRA_EXTERNAL_NS::Eigen::half& h) {
  return (half_impl::isnan)(h);
}

template<>
HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_ALWAYS_INLINE
bool (isinf)(const HYDRA_EXTERNAL_NS::Eigen::half& h) {
  return (half_impl::isinf)(h);
}

template<>
HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_ALWAYS_INLINE
bool (isfinite)(const HYDRA_EXTERNAL_NS::Eigen::half& h) {
  return (half_impl::isfinite)(h);
}

} // namespace Eigen
HYDRA_EXTERNAL_NAMESPACE_END
}  // namespace numext
#endif

#endif // HYDRA_EIGEN_HALF_CUDA_H
