//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_SIGNBIT_HPP
#define HYDRA_BOOST_MATH_CCMATH_SIGNBIT_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/assert.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/detail/fp_traits.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#ifdef HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <hydra/detail/external/hydra_boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

#ifdef __has_include
#  if __has_include(<bit>)
#    include <bit>
#    if __cpp_lib_bit_cast >= 201806L
#      define HYDRA_BOOST_MATH_BIT_CAST(T, x) std::bit_cast<T>(x)
#    endif
#  elif defined(__has_builtin)
#    if __has_builtin(__builtin_bit_cast)
#      define HYDRA_BOOST_MATH_BIT_CAST(T, x) __builtin_bit_cast(T, x)
#    endif
#  endif
#endif

/*
The following error is given using Apple Clang version 13.1.6, and Clang 13, and 14 on Ubuntu 22.04.01
TODO: Remove the following undef when Apple Clang supports

ccmath_signbit_test.cpp:32:19: error: static_assert expression is not an integral constant expression
    static_assert(hydra_boost::math::ccmath::signbit(T(-1)) == true);
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
../../../boost/math/ccmath/signbit.hpp:62:24: note: constexpr bit_cast involving bit-field is not yet supported
        const auto u = HYDRA_BOOST_MATH_BIT_CAST(float_bits, arg);
                       ^
../../../boost/math/ccmath/signbit.hpp:20:37: note: expanded from macro 'HYDRA_BOOST_MATH_BIT_CAST'
#  define HYDRA_BOOST_MATH_BIT_CAST(T, x) __builtin_bit_cast(T, x)
                                    ^
*/

#if defined(__clang__) && defined(HYDRA_BOOST_MATH_BIT_CAST)
#  undef HYDRA_BOOST_MATH_BIT_CAST
#endif

namespace hydra_boost::math::ccmath {

namespace detail {

#ifdef HYDRA_BOOST_MATH_BIT_CAST

struct IEEEf2bits
{
#if HYDRA_BOOST_MATH_ENDIAN_LITTLE_BYTE
    std::uint32_t mantissa : 23;
    std::uint32_t exponent : 8;
    std::uint32_t sign : 1;
#else // Big endian
    std::uint32_t sign : 1;
    std::uint32_t exponent : 8;
    std::uint32_t mantissa : 23;
#endif 
};

struct IEEEd2bits
{
#if HYDRA_BOOST_MATH_ENDIAN_LITTLE_BYTE
    std::uint32_t mantissa_l : 32;
    std::uint32_t mantissa_h : 20;
    std::uint32_t exponent : 11;
    std::uint32_t sign : 1;
#else // Big endian
    std::uint32_t sign : 1;
    std::uint32_t exponent : 11;
    std::uint32_t mantissa_h : 20;
    std::uint32_t mantissa_l : 32;
#endif
};

// 80 bit long double
#if LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384

struct IEEEl2bits
{
#if HYDRA_BOOST_MATH_ENDIAN_LITTLE_BYTE
    std::uint32_t mantissa_l : 32;
    std::uint32_t mantissa_h : 32;
    std::uint32_t exponent : 15;
    std::uint32_t sign : 1;
    std::uint32_t pad : 32;
#else // Big endian
    std::uint32_t pad : 32;
    std::uint32_t sign : 1;
    std::uint32_t exponent : 15;
    std::uint32_t mantissa_h : 32;
    std::uint32_t mantissa_l : 32;
#endif
};

// 128 bit long double
#elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384

struct IEEEl2bits
{
#if HYDRA_BOOST_MATH_ENDIAN_LITTLE_BYTE
    std::uint64_t mantissa_l : 64;
    std::uint64_t mantissa_h : 48;
    std::uint32_t exponent : 15;
    std::uint32_t sign : 1;
#else // Big endian
    std::uint32_t sign : 1;
    std::uint32_t exponent : 15;
    std::uint64_t mantissa_h : 48;
    std::uint64_t mantissa_l : 64;
#endif
};

// 64 bit long double (double == long double on ARM)
#elif LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024

struct IEEEl2bits
{
#if HYDRA_BOOST_MATH_ENDIAN_LITTLE_BYTE
    std::uint32_t mantissa_l : 32;
    std::uint32_t mantissa_h : 20;
    std::uint32_t exponent : 11;
    std::uint32_t sign : 1;
#else // Big endian
    std::uint32_t sign : 1;
    std::uint32_t exponent : 11;
    std::uint32_t mantissa_h : 20;
    std::uint32_t mantissa_l : 32;
#endif
};

#else // Unsupported long double representation
#  define HYDRA_BOOST_MATH_UNSUPPORTED_LONG_DOUBLE
#endif

template <typename T>
constexpr bool signbit_impl(T arg)
{
    if constexpr (std::is_same_v<T, float>)
    {   
        const auto u = HYDRA_BOOST_MATH_BIT_CAST(IEEEf2bits, arg);
        return u.sign;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        const auto u = HYDRA_BOOST_MATH_BIT_CAST(IEEEd2bits, arg);
        return u.sign;
    }
    #ifndef HYDRA_BOOST_MATH_UNSUPPORTED_LONG_DOUBLE
    else if constexpr (std::is_same_v<T, long double>)
    {
        const auto u = HYDRA_BOOST_MATH_BIT_CAST(IEEEl2bits, arg);
        return u.sign;
    }
    #endif
    else
    {
        HYDRA_BOOST_MATH_ASSERT_MSG(!hydra_boost::math::ccmath::isnan(arg), "NAN is not supported with this type or platform");
        HYDRA_BOOST_MATH_ASSERT_MSG(hydra_boost::math::ccmath::abs(arg) != 0, "Signed 0 is not support with this type or platform");

        return arg < static_cast<T>(0);
    }
}

#else

// Typical implementations of signbit involve type punning via union and manipulating
// overflow (see libc++ or musl). Neither of these are allowed in constexpr contexts
// (technically type punning via union in general is UB in c++ but well defined in C) 
// therefore we static assert these cases.

template <typename T>
constexpr bool signbit_impl(T arg)
{
    HYDRA_BOOST_MATH_ASSERT_MSG(!hydra_boost::math::ccmath::isnan(arg), "NAN is not supported without __builtin_bit_cast or std::bit_cast");
    HYDRA_BOOST_MATH_ASSERT_MSG(hydra_boost::math::ccmath::abs(arg) != 0, "Signed 0 is not support without __builtin_bit_cast or std::bit_cast");

    return arg < static_cast<T>(0);
}

#endif

}

// Return value: true if arg is negative, false if arg is 0, NAN, or positive
template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr bool signbit(Real arg)
{
    if (HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return hydra_boost::math::ccmath::detail::signbit_impl(arg);
    }
    else
    {
        using std::signbit;
        return signbit(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
constexpr bool signbit(Z arg)
{
    return hydra_boost::math::ccmath::signbit(static_cast<double>(arg));
}

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_SIGNBIT_HPP
