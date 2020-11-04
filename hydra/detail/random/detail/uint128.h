/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * uint128.h
 *
 *  Created on: 01/11/2020
 *      Author: Antonio Augusto Alves Junior
 *        Note: Loosely based on the GPLv3 code https://github.com/curtisseizert/CUDA-uint128
 */

#ifndef UINT128_H_
#define UINT128_H_

#include <hydra/detail/Config.h>
#include <iostream>
#include <iomanip>
#include <cinttypes>
#include <cuda.h>
#include <cmath>
#include <string>
#include <vector>
#include <iterator>
#include <utility>

#ifdef __has_builtin
# define uint128_t_has_builtin(x) __has_builtin(x)
#else
# define uint128_t_has_builtin(x) 0
#endif

namespace hydra {

namespace random {

class uint128_t;

namespace detail {

template<typename T>
using check_uint128_t = typename std::enable_if<std::is_integral<T>::value || std::is_same<T,uint128_t >::value, uint128_t>::type;

template<typename T>
using T_type = typename std::enable_if<std::is_integral<T>::value, T>::type;

}

class uint128_t
{
	uint64_t lo;
	uint64_t hi;

  public:

	uint128_t() = default;

	__hydra_host__ __hydra_device__
	uint128_t(uint64_t a){

		lo = a ;
		hi = 0;
	}

	__hydra_host__ __hydra_device__
	uint128_t(uint128_t const& a){

		lo = a.lo ;
		hi = a.hi;
	}

    template<typename T, typename=typename std::enable_if<std::is_integral<T>::value>::type>
	explicit inline operator T() const {

		return T(lo);
	}

	__hydra_host__ __hydra_device__
	uint128_t & operator=( uint128_t const& n){

		lo = n.lo;
		hi = n.hi;
		return * this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<std::is_integral<T>::value>::type&
	operator=(T const& n){

		hi = 0;
		lo = n;
		return * this;
	}

    //
	// Compound assignment
	//--------------------------------------
	//
	// arithmetic
	//
	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T> & operator+=(const T & a){

		uint128_t b = (uint128_t) a;
		hi += b.hi + ((lo + b.lo) < lo);
        lo += b.lo;
        return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T>& operator-=(const T & b){

		uint128_t temp = (uint128_t)b;
		if(lo < temp.lo) hi--;
		lo -= temp.lo;
		return * this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T>& operator*=( const T & b){

		 *this = mul128(*this, b);
		return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T>& operator/=( const T & b){

		 *this = divide_u128_to_u64(*this, (uint64_t)b);
		return *this;
	}
	//
	// increment
	//
	__hydra_host__ __hydra_device__
	inline uint128_t  & operator++(){

		return *this +=1;
	}

	__hydra_host__ __hydra_device__
	inline uint128_t & operator--(){

		return *this -=1;
	}

	//
	// bitwise
	//
	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T>& operator>>=(const T & b)
	{
		lo = (lo >> b) | (hi << (int)(b - 64));
		(b < 64) ? hi >>= b : hi = 0;
		return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	inline detail::check_uint128_t<T>& operator<<=(const T & b){

		hi = (hi << b) | (lo << (int)(b - 64));
		(b < 64) ? lo <<= b : lo = 0;
		return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	detail::check_uint128_t<T>  & operator|=(const T & b){

		*this = *this | b;
		return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	detail::check_uint128_t<T>  & operator&=(const T & b){

		*this = *this & b; return *this;
	}

	template <typename T>
	__hydra_host__ __hydra_device__
	detail::check_uint128_t<T>  & operator^=(const T & b){

		*this = *this ^ b;
		return *this;
	}

	//
	// friend functions
	//-----------------------------
	template <typename T>
	__hydra_host__ __hydra_device__
	friend inline detail::check_uint128_t<T> operator>>(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend inline detail::check_uint128_t<T>  operator<<(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T> operator+(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T>  operator-(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T>  operator*(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend T operator/(uint128_t x, const T & v);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend T operator%(uint128_t x, const T & v);

	__hydra_host__ __hydra_device__
	friend bool operator<(uint128_t const&  a, uint128_t const&  b);

	__hydra_host__ __hydra_device__
	friend bool operator>(uint128_t const&  a, uint128_t const&  b);

	__hydra_host__ __hydra_device__
	friend bool operator<=(uint128_t  const& a, uint128_t const&  b);

	__hydra_host__ __hydra_device__
	friend bool operator>=(uint128_t  const& a, uint128_t  const& b);

	__hydra_host__ __hydra_device__
	friend bool operator==(uint128_t  const& a, uint128_t const&  b);

	__hydra_host__ __hydra_device__
	friend bool operator!=(uint128_t  const& a, uint128_t  const& b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T>  operator|(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T>  operator&(uint128_t a, const T & b);

	template <typename T>
	__hydra_host__ __hydra_device__
	friend detail::check_uint128_t<T>  operator^(uint128_t a, const T & b);

	__hydra_host__ __hydra_device__
	friend uint128_t operator~(uint128_t a);

	__hydra_host__ __hydra_device__
	friend uint128_t min(uint128_t a, uint128_t b);

	__hydra_host__ __hydra_device__
	friend uint128_t max(uint128_t a, uint128_t b);

    // This just makes it more convenient to count leading zeros for uint128_t
	__hydra_host__ __hydra_device__
	friend inline uint64_t clz128(uint128_t  const& x);

	//  iostream
	//------------------
	friend inline std::ostream & operator<<(std::ostream & out, uint128_t x);

	//
	// static members
	//------------------------
	__hydra_host__ __hydra_device__
	static  bool is_less_than(uint128_t  const& a, uint128_t  const& b){

		if(a.hi < b.hi) return 1;
		if(a.hi > b.hi) return 0;
		if(a.lo < b.lo) return 1;
		else return 0;
	}

	__hydra_host__ __hydra_device__
	static  bool is_less_than_or_equal(uint128_t  const& a, uint128_t const&  b){

		if(a.hi < b.hi) return 1;
		if(a.hi > b.hi) return 0;
		if(a.lo <= b.lo) return 1;
		else return 0;
	}

	__hydra_host__ __hydra_device__
	static  bool is_greater_than(uint128_t  const& a, uint128_t  const& b){

		if(a.hi < b.hi) return 0;
		if(a.hi > b.hi) return 1;
		if(a.lo <= b.lo) return 0;
		else return 1;
	}

	__hydra_host__ __hydra_device__
	static  bool is_greater_than_or_equal(uint128_t  const& a, uint128_t  const& b)
	{
		if(a.hi < b.hi) return 0;
		if(a.hi > b.hi) return 1;
		if(a.lo < b.lo) return 0;
		else return 1;
	}

	__hydra_host__ __hydra_device__
	static  bool is_equal_to(uint128_t  const& a, uint128_t  const& b)
	{
		if(a.lo == b.lo && a.hi == b.hi) return 1;
		else return 0;
	}

	__hydra_host__ __hydra_device__
	static  bool is_not_equal_to(uint128_t  const& a, uint128_t  const& b)
	{
		if(a.lo != b.lo || a.hi != b.hi) return 1;
		else return 0;
	}


	//   bit operations
	//---------------------

	/// This counts leading zeros for 64 bit unsigned integers.  It is used internally
	/// in a few of the functions defined below.
	__hydra_host__ __hydra_device__
	static inline int clz64(uint64_t const&  x)
	{
		int res;
#ifdef __CUDA_ARCH__
		res = __clzll(x);
#elif __GNUC__ || uint128_t_has_builtin(__builtin_clzll)
		res = __builtin_clzll(x);
#elif __x86_64__
		asm("bsr %1, %0\nxor $0x3f, %0" : "=r" (res) : "rm" (x) : "cc", "flags");
#elif __aarch64__
		asm("clz %0, %1" : "=r" (res) : "r" (x));
#else
# error Architecture not supported
#endif
		return res;
	}


	__hydra_host__ __hydra_device__
	static  uint128_t bitwise_or(uint128_t a, uint128_t const&  b)
	{
		a.lo |= b.lo;
		a.hi |= b.hi;
		return a;
	}

	__hydra_host__ __hydra_device__
	static  uint128_t bitwise_and(uint128_t a, uint128_t  const& b)
	{
		a.lo &= b.lo;
		a.hi &= b.hi;
		return a;
	}

	__hydra_host__ __hydra_device__
	static inline  uint128_t bitwise_xor(uint128_t a, uint128_t const&  b)
	{
		a.lo ^= b.lo;
		a.hi ^= b.hi;
		return a;
	}

	__hydra_host__ __hydra_device__
	static inline  uint128_t bitwise_not(uint128_t a)
	{
		a.lo = ~a.lo;
		a.hi = ~a.hi;
		return a;
	}

	//   arithmetic
	//-------------------

	__hydra_host__ __hydra_device__
	static inline uint128_t add128(uint128_t x, uint128_t const& y)
	{
		x+=y;
		return x;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t add128(uint128_t x, uint64_t const& y)
	{
		x+=uint128_t(y);
		return x;

	}

	__hydra_host__ __hydra_device__
	static inline uint128_t mul128(uint64_t const&  x, uint64_t  const& y)
	{
		uint128_t res;
#ifdef __CUDA_ARCH__
		res.lo = x * y;
		res.hi = __mul64hi(x, y);
#elif __x86_64__
		asm( "mulq %3\n\t"
				: "=a" (res.lo), "=d" (res.hi)
				  : "%0" (x), "rm" (y));
#elif __aarch64__
		asm( "mul %0, %2, %3\n\t"
				"umulh %1, %2, %3\n\t"
				: "=&r" (res.lo), "=r" (res.hi)
				  : "r" (x), "r" (y));
#else
# error Architecture not supported
#endif
		return res;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t mul128(uint128_t  const& x, uint128_t  const& y)
	{
		uint128_t z = uint128_t::mul128(x.lo, y.lo);
		z.hi +=(x.hi * y.lo) + (x.lo * y.hi);
		return z;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t mul128(uint128_t  const& x, uint64_t  const& y)
	{
		uint128_t res;
#ifdef __CUDA_ARCH__
		res.lo = x.lo * y;
		res.hi = __mul64hi(x.lo, y);
		res.hi += x.hi * y;
#elif __x86_64__
		asm( "mulq %3\n\t"
				: "=a" (res.lo), "=d" (res.hi)
				  : "%0" (x.lo), "rm" (y));
		res.hi += x.hi * y;
#elif __aarch64__
		res.lo = x.lo * y;
		asm( "umulh %0, %1, %2\n\t"
				: "=r" (res.hi)
				  : "r" (x.lo), "r" (y));
		res.hi += x.hi * y;
#else
# error Architecture not supported
#endif
		return res;
	}

	// taken from libdivide's adaptation of this implementation origininally in
	// Hacker's Delight: http://www.hackersdelight.org/hdcodetxt/divDouble.c.txt
	// License permits inclusion here per:
	// http://www.hackersdelight.org/permissions.htm
	__hydra_host__ __hydra_device__
	static inline uint64_t divide_u128_to_u64(uint128_t  const& x, uint64_t v, uint64_t * r = NULL) // x / v
	{
		const uint64_t b = 1ull << 32;
		uint64_t  un1, un0,
		vn1, vn0,
		q1, q0,
		un64, un21, un10,
		rhat;
		int s;

		if(x.hi >= v){
			if( r != NULL) *r = (uint64_t) -1;
			return  (uint64_t) -1;
		}

		s = clz64(v);

		if(s > 0){
			v = v << s;
			un64 = (x.hi << s) | ((x.lo >> (64 - s)) & (-s >> 31));
			un10 = x.lo << s;
		}else{
			un64 = x.lo | x.hi;
			un10 = x.lo;
		}

		vn1 = v >> 32;
		vn0 = v & 0xffffffff;

		un1 = un10 >> 32;
		un0 = un10 & 0xffffffff;

		q1 = un64/vn1;
		rhat = un64 - q1*vn1;

		again1:
		if (q1 >= b || q1*vn0 > b*rhat + un1){
			q1 -= 1;
			rhat = rhat + vn1;
			if(rhat < b) goto again1;
		}

		un21 = un64*b + un1 - q1*v;

		q0 = un21/vn1;
		rhat = un21 - q0*vn1;
		again2:
		if(q0 >= b || q0 * vn0 > b*rhat + un0){
			q0 = q0 - 1;
			rhat = rhat + vn1;
			if(rhat < b) goto again2;
		}

		if(r != NULL) *r = (un21*b + un0 - q0*v) >> s;
		return q1*b + q0;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t divide_u128_to_u128(uint128_t  x, uint64_t v, uint64_t * r = NULL)
	{
		uint128_t res;

		res.hi = x.hi/v;
		x.hi %= v;
		res.lo = divide_u128_to_u64(x, v, r);

		return res;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t sub128(uint128_t  const& x, uint128_t const&  y) // x - y
	{
		uint128_t res;

		res.lo = x.lo - y.lo;
		res.hi = x.hi - y.hi;
		if(x.lo < y.lo) res.hi--;

		return res;
	}

	__hydra_host__ __hydra_device__
	static inline uint64_t _isqrt(uint64_t  const& x)
	{
		uint64_t res0 = 0;
#ifdef __CUDA_ARCH__
		res0 = sqrtf(x);
#else
		res0 = sqrt(x);
		for(uint16_t i = 0; i < 8; i++)
			res0 = (res0 + x/res0) >> 1;
#endif
		return res0;
	}

	//    roots
	//-------------------

	__hydra_host__ __hydra_device__
	static inline uint64_t _isqrt(const uint128_t & x) // this gives errors above 2^124
	{
		uint64_t res0 = 0;

		if(x == 0 || x.hi > 1ull << 60)
			return 0;

#ifdef __CUDA_ARCH__
		res0 = sqrtf(u128_to_float(x));
#else
		res0 = std::sqrt(u128_to_float(x));
#endif
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for(uint16_t i = 0; i < 8; i++)
			res0 = (res0 + x/res0) >> 1;

		return res0;
	}

	__hydra_host__ __hydra_device__
	static inline uint64_t _icbrt(const uint128_t & x)
	{
		uint64_t res0 = 0;

#ifdef __CUDA_ARCH__
		res0 = cbrtf(u128_to_float(x));
#else
		res0 = std::cbrt(u128_to_float(x));
#endif
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for(uint16_t i = 0; i < 47; i++) // there needs to be an odd number of iterations
			// for the case of numbers of the form x^2 - 1
			// where this will not converge
			res0 = (res0 + divide_u128_to_u128(x,res0)/res0) >> 1;
		return res0;
	}

	__hydra_host__ __hydra_device__
	// this function is to avoid off by 1 errors from nesting integer square roots
	static inline uint64_t _iqrt(const uint128_t & x)
	{
		uint64_t res0 = 0, res1 = 0;

		res0 = _isqrt(_isqrt(x));
		res1 = (res0 + divide_u128_to_u128(x,res0*res0)/res0) >> 1;
		res0 = (res1 + divide_u128_to_u128(x,res1*res1)/res1) >> 1;

		return res0 < res1 ? res0 : res1;
	}

	//  typecasting
	//-------------------------
	static inline uint128_t string_to_u128(std::string s)
	{
		uint128_t res = 0;
		for(std::string::iterator iter = s.begin(); iter != s.end() && (int) *iter >= 48; iter++){
			res = mul128(res, 10);
			res += (uint16_t) *iter - 48;
		}
		return res;
	}

	__hydra_host__ __hydra_device__
	static inline uint64_t u128_to_u64(uint128_t x){

		return x.lo;
	}

	__hydra_host__ __hydra_device__
	static inline double u128_to_double(uint128_t x)
	{
		double dbl;
#ifdef __CUDA_ARCH__
		if(x.hi == 0) return __ull2double_rd(x.lo);
#else
		if(x.hi == 0) return (double) x.lo;
#endif
		uint64_t r = clz64(x.hi);
		x <<= r;

#ifdef __CUDA_ARCH__
		dbl = __ull2double_rd(x.hi);
#else
		dbl = (double) x.lo;
#endif

		dbl *= (1ull << (64-r));

		return dbl;
	}

	__hydra_host__ __hydra_device__
	static inline float u128_to_float(uint128_t x)
	{
		float flt;
#ifdef __CUDA_ARCH__
		if(x.hi == 0) return __ull2float_rd(x.lo);
#else
		if(x.hi == 0) return (float) x.lo;
#endif
		uint64_t r = clz64(x.hi);
		x <<= r;

#ifdef __CUDA_ARCH__
		flt = __ull2float_rd(x.hi);
#else
		flt = (float) x.hi;
#endif

		flt *= (1ull << (64-r));
		flt *= 2;

		return flt;
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t double_to_u128(double dbl)
	{
		uint128_t x;
		if(dbl < 1 || dbl > 1e39) return 0;
		else{

#ifdef __CUDA_ARCH__
			uint32_t shft = __double2uint_rd(log2(dbl));
			uint64_t divisor = 1ull << shft;
			dbl /= divisor;
			x.lo = __double2ull_rd(dbl);
			x <<= shft;
#else
			uint32_t shft = (uint32_t) log2(dbl);
			uint64_t divisor = 1ull << shft;
			dbl /= divisor;
			x.lo = (uint64_t) dbl;
			x <<= shft;
#endif
			return x;
		}
	}

	__hydra_host__ __hydra_device__
	static inline uint128_t float_to_u128(float flt)
	{
		uint128_t x;
		if(flt < 1 || flt > 1e39) return 0;
		else{

#ifdef __CUDA_ARCH__
			uint32_t shft = __double2uint_rd(log2(flt));
			uint64_t divisor = 1ull << shft;
			flt /= divisor;
			x.lo = __double2ull_rd(flt);
			x <<= shft;
#else
			uint32_t shft = (uint32_t) log2(flt);
			uint64_t divisor = 1ull << shft;
			flt /= divisor;
			x.lo = (uint64_t) flt;
			x <<= shft;
#endif
			return x;
		}
	}

}; // class uint128_t

}  // namespace random

}  // namespace hydra

#include <hydra/detail/random/detail/uint128.inl>

#endif /* UINT128_H_ */
