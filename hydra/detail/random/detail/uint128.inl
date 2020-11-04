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
 * uint128.inl
 *
 *  Created on: 04/11/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef UINT128_INL_
#define UINT128_INL_

#include <hydra/detail/Config.h>

namespace hydra {

namespace random {

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T> operator>>(uint128_t a, const T & b) {

	a >>= b;
	return a;
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator<<(uint128_t a, const T & b) {

	a <<= b;
	return a;
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T> operator+(uint128_t a, const T & b) {

	return uint128_t::add128(a, b);
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator-(uint128_t a, const T & b) {

	return uint128_t::sub128(a, (uint128_t)b);
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator*(uint128_t a, const T & b) {

	return uint128_t::mul128(a, b);
}

template <typename T>
__hydra_host__ __hydra_device__
inline T operator/(uint128_t x, const T & v) {

	return uint128_t::divide_u128_to_u64(x, (uint64_t)v);
}

template <typename T>
__hydra_host__ __hydra_device__
inline T operator%(uint128_t x, const T & v) {

	uint64_t res;
	uint128_t::divide_u128_to_u64(x, v, &res);
	return (T)res;
}

__hydra_host__ __hydra_device__
inline bool operator<(uint128_t const&  a, uint128_t  const& b) {

	return uint128_t::is_less_than(a, b);
}

__hydra_host__ __hydra_device__
inline bool operator>(uint128_t  const& a, uint128_t  const& b) {

	return uint128_t::is_greater_than(a, b);
}

__hydra_host__ __hydra_device__
inline bool operator<=(uint128_t  const& a, uint128_t const&  b) {

	return uint128_t::is_less_than_or_equal(a, b);
}

__hydra_host__ __hydra_device__
inline bool operator>=(uint128_t  const& a, uint128_t  const& b) {

	return uint128_t::is_greater_than_or_equal(a, b);
}

__hydra_host__ __hydra_device__
inline bool operator==(uint128_t const&  a, uint128_t  const& b) {

	return uint128_t::is_equal_to(a, b);
}

__hydra_host__ __hydra_device__
inline bool operator!=(uint128_t const&  a, uint128_t const&  b) {

	return uint128_t::is_not_equal_to(a, b);
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator|(uint128_t a, const T & b) {

	return uint128_t::bitwise_or(a, (uint128_t)b);
}

template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator&(uint128_t a, const T & b) {

	return uint128_t::bitwise_and(a, (uint128_t)b);
}


template <typename T>
__hydra_host__ __hydra_device__
inline detail::check_uint128_t<T>  operator^(uint128_t a, const T & b) {

	return uint128_t::bitwise_xor(a, (uint128_t)b);
}

__hydra_host__ __hydra_device__
inline uint128_t operator~(uint128_t a) {

	return uint128_t::bitwise_not(a);
}

__hydra_host__ __hydra_device__
inline uint128_t min(uint128_t const& a, uint128_t const& b) {

	return a < b ? a : b;
}

__hydra_host__ __hydra_device__
inline uint128_t max(uint128_t const& a, uint128_t const& b) {

	return a > b ? a : b;
}

__hydra_host__ __hydra_device__
inline uint64_t clz128(uint128_t const& x){

	uint64_t res;

	res = x.hi != 0 ? uint128_t::clz64(x.hi) : 64 + uint128_t::clz64(x.lo);

	return res;
}

//  iostream
//------------------
inline std::ostream & operator<<(std::ostream & out, uint128_t x)
{
	std::vector<uint16_t> rout;
	uint64_t v = 10, r = 0;
	if (x == 0) {
		out << "0";
		return out;
	}
	do {
		x = uint128_t::divide_u128_to_u128(x, v, &r);
		rout.push_back(r);
	} while(x != 0);
	for(std::reverse_iterator<std::vector<uint16_t>::iterator> rit = rout.rbegin(); rit != rout.rend(); rit++){
		out << *rit;
	}
	return out;
}

}  // namespace random

}  // namespace hydra

#endif /* UINT128_INL_ */
