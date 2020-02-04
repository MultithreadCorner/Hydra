/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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

/* NOTE:
 *
 * The Hydra implementation of Sobol algorithm tries to follow as
 * closely as possible the implementation found in the BOOST library
 * at http://boost.org/libs/random.
 *
 * See:
 *  - Boost Software License, Version 1.0 at http://www.boost.org/LICENSE-1.0
 *  - Primary copyright information for Boost.Random at https://www.boost.org/doc/libs/1_72_0/doc/html/boost_random.html
 *
 */

/*
 * Sobol.h
 *
 *  Created on: 04/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SOBOL_H_
#define SOBOL_H_

#include <algorithm>
#include<hydra/detail/Config.h>
#include<hydra/detail/SobolTable.h>
#include<hydra/detail/GrayCode.h>
#include<hydra/detail/utility/MSB.h>
#include <cassert>

namespace hydra {


namespace detail {

// sobol_lattice sets up the random-number generator to produce a Sobol
// sequence of at most max dims-dimensional quasi-random vectors.
// Adapted from ACM TOMS algorithm 659, see

// http://doi.acm.org/10.1145/42288.214372

template<typename UIntType, unsigned D, unsigned W, typename SobolTables>
struct sobol_lattice
{
	typedef UIntType value_type;

	static_assert(D > 0u, "[hydra::sobol_lattice] Problem: D < 0. (D) - dimension has to be greater than zero.");
	static_assert(D <= HYDRA_SOBOL_MAX_DIMENSION, "[hydra::sobol_lattice] Problem: D > SOBOL_MAX_DIMENSION. (D) - dimension has to be greater than zero.");
	static_assert(W > 0u, "[hydra::sobol_lattice] Problem: W < 0. (W) - bit count has to be greater than zero.");

	static const unsigned bit_count = W;
	static const unsigned lattice_dimension= D;
	static const unsigned storage_size=W*D;


public:
	 __hydra_host__ __hydra_device__
	sobol_lattice(){

		init();
	}

	 __hydra_host__ __hydra_device__
	sobol_lattice(sobol_lattice< UIntType,D,W,SobolTables> const& other ){

#pragma unroll storage_size
		for(unsigned i=0;i<storage_size; ++i)
			bits[i] = other.GetBits()[i];
	}


	sobol_lattice<UIntType,D,W,SobolTables>
	 __hydra_host__ __hydra_device__
	operator=(sobol_lattice< UIntType,D,W,SobolTables> const& other ){
		if(this == &other) return *this;

#pragma unroll storage_size
		for(unsigned i=0;i<storage_size; ++i)
			bits[i] = other.GetBits()[i];
		return *this;

	}

	 __hydra_host__ __hydra_device__
	const  value_type* iter_at(std::size_t n) const
	{
		assert(!(n > storage_size-1 ));
		return bits + n;
	}

	 __hydra_host__ __hydra_device__
	const value_type* GetBits() const {
		return bits;
	}

private:

	 __hydra_host__ __hydra_device__
	inline void init()
	{

		// Initialize direction table in dimension 0
		for (unsigned k = 0; k != bit_count; ++k)
			bits[lattice_dimension*k] = static_cast<value_type>(1);

		// Initialize in remaining dimensions.
		for (std::size_t dim = 1; dim < lattice_dimension; ++dim)
		{
			const typename SobolTables::value_type poly = SobolTables::polynomial(dim-1);

			const unsigned degree = msb(poly); // integer log2(poly)

			// set initial values of m from table
			for (unsigned k = 0; k != degree; ++k)
				bits[lattice_dimension*k + dim] = SobolTables::minit(dim-1, k);

			// Calculate remaining elements for this dimension,
			// as explained in Bratley+Fox, section 2.
			for (unsigned j = degree; j < bit_count; ++j)
			{
				typename SobolTables::value_type p_i = poly;
				const std::size_t bit_offset = lattice_dimension*j + dim;

				bits[bit_offset] = bits[lattice_dimension*(j-degree) + dim];
				for (unsigned k = 0; k != degree; ++k, p_i >>= 1)
				{
					int rem = degree - k;
					bits[bit_offset] ^= ((p_i & 1) * bits[lattice_dimension*(j-rem) + dim]) << rem;
				}
			}
		}

		// Shift columns by appropriate power of 2.
		unsigned p = 1u;
		for (int j = bit_count-1-1; j >= 0; --j, ++p)
		{
			const std::size_t bit_offset = lattice_dimension * j;
			for (std::size_t dim = 0; dim != lattice_dimension; ++dim)
				bits[bit_offset + dim] <<= p;
		}

	}

	//container_type bits;
	value_type bits[storage_size];

};

} // namespace detail

typedef detail::SobolTable default_sobol_table;


//!Instantiations of class template sobol.
//!The sobol_engine uses the algorithm described in
//! \blockquote
//![Bratley+Fox, TOMS 14, 88 (1988)]
//!and [Antonov+Saleev, USSR Comput. Maths. Math. Phys. 19, 252 (1980)]
//! \endblockquote
//!
//!\attention sobol_engine skips trivial zeroes at the start of the sequence. For example, the beginning
//!of the 2-dimensional Sobol sequence in @c uniform_01 distribution will look like this:
//!\code{.cpp}
//!0.5, 0.5,
//!0.75, 0.25,
//!0.25, 0.75,
//!0.375, 0.375,
//!0.875, 0.875,
//!...
//!\endcode

template<typename UIntType,  unsigned D, unsigned W, typename SobolTables = default_sobol_table>
class sobol_engine
		: public detail::gray_code<detail::sobol_lattice<UIntType, D, W, SobolTables>>
{
  typedef detail::sobol_lattice<UIntType, D, W, SobolTables> lattice_t;
  typedef detail::gray_code<lattice_t> base_t;

public:

  static const  UIntType min=0;
  static const  UIntType max=base_t::max;

  __hydra_host__ __hydra_device__
  sobol_engine() : base_t() {}

  __hydra_host__ __hydra_device__
  sobol_engine( UIntType s) : base_t() {
	  base_t::seed(s);
  }

  // default copy c-tor is fine

  // default assignment  is fine

};

/**
 * @attention This specialization of \sobol_engine supports up to 3667 dimensions.
 *
 * Data on the primitive binary polynomials `a` and the corresponding starting values `m`
 * for Sobol sequences in up to 21201 dimensions was taken from
 *
 *  @blockquote
 *  S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections,
 *  SIAM J. Sci. Comput. 30, 2635-2654 (2008).
 *  @endblockquote
 *
 * See the original tables up to dimension 21201: https://web.archive.org/web/20170802022909/http://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
 *
 * For practical reasons the default table uses only the subset of binary polynomials `a` < 2<sup>16</sup>.
 *
 * However, it is possible to provide your own table to \sobol_engine should the default one be insufficient.
 */
template<unsigned D>
using sobol= sobol_engine<uint_least64_t, D, 64u, default_sobol_table> ;

}  // namespace hydra

#endif /* SOBOL_H_ */
