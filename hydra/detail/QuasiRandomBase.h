/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
 * QuasiRandomBase.h
 *
 *  Created on: 04/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef QUASIRANDOMBASE_H_
#define QUASIRANDOMBASE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Exception.h>
#include <hydra/Tuple.h>

#include <limits>
#include <type_traits>
#include <cstdint>


namespace hydra {

namespace detail {



template<typename DerivedT, typename LatticeT, typename SizeT>
class quasi_random_base
{

public:

	typedef SizeT size_type;
	typedef typename LatticeT::value_type result_type;

	 __hydra_host__ __hydra_device__
	quasi_random_base():
			lattice(),
			quasi_state()
	{
		derived().seed();
	}

	 __hydra_host__ __hydra_device__
	quasi_random_base(quasi_random_base<DerivedT,LatticeT, SizeT> const& other):
		lattice(other.lattice),
		curr_elem(other.curr_elem),
		seq_count(other.seq_count)
	{
#pragma unroll LatticeT::lattice_dimension
		for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
			quasi_state[i] = other.quasi_state[i];
	}

	 __hydra_host__ __hydra_device__
	quasi_random_base<DerivedT,LatticeT, SizeT>
	operator=(quasi_random_base<DerivedT,LatticeT, SizeT> const& other){

		 if(this==&other) return *this;

		lattice = other.lattice;
		curr_elem = other.curr_elem;
		seq_count = other.seq_count;

#pragma unroll LatticeT::lattice_dimension
		for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
			quasi_state[i] = other.quasi_state[i];

		return *this;
	}

	/**
	 *Returns: The dimension of of the quasi-random domain.
	 */
	 __hydra_host__ __hydra_device__
	constexpr unsigned dimension() const {
		return LatticeT::lattice_dimension;
	}

	 /**
	  * Returns: Returns a successive element of an s-dimensional
	  * (s = X::dimension()) vector at each invocation. When all elements are
	  * exhausted, X::operator() begins anew with the starting element of a
	  * subsequent s-dimensional vector.
	  *
	  */
	 __hydra_host__ __hydra_device__
	inline result_type operator()()
	{
		return curr_elem != dimension() ? load_cached(): next_state();
	}

	//!Fills a range with quasi-random values.
	template<typename ...T>
	 __hydra_host__ __hydra_device__
	inline typename std::enable_if<
	sizeof...(T)==LatticeT::lattice_dimension, void>::type
	generate(hydra::tuple<T...>& data){
		generate_helper(data);
	}

	//!Effects: Advances *this state as if z consecutive
	//!X::operator() invocations were executed.
	//!
	//!Throws: range_error.
	 __hydra_host__ __hydra_device__
	inline void discard(uintmax_t z)
	{
		 const std::size_t dimension_value = dimension();

		// Compiler knows how to optimize subsequent x / y and x % y
		// statements. In fact, gcc does this even at -O1, so don't
		// be tempted to "optimize" % via subtraction and multiplication.

		uintmax_t vec_n = z / dimension_value;
		std::size_t carry = curr_elem + (z % dimension_value);

		vec_n += carry / dimension_value;
		carry  = carry % dimension_value;

		// Avoid overdiscarding by branchlessly correcting the triple
		// (D, S + 1, 0) to (D, S, D) (see equality operator)
		const bool corr = (!carry) & static_cast<bool>(vec_n);

		// Discards vec_n (with correction) consecutive s-dimensional vectors
		discard_vector(vec_n - static_cast<uintmax_t>(corr));

		// Sets up the proper position of the element-to-read
		// curr_elem = carry + corr*dimension_value
		curr_elem = carry ^ (-static_cast<std::size_t>(corr) & dimension_value);

	}

	//!Returns true if the two generators will produce identical sequences of outputs.
	 __hydra_host__ __hydra_device__
	friend bool operator==(const quasi_random_base& x,
			const quasi_random_base& y)
		  {
		const std::size_t dimension_value = x.dimension();

		// Note that two generators with different seq_counts and curr_elems can
		// produce the same sequence because the generator triple
		// (D, S, D) is equivalent to (D, S + 1, 0), where D is dimension, S -- get_seq_count,
		// and the last one is curr_elem.

		return (dimension_value == y.dimension()) &&
				// |x.seq_count - y.get_seq_count| <= 1
				!((x.seq_count < y.seq_count ?
						y.seq_count - x.seq_count :
						x.seq_count - y.seq_count)> static_cast<size_type>(1)) &&
						// Potential overflows don't matter here, since we've already ascertained
						// that sequence counts differ by no more than 1, so if they overflow, they
						// can overflow together.
						(x.seq_count + (x.curr_elem / dimension_value) == y.seq_count + (y.curr_elem / dimension_value)) &&
						(x.curr_elem % dimension_value == y.curr_elem % dimension_value);
		  }

	//!Returns true if the two generators will produce different sequences of outputs.
	 __hydra_host__ __hydra_device__
	friend bool operator!=(const quasi_random_base& lhs,
			const quasi_random_base& rhs)
				  {  return !(lhs == rhs); }

protected:

	//typedef std::vector<result_type> state_type;
	typedef  result_type* state_iterator;


// Getters
	 __hydra_host__ __hydra_device__
	inline size_type curr_seq() const {
		return seq_count;
	}

	 __hydra_host__ __hydra_device__
	inline state_iterator state_begin() {
		return &(quasi_state[0]);
	}

	 __hydra_host__ __hydra_device__
	inline state_iterator state_end() {
		return &(quasi_state[0]) + LatticeT::lattice_dimension;
	}

	// Setters
	 __hydra_host__ __hydra_device__
	inline void reset_seq(size_type seq){

		seq_count = seq;
		curr_elem = 0u;
	}

private:

	 template<typename T, unsigned I>
	 __hydra_host__ __hydra_device__
	 inline typename std::enable_if< (I== LatticeT::lattice_dimension), void>::type
	 generate_helper(T& ){ }

	 template<typename T, unsigned I=0>
	 __hydra_host__ __hydra_device__
	 inline typename std::enable_if< (I< LatticeT::lattice_dimension), void>::type
	 generate_helper(T& data){
		 hydra::get<I>(data)=this->operator()();
		 generate_helper<T,I+1>(data);
	 }



	 __hydra_host__ __hydra_device__
	inline DerivedT& derived()
	 {
		return *const_cast< DerivedT*>( (static_cast<DerivedT* >(this)));
	}

	// Load the result from the saved state.
	 __hydra_host__ __hydra_device__
	inline result_type load_cached()
	{
		return quasi_state[curr_elem++];
	}

	 __hydra_host__ __hydra_device__
	inline 	result_type next_state()
	{
		size_type new_seq = seq_count;

		if (HYDRA_HOST_LIKELY(++new_seq > seq_count))
		{
			derived().compute_seq(new_seq);
			reset_seq(new_seq);
			return load_cached();
		}

		HYDRA_EXCEPTION("hydra::quasi_random_base: next_state overflow. Returning current state.")
		return load_cached();
	}

	// Discards z consecutive s-dimensional vectors,
	// and preserves the position of the element-to-read
	 __hydra_host__ __hydra_device__
	inline void discard_vector(uintmax_t z)
	{
		const uintmax_t max_z = std::numeric_limits<size_type>::max() - seq_count;

		// Don't allow seq_count + z overflows here
		if (max_z < z){
			HYDRA_EXCEPTION("hydra::quasi_random_base: discard_vector. Returning without doing nothing.")
			return ;
		}
		std::size_t tmp = curr_elem;
		derived().seed(static_cast<size_type>(seq_count + z));
		curr_elem = tmp;
	}


	// Member variables are so ordered with the intention
	// that the typical memory access pattern would be
	// incremental. Moreover, lattice is put before quasi_state
	// because we want to construct lattice first. Lattices
	// can do some kind of dimension sanity check (as in
	// dimension_assert below), and if that fails then we don't
	// need to do any more work.
private:
	std::size_t curr_elem;
	size_type seq_count;
protected:
	LatticeT lattice;
private:
	//state_type quasi_state;
	result_type quasi_state[LatticeT::lattice_dimension];
};


}  // namespace detail

}  // namespace hydra


#endif /* QUASIRANDOMBASE_H_ */
