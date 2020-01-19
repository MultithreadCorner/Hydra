/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * GrayCode.h
 *
 *  Created on: 28/12/2019
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GRAYCODE_H_
#define GRAYCODE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/LSB.h>
#include <hydra/detail/utility/IntegerMask.h>

#include <hydra/detail/QuasiRandomBase.h>
#include <functional> // bit_xor
#include <type_traits>


//!\file
//!Describes the gray-coded quasi-random number generator base class template.

namespace hydra {

namespace detail {

template<typename LatticeT>
class gray_code: public quasi_random_base< gray_code<LatticeT>, LatticeT, typename LatticeT::value_type>
{

public:
  typedef typename LatticeT::value_type result_type;
  typedef result_type size_type;

   static const result_type min=0;
   static const result_type max=
		  low_bits_mask_t<LatticeT::bit_count>::sig_bits;

private:
  typedef gray_code<LatticeT> self_t;
  typedef quasi_random_base<self_t, LatticeT, size_type> base_t;

  // The base needs to access modifying member f-ns, and we
  // don't want these functions to be available for the public use
  friend class quasi_random_base<self_t, LatticeT, size_type>;

  // Respect lattice bit_count here
  struct check_nothing {
	__hydra_host__ __hydra_device__
    inline static void bit_pos(unsigned) {}

	__hydra_host__ __hydra_device__
    inline static void code_size(size_type) {}
  };
  struct check_bit_range {

	__hydra_host__ __hydra_device__
	static void raise_bit_count() {
    	HYDRA_EXCEPTION("GrayCode: bit_count" );
    }

	__hydra_host__ __hydra_device__
    inline static void bit_pos(unsigned bit_pos) {
      if (bit_pos >= LatticeT::bit_count)
        raise_bit_count();
    }

	__hydra_host__ __hydra_device__
    inline static void code_size(size_type code) {
      if (code > (self_t::max))
        raise_bit_count();
    }
  };

  // We only want to check whether bit pos is outside the range if given bit_count
  // is narrower than the size_type, otherwise checks compile to nothing.
 static_assert(LatticeT::bit_count <= std::numeric_limits<size_type>::digits,
		 "hydra::gray_code : bit_count in LatticeT' > digits");

  typedef typename std::conditional<
	 std::integral_constant<bool,((LatticeT::bit_count) < std::numeric_limits<size_type>::digits)>::value
    , check_bit_range
    , check_nothing
  >::type check_bit_range_t;


public:

 __hydra_host__ __hydra_device__
  explicit gray_code(): base_t() {}

  // default copy c-tor is fine

  // default assignment operator is fine

 __hydra_host__ __hydra_device__
 inline  void seed()
  {
    set_zero_state();
    update_quasi(0);
    base_t::reset_seq(0);
  }

 __hydra_host__ __hydra_device__
  inline  void seed(const size_type init)
  {
    if (init != this->curr_seq())
    {
      // We don't want negative seeds.
     // check_seed_sign(init);

      size_type seq_code =  init+1;
     if(HYDRA_HOST_UNLIKELY(!(init < seq_code))){
    	 HYDRA_EXCEPTION("hydra::gray_code : seed overflow. Returning without set seed")
         return ;
     }

      seq_code ^= (seq_code >> 1);
      // Fail if we see that seq_code is outside bit range.
      // We do that before we even touch engine state.
      check_bit_range_t::code_size(seq_code); //<< uncoment for debug

      set_zero_state();
      for (unsigned r = 0; seq_code != 0; ++r, seq_code >>= 1)
      {
        if (seq_code & static_cast<size_type>(1))
          update_quasi(r);
      }
    }
    // Everything went well, set the new seq count
    base_t::reset_seq(init);
  }

private:

 __hydra_host__ __hydra_device__
  inline  void compute_seq(size_type seq)
  {
    // Find the position of the least-significant zero in sequence count.
    // This is the bit that changes in the Gray-code representation as
    // the count is advanced.
    // Xor'ing with max() has the effect of flipping all the bits in seq,
    // except for the sign bit.
    unsigned r = lsb(seq ^ (self_t::max));
    check_bit_range_t::bit_pos(r); //<< uncoment for debug
    update_quasi(r);
  }

 __hydra_host__ __hydra_device__
  inline void update_quasi(unsigned r){

    // Calculate the next state.

	result_type* i= this->state_begin();
	const  result_type* j= this->lattice.iter_at(r * this->dimension());

#pragma unroll LatticeT::lattice_dimension
   for(size_t s=0;s<LatticeT::lattice_dimension; ++s)
	   i[s]=(i[s])^(j[s]);

  }

 __hydra_host__ __hydra_device__
  inline void set_zero_state(){

   result_type* s= this->state_begin();

#pragma unroll LatticeT::lattice_dimension
   for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
    	s[i]=result_type{};
  }

};


}  // namespace detail

}  // namespace hydra
#endif /* GRAYCODE_H_ */
