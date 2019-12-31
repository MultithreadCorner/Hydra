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
//#include <boost/random/detail/GrayCodeBase.hpp>
#include <functional> // bit_xor
#include <type_traits>


#include <boost/integer/integer_mask.hpp>

//!\file
//!Describes the gray-coded quasi-random number generator base class template.

namespace hydra {

namespace detail {

template<typename LatticeT>
class GrayCode: public GrayCodeBase< GrayCode<LatticeT>, LatticeT, typename LatticeT::value_type>
{
public:
  typedef typename LatticeT::value_type result_type;
  typedef result_type size_type;

private:
  typedef GrayCode<LatticeT> self_t;
  typedef GrayCodeBase<self_t, LatticeT, size_type> base_t;

  // The base needs to access modifying member f-ns, and we
  // don't want these functions to be available for the public use
  friend class GrayCodeBase<self_t, LatticeT, size_type>;

  // Respect lattice bit_count here
  struct check_nothing {
    inline static void bit_pos(unsigned) {}
    inline static void code_size(size_type) {}
  };
  struct check_bit_range {
    static void raise_bit_count() {
      throw std::exception( std::range_error("GrayCode: bit_count") );
    }
    inline static void bit_pos(unsigned bit_pos) {
      if (bit_pos >= LatticeT::bit_count)
        raise_bit_count();
    }
    inline static void code_size(size_type code) {
      if (code > (self_t::max)())
        raise_bit_count();
    }
  };

  // We only want to check whether bit pos is outside the range if given bit_count
  // is narrower than the size_type, otherwise checks compile to nothing.
 static_assert(LatticeT::bit_count <= std::numeric_limits<size_type>::digits,
		 "hydra::GrayCode : bit_count in LatticeT' > digits");

  typedef typename std::conditional<
	 std::integral_constant<bool,((LatticeT::bit_count) < std::numeric_limits<size_type>::digits)>::value
    , check_bit_range
    , check_nothing
  >::type check_bit_range_t;

public:
  //!Returns: Tight lower bound on the set of values returned by operator().
  //!
  //!Throws: nothing.
  constexpr static const result_type Min(){
	  return 0;
  }

  //!Returns: Tight upper bound on the set of values returned by operator().
  //!
  //!Throws: nothing.
  constexpr static const result_type Max(){
	  return low_bits_mask_t<LatticeT::bit_count>::sig_bits;
  }

  explicit GrayCode(std::size_t dimension)
    : base_t(dimension)
  {}

  // default copy c-tor is fine

  // default assignment operator is fine

  void seed()
  {
    set_zero_state();
    update_quasi(0);
    base_t::reset_seq(0);
  }

  void seed(const size_type init)
  {
    if (init != this->curr_seq())
    {
      // We don't want negative seeds.
      check_seed_sign(init);

      size_type seq_code = boost::next(init);
      if(HYDRA_HOST_UNLIKELY(!(init < seq_code)))
    	  throw std::exception( std::range_error("GrayCode: seed") );

      seq_code ^= (seq_code >> 1);
      // Fail if we see that seq_code is outside bit range.
      // We do that before we even touch engine state.
      check_bit_range_t::code_size(seq_code);

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


  void compute_seq(size_type seq)
  {
    // Find the position of the least-significant zero in sequence count.
    // This is the bit that changes in the Gray-code representation as
    // the count is advanced.
    // Xor'ing with max() has the effect of flipping all the bits in seq,
    // except for the sign bit.
    unsigned r = detail::lsb(seq ^ (self_t::Max)());
    check_bit_range_t::bit_pos(r);
    update_quasi(r);
  }

  void update_quasi(unsigned r)
  {
    // Calculate the next state.
    std::transform(this->state_begin(), this->state_end(),
      this->lattice.iter_at(r * this->dimension()), this->state_begin(),
      std::bit_xor<result_type>());
  }

  void set_zero_state()
  {
    std::fill(this->state_begin(), this->state_end(), result_type /*zero*/ ());
  }
};


}  // namespace detail

}  // namespace hydra
#endif /* GRAYCODE_H_ */
