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
 * QuasiRandomBase.h
 *
 *  Created on: 04/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef QUASIRANDOMBASE_H_
#define QUASIRANDOMBASE_H_

#include <hydra/detail/Config.h>
#include <stdexcept>
#include <vector>
#include <limits>

#include <istream>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <cstdint>


namespace hydra {

namespace detail {

// If the seed is a signed integer type, then we need to
// check that the value is positive:
template <typename Integer>
inline void check_seed_sign(const Integer& v, const std::true_type)
{
  if (v < 0)
  {
    throw std::exception( std::range_error("seed must be a positive integer") );
  }
}
template <typename Integer>
inline void check_seed_sign(const Integer&, const std::false_type) {}

template <typename Integer>
inline void check_seed_sign(const Integer& v)
{
  check_seed_sign(v, std::integral_constant<bool, std::numeric_limits<Integer>::is_signed>());
}


template<typename DerivedT, typename LatticeT, typename SizeT>
class QuasiRandomBase
{
public:
  typedef SizeT size_type;
  typedef typename LatticeT::value_type result_type;

  explicit QuasiRandomBase(std::size_t dimension)
    // Guard against invalid dimensions before creating the lattice
    : lattice(prevent_zero_dimension(dimension))
    , quasi_state(dimension)
  {
    derived().seed();
  }

  // default copy c-tor is fine

  // default assignment operator is fine

  //!Returns: The dimension of of the quasi-random domain.
  //!
  //!Throws: nothing.
  std::size_t dimension() const { return quasi_state.size(); }

  //!Returns: Returns a successive element of an s-dimensional
  //!(s = X::dimension()) vector at each invocation. When all elements are
  //!exhausted, X::operator() begins anew with the starting element of a
  //!subsequent s-dimensional vector.
  //!
  //!Throws: range_error.
  result_type operator()()
  {
    return curr_elem != dimension() ? load_cached(): next_state();
  }

  //!Fills a range with quasi-random values.
  template<typename Iter> void generate(Iter first, Iter last)
  {
    for (; first != last; ++first)
      *first = this->operator()();
  }

  //!Effects: Advances *this state as if z consecutive
  //!X::operator() invocations were executed.
  //!
  //!Throws: range_error.
  void discard(uintmax_t z)
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

  //!Writes the textual representation of the generator to a @c std::ostream.
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const QuasiRandomBase& s)
  {
    os << s.dimension() << " " << s.seq_count << " " << s.curr_elem;
    return os;
  }

  //!Reads the textual representation of the generator from a @c std::istream.
  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, QuasiRandomBase& s)
  {
    std::size_t dim;
    size_type seed;
    uintmax_t z;

    if (is >> dim >> std::ws >> seed >> std::ws >> z) // initialize iff success!
    {
      // Check seed sign before resizing the lattice and/or recomputing state
      check_seed_sign(seed);

      if (s.dimension() != prevent_zero_dimension(dim))
      {
        s.lattice.resize(dim);
        s.quasi_state.resize(dim);
      }
      // Fast-forward to the correct state
      s.derived().seed(seed);
      if (z != 0) s.discard(z);
    }
    return is;
  }

  //!Returns true if the two generators will produce identical sequences of outputs.
  friend bool operator==(const QuasiRandomBase& x, const QuasiRandomBase& y)
  {
    const std::size_t dimension_value = x.dimension();

    // Note that two generators with different seq_counts and curr_elems can
    // produce the same sequence because the generator triple
    // (D, S, D) is equivalent to (D, S + 1, 0), where D is dimension, S -- seq_count,
    // and the last one is curr_elem.

    return (dimension_value == y.dimension()) &&
      // |x.seq_count - y.seq_count| <= 1
      !((x.seq_count < y.seq_count ? y.seq_count - x.seq_count : x.seq_count - y.seq_count)
          > static_cast<size_type>(1)) &&
      // Potential overflows don't matter here, since we've already ascertained
      // that sequence counts differ by no more than 1, so if they overflow, they
      // can overflow together.
      (x.seq_count + (x.curr_elem / dimension_value) == y.seq_count + (y.curr_elem / dimension_value)) &&
      (x.curr_elem % dimension_value == y.curr_elem % dimension_value);
  }

  //!Returns true if the two generators will produce different sequences of outputs.
  friend bool operator!=(const QuasiRandomBase& lhs, const QuasiRandomBase& rhs)
		  {  return !(lhs == rhs); }

protected:

  typedef std::vector<result_type> state_type;
  typedef typename state_type::iterator state_iterator;

  // Getters
  size_type curr_seq() const { return seq_count; }

  state_iterator state_begin() { return quasi_state.begin(); }
  state_iterator state_end() { return quasi_state.end(); }

  // Setters
  void reset_seq(size_type seq)
  {
    seq_count = seq;
    curr_elem = 0u;
  }

private:

  DerivedT& derived() throw()
  {
    return *static_cast<DerivedT * const>(this);
  }

  // Load the result from the saved state.
  result_type load_cached()
  {
    return quasi_state[curr_elem++];
  }

  result_type next_state()
  {
    size_type new_seq = seq_count;

    if (HYDRA_HOST_LIKELY(++new_seq > seq_count))
    {
      derived().compute_seq(new_seq);
      reset_seq(new_seq);
      return load_cached();
    }

    throw std::exception( std::range_error("QuasiRandomBase: next_state") );
  }

  // Discards z consecutive s-dimensional vectors,
  // and preserves the position of the element-to-read
  void discard_vector(uintmax_t z)
  {
    const uintmax_t max_z = std::numeric_limits<size_type>::max() - seq_count;

    // Don't allow seq_count + z overflows here
    if (max_z < z)
      throw std::exception( std::range_error("QuasiRandomBase: discard_vector") );

    std::size_t tmp = curr_elem;
    derived().seed(static_cast<size_type>(seq_count + z));
    curr_elem = tmp;
  }

  static std::size_t prevent_zero_dimension(std::size_t dimension)
  {
    if (dimension == 0)
      throw std::exception( std::invalid_argument("QuasiRandomBase: zero dimension") );
    return dimension;
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
  state_type quasi_state;
};

inline void dimension_assert(const char* generator, std::size_t dim, std::size_t maxdim)
{
  if (!dim || dim > maxdim)
  {
    std::ostringstream os;
    os << "The " << generator << " quasi-random number generator only supports dimensions in range [1; "
      << maxdim << "], but dimension " << dim << " was supplied.";
    throw std::exception( std::invalid_argument(os.str()) );
  }
}

}  // namespace detail

}  // namespace hydra


#endif /* QUASIRANDOMBASE_H_ */
