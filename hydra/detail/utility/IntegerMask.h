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
 * IntegerMask.h
 *
 *  Created on: 02/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef INTEGERMASK_H_
#define INTEGERMASK_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Integer.h>
#include <climits>  // for UCHAR_MAX, etc.
#include <cstddef>  // for std::size_t
#include <limits>  // for std::numeric_limits

//
// We simply cannot include this header on gcc without getting copious warnings of the kind:
//
// boost/integer/integer_mask.hpp:93:35: warning: use of C99 long long integer constant
//
// And yet there is no other reasonable implementation, so we declare this a system header
// to suppress these warnings.
//
#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

namespace hydra {

namespace detail
{


//  Specified single-bit mask class declaration  -----------------------------//
//  (Lowest bit starts counting at 0.)

template < std::size_t Bit >
struct high_bit_mask_t
{
    typedef typename uint_t<(Bit + 1)>::least  least;
    typedef typename uint_t<(Bit + 1)>::fast   fast;

    static const  least high_bit = (least( 1u ) << Bit) ;
    static const  fast  high_bit_fast = (fast( 1u ) << Bit) ;

    static const  std::size_t bit_position = Bit ;

};  //high_bit_mask_t


//  Specified bit-block mask class declaration  ------------------------------//
//  Makes masks for the lowest N bits
//  (Specializations are needed when N fills up a type.)


template < std::size_t Bits >
struct low_bits_mask_t
{
    typedef typename uint_t<Bits>::least  least;
    typedef typename uint_t<Bits>::fast   fast;

    static const least sig_bits = least(~(least(~(least( 0u ))) << Bits )) ;
    static const fast sig_bits_fast = fast(sig_bits) ;

    static const std::size_t bit_count = Bits ;

};  //low_bits_mask_t


#define HYDRA_LOW_BITS_MASK_SPECIALIZE( Type )                                  \
  template <  >  struct low_bits_mask_t< std::numeric_limits<Type>::digits >  { \
      typedef std::numeric_limits<Type>           limits_type;                  \
      typedef uint_t<limits_type::digits>::least  least;                        \
      typedef uint_t<limits_type::digits>::fast   fast;                         \
      static const least sig_bits = (~( least(0u) )) ;                          \
      static const fast sig_bits_fast = fast(sig_bits) ;                        \
      static const std::size_t bit_count = limits_type::digits ;                \
  }

HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned char );

#if USHRT_MAX > UCHAR_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned short );
#endif

#if UINT_MAX > USHRT_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned int );
#endif

#if ULONG_MAX > UINT_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned long );
#endif

#if (defined(ULLONG_MAX) && (ULLONG_MAX > ULONG_MAX))
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned long long );
#endif



#undef HYDRA_LOW_BITS_MASK_SPECIALIZE


}  // namespace detail

}  // namespace hydra





#endif /* INTEGERMASK_H_ */
