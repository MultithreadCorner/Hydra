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
 * Integer.h
 *
 *  Created on: 02/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef INTEGER_H_
#define INTEGER_H_

#include <hydra/detail/Config.h>
#include <type_traits>
#include <limits>
#include <cstdint>
#include <climits>

//
// We simply cannot include this header on gcc without getting copious warnings of the kind:
//
// boost/integer.hpp:77:30: warning: use of C99 long long integer constant
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

  //  Helper templates  ------------------------------------------------------//

  //  fast integers from least integers
  //  int_fast_t<> works correctly for unsigned too, in spite of the name.
  template< typename LeastInt >
  struct int_fast_t
  {
     typedef LeastInt fast;
     typedef fast     type;
  }; // imps may specialize

  namespace impl{

  //  convert category to type
  template< int Category > struct int_least_helper {}; // default is empty
  template< int Category > struct uint_least_helper {}; // default is empty

  //  specializatons: 1=long, 2=int, 3=short, 4=signed char,
  //     6=unsigned long, 7=unsigned int, 8=unsigned short, 9=unsigned char
  //  no specializations for 0 and 5: requests for a type > long are in error

  template<> struct int_least_helper<1> { typedef long long least; };
  template<> struct int_least_helper<2> { typedef long least; };
  template<> struct int_least_helper<3> { typedef int least; };
  template<> struct int_least_helper<4> { typedef short least; };
  template<> struct int_least_helper<5> { typedef signed char least; };

  template<> struct uint_least_helper<1> { typedef unsigned long long least; };
  template<> struct uint_least_helper<2> { typedef unsigned long least; };
  template<> struct uint_least_helper<3> { typedef unsigned int least; };
  template<> struct uint_least_helper<4> { typedef unsigned short least; };
  template<> struct uint_least_helper<5> { typedef unsigned char least; };

  template <int Bits>
  struct exact_signed_base_helper{};
  template <int Bits>
  struct exact_unsigned_base_helper{};

  template <> struct exact_signed_base_helper<sizeof(signed char)* CHAR_BIT> { typedef signed char exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned char)* CHAR_BIT> { typedef unsigned char exact; };

#if USHRT_MAX != UCHAR_MAX
  template <> struct exact_signed_base_helper<sizeof(short)* CHAR_BIT> { typedef short exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned short)* CHAR_BIT> { typedef unsigned short exact; };
#endif

#if UINT_MAX != USHRT_MAX
  template <> struct exact_signed_base_helper<sizeof(int)* CHAR_BIT> { typedef int exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned int)* CHAR_BIT> { typedef unsigned int exact; };
#endif

#if ULONG_LONG_MAX != ULONG_MAX
  template <> struct exact_signed_base_helper<sizeof(long long)* CHAR_BIT> { typedef long long exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned long long)* CHAR_BIT> { typedef unsigned  long long exact; };
#endif


  } // namespace impl

  //  integer templates specifying number of bits  ---------------------------//

  //  signed
  template< int Bits >   // bits (including sign) required
  struct int_t : public impl::exact_signed_base_helper<Bits>
  {
      static_assert(Bits <= (int)(sizeof(long long) * CHAR_BIT),
         "No suitable signed integer type with the requested number of bits is available.");

      typedef typename impl::int_least_helper
        < (Bits   <= (int)(sizeof(long long) * CHAR_BIT)) +
          (Bits-1 <= std::numeric_limits<long>::digits) +
          (Bits-1 <= std::numeric_limits<int>::digits)  +
          (Bits-1 <= std::numeric_limits<short>::digits) +
          (Bits-1 <= std::numeric_limits<signed char>::digits)
        >::least  least;

      typedef typename int_fast_t<least>::type  fast;
  };

  //  unsigned
  template< int Bits >   // bits required
  struct uint_t : public impl::exact_unsigned_base_helper<Bits>
  {
     static_assert(Bits <= (int)(sizeof(unsigned long long) * CHAR_BIT),
         "No suitable unsigned integer type with the requested number of bits is available.");

      typedef typename impl::uint_least_helper
        < (Bits <= (int)(sizeof(unsigned long long) * CHAR_BIT)) +
          (Bits <= ::std::numeric_limits<unsigned long>::digits) +
          (Bits <= ::std::numeric_limits<unsigned int>::digits) +
          (Bits <= ::std::numeric_limits<unsigned short>::digits) +
          (Bits <= ::std::numeric_limits<unsigned char>::digits)
        >::least  least;

      typedef typename int_fast_t<least>::type  fast;
  };


} // namespace detail


}  // namespace hydra

#endif /* INTEGER_H_ */
