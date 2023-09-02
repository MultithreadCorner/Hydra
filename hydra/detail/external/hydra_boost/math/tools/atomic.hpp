///////////////////////////////////////////////////////////////////////////////
//  Copyright 2017 John Maddock
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_ATOMIC_DETAIL_HPP
#define HYDRA_BOOST_MATH_ATOMIC_DETAIL_HPP

#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/cxx03_warn.hpp>

#ifdef HYDRA_BOOST_HAS_THREADS
#include <atomic>

namespace hydra_boost {
   namespace math {
      namespace detail {
#if (ATOMIC_INT_LOCK_FREE == 2) && !defined(HYDRA_BOOST_MATH_NO_ATOMIC_INT)
         typedef std::atomic<int> atomic_counter_type;
         typedef std::atomic<unsigned> atomic_unsigned_type;
         typedef int atomic_integer_type;
         typedef unsigned atomic_unsigned_integer_type;
#elif (ATOMIC_SHORT_LOCK_FREE == 2) && !defined(HYDRA_BOOST_MATH_NO_ATOMIC_INT)
         typedef std::atomic<short> atomic_counter_type;
         typedef std::atomic<unsigned short> atomic_unsigned_type;
         typedef short atomic_integer_type;
         typedef unsigned short atomic_unsigned_type;
#elif (ATOMIC_LONG_LOCK_FREE == 2) && !defined(HYDRA_BOOST_MATH_NO_ATOMIC_INT)
         typedef std::atomic<long> atomic_unsigned_integer_type;
         typedef std::atomic<unsigned long> atomic_unsigned_type;
         typedef unsigned long atomic_unsigned_type;
         typedef long atomic_integer_type;
#elif (ATOMIC_LLONG_LOCK_FREE == 2) && !defined(HYDRA_BOOST_MATH_NO_ATOMIC_INT)
         typedef std::atomic<long long> atomic_unsigned_integer_type;
         typedef std::atomic<unsigned long long> atomic_unsigned_type;
         typedef long long atomic_integer_type;
         typedef unsigned long long atomic_unsigned_integer_type;
#elif !defined(HYDRA_BOOST_MATH_NO_ATOMIC_INT)
#  define HYDRA_BOOST_MATH_NO_ATOMIC_INT
#endif
      } // Namespace detail
   } // Namespace math
} // Namespace boost

#else
#  define HYDRA_BOOST_MATH_NO_ATOMIC_INT
#endif // HYDRA_BOOST_HAS_THREADS

#endif // HYDRA_BOOST_MATH_ATOMIC_DETAIL_HPP
