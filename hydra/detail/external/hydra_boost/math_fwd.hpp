//  Boost math_fwd.hpp header file  ------------------------------------------//

//  (C) Copyright Hubert Holin and Daryle Walker 2001-2002.  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org/libs/math for documentation.

#ifndef HYDRA_BOOST_MATH_FWD_HPP
#define HYDRA_BOOST_MATH_FWD_HPP

namespace boost
{
namespace math
{


//  From <hydra/detail/external/hydra_boost/math/quaternion.hpp>  ----------------------------------------//

template < typename T >
    class quaternion;

// Also has many function templates (including operators)


//  From <hydra/detail/external/hydra_boost/math/octonion.hpp>  ------------------------------------------//

template < typename T >
    class octonion;

template < >
    class octonion< float >;
template < >
    class octonion< double >;
template < >
    class octonion< long double >;

}  // namespace math
}  // namespace boost


#endif  // HYDRA_BOOST_MATH_FWD_HPP
