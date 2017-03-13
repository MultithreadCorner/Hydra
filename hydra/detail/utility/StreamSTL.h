/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * StreamSTL.h
 *
 *  Created on: Mar 13, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef STREAMSTL_H_
#define STREAMSTL_H_

#include <iostream>
#include <array>
#include <tuple>
#include <type_traits>

namespace hydra {


/** array streamer helper **/
template<  size_t N, typename T, size_t I>
typename std::enable_if<(I==N), void>::type
stream_array_helper(std::ostream& os, std::array<T,N> const&  obj)
{ }

template< size_t N, typename T, size_t I=0>
typename std::enable_if< (I < N), void>::type
stream_array_helper(std::ostream& os, std::array<T,N> const&  obj)
{
 os << " " << std::get<I>(obj) 	;
 stream_array_helper<N, T, I+1>(os,obj);
}


/* stream std::array */
template<size_t N, typename T>
std::ostream& operator<<(std::ostream& os, std::array<T, N> const&  obj)
{
  os << "{"; stream_array_helper(os, obj); os << " }";

  return os;
}

/** tuple streamer helper **/
template<size_t I, typename ...T>
typename std::enable_if<(I==sizeof ...(T)), void>::type
stream_tuple_helper(std::ostream& os, std::tuple<T...> const&  obj)
{ }

template<size_t I=0, typename ...T>
typename std::enable_if< (I < sizeof ...(T)), void>::type
stream_tuple_helper(std::ostream& os, std::tuple<T...> const&  obj)
{
 os << " "<< std::get<I>(obj)	;
 stream_tuple_helper<I+1, T...>(os,obj);
}

/* stream std::tuple */
template<typename ...T>
std::ostream& operator<<(std::ostream& os, std::tuple<T...> const&  obj)
{
  os << "("; stream_tuple_helper(os,obj);  os << " )";

  return os;
}

/* stream std::pair */
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1,T2> const&  obj)
{
  os << "("<<  obj.first <<", "<< obj.second << " )";

  return os;
}




}  // namespace hydra



#endif /* STREAMSTL_H_ */
