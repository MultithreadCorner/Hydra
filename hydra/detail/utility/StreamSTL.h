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
#include <cstdio>
#include <string>
#include <cassert>
#include <memory>


namespace hydra {

/** array streamer helper **/
template<  size_t N, typename T, size_t I>
inline typename std::enable_if<(I==N), void>::type
stream_array_helper(std::ostream& , std::array<T,N> const& )
{ }

template< size_t N, typename T, size_t I=0>
inline typename std::enable_if< (I < N), void>::type
stream_array_helper(std::ostream& os, std::array<T,N> const&  obj)
{
 char separator = (I==N-1)?char(0):',';
 os << " " << std::get<I>(obj) << separator ;
 stream_array_helper<N, T, I+1>(os,obj);
}


/* stream std::array */
template<size_t N, typename T>
inline std::ostream& operator<<(std::ostream& os, std::array<T, N> const&  obj)
{
  os << "{"; stream_array_helper(os, obj); os << "}";

  return os;
}



/** tuple streamer helper **/
template<size_t I, typename ...T>
inline typename std::enable_if<(I==sizeof ...(T)), void>::type
stream_tuple_helper(std::ostream& , std::tuple<T...> const&  )
{ }

template<size_t I=0, typename ...T>
inline typename std::enable_if< (I < sizeof ...(T)), void>::type
 stream_tuple_helper(std::ostream& os, std::tuple<T...> const&  obj)
{
 char separator = (I==sizeof ...(T)-1)?char(0):char(',');
 os << char(' ')<< std::get<I>(obj)<< separator;
 stream_tuple_helper<I+1, T...>(os,obj);
}

/* stream std::tuple */
template<typename ...T>
inline std::ostream& operator<<(std::ostream& os, std::tuple<T...> const&  obj)
{
  os << char('{'); stream_tuple_helper(os,obj);  os << char('}');

  return os;
}

/* stream std::pair */
template<typename T1, typename T2>
inline std::ostream& operator<<(std::ostream& os, std::pair<T1,T2> const&  obj)
{
  os << char('{')<<  obj.first <<char(',')<< char(' ')<< obj.second << char('}');

  return os;
}


template< typename... Args >
std::string GetFormatedString( const char* format, Args... args )
{
  int length = std::snprintf( nullptr, 0, format, args... );
  assert( length >= 0 );

  std::unique_ptr<char[]> buf(new char[length + 1]);
  std::snprintf( buf.get(), length + 1, format, args... );

  std::string str( buf.get() );

  return std::move(str);
}

template< typename... Args >
void PrintToStream(std::ostream &ostream, const char* format, Args... args )
{
   ostream << GetFormatedString(format, args...);
}

}  // namespace hydra



#endif /* STREAMSTL_H_ */
