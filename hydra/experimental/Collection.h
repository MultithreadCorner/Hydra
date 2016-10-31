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
 * Collection.h
 *
 *  Created on: Oct 28, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COLLECTION_H_
#define COLLECTION_H_



namespace hydra {

namespace experimental {

#define _DeclareStorable(class_name, args...) \
public: \
typedef decltype( thrust::make_tuple(args)) args_type; \
template<typename ...T> \
__host__ __device__ \
class_name( thrust::tuple<T...> const& t) \
{ thrust::tie(args) = t; } \
template<typename ...T> \
__host__ __device__ \
class_name& operator= ( thrust::tuple<T...> const& t ) \
{thrust::tie(args) = t;\
return *this; } \
__host__ __device__ \
template<typename ...T> \
operator thrust::tuple<T...> () { return thrust::make_tuple(args); } \



}  // namespace experimental

}  // namespace hydra

#endif /* COLLECTION_H_ */
