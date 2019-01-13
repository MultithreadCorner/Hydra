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
 * Collection.h
 *
 *  Created on: Oct 28, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COLLECTION_H_
#define COLLECTION_H_

#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>

namespace hydra {

/**
 * @ingroup generic
 */
#define _DeclareStorable(class_name,...) \
public: \
typedef decltype( HYDRA_EXTERNAL_NS::thrust::make_tuple(__VA_ARGS__)) args_type; \
typedef void hydra_convertible_to_tuple_tag; \
template<typename ...T> \
__hydra_host__ __hydra_device__ \
class_name( HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& t) \
{ HYDRA_EXTERNAL_NS::thrust::tie(__VA_ARGS__) = t; } \
template<typename ...T> \
__hydra_host__ __hydra_device__ \
class_name& operator= ( HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& t ) \
{HYDRA_EXTERNAL_NS::thrust::tie(__VA_ARGS__) = t;\
return *this; } \
template<typename ...T> \
__hydra_host__ __hydra_device__ \
class_name& operator= (HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references<T&...> const&  t ) \
{HYDRA_EXTERNAL_NS::thrust::tie(__VA_ARGS__) = t; \
return *this; } \
template<typename ...T> \
__hydra_host__ __hydra_device__ \
operator HYDRA_EXTERNAL_NS::thrust::tuple<T...> () { return HYDRA_EXTERNAL_NS::thrust::make_tuple(__VA_ARGS__); } \
template<typename ...T> \
__hydra_host__ __hydra_device__ \
 operator HYDRA_EXTERNAL_NS::thrust::tuple<T...> () const { return HYDRA_EXTERNAL_NS::thrust::make_tuple(__VA_ARGS__); } \


}  // namespace hydra

#endif /* COLLECTION_H_ */
