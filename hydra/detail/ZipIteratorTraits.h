/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * ZipIteratorTraits.h
 *
 *  Created on: 22/05/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ZIPITERATORTRAITS_H_
#define ZIPITERATORTRAITS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/detail/ZipIteratorTraits.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include <type_traits>

namespace hydra {

namespace detail {

template<typename T >
struct is_zip_iterator: std::false_type {};

template<class ...T>
struct is_zip_iterator< hydra_thrust::zip_iterator<T...> >: std::true_type {};

template<typename ...T>
struct merged_zip_iterator;

template<typename ...T>
struct merged_zip_iterator<hydra_thrust::zip_iterator<T...>>
{
    typedef hydra_thrust::zip_iterator<T...> type;
};

template<typename ...T, typename ...U>
struct merged_zip_iterator<hydra_thrust::zip_iterator<T...>, hydra_thrust::zip_iterator<U...>> :
    merged_zip_iterator<hydra_thrust::zip_iterator<T..., U...> > {};

template<typename ...Z, typename ...T, typename ...U>
struct merged_zip_iterator< hydra_thrust::zip_iterator<T...>, hydra_thrust::zip_iterator<U...>, Z...>:
    merged_zip_iterator< hydra_thrust::zip_iterator<T..., U...>, Z... > {};




}  // namespace detail

}  // namespace hydra

#endif /* ZIPITERATORTRAITS_H_ */
