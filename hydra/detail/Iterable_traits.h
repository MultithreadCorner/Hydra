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
 * Iterable_traits.h
 *
 *  Created on: 12/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ITERABLE_TRAITS_H_
#define ITERABLE_TRAITS_H_

#include <hydra/Iterator.h>
#include <utility>

namespace hydra {

namespace detail {

// Primary template
template <typename T, typename U= int>
struct is_iterable : std::false_type { };

// Specialization for U = int
template <typename T>
struct is_iterable<T, decltype (
        hydra::begin(std::declval<T&>()) != hydra::end(std::declval<T&>()),
        void(), //'operator ,' overload ?
        ++std::declval<decltype(hydra::begin(std::declval<T&>()))&>(),
        void(*hydra::begin(std::declval<T&>())),0)> : std::true_type { };


// Primary template
template <typename T, typename U= int>
struct is_reverse_iterable : std::false_type { };

// Specialization for U = int
template <typename T>
struct is_reverse_iterable<T, decltype (
        hydra::rbegin(std::declval<T&>()) != hydra::rend(std::declval<T&>()),
        void(), //'operator ,' overload ?
        ++std::declval<decltype(hydra::rbegin(std::declval<T&>()))&>(),
        void(*hydra::rbegin(std::declval<T&>())),0)> : std::true_type { };

}  // namespace detail

}  // namespace hydra



#endif /* ITERABLE_TRAITS_H_ */
