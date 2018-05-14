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

template <typename T>
    auto is_iterable_impl(int)
    -> decltype (
        hydra::begin(std::declval<T&>()) != hydra::end(std::declval<T&>()),
        void(), //'operator ,' overload ?
        ++std::declval<decltype(hydra::begin(std::declval<T&>()))&>(),
        void(*hydra::begin(std::declval<T&>())),
        std::true_type{});

    template <typename T>
    std::false_type is_iterable_impl(...);

template <typename T>
using is_iterable = decltype(is_iterable_impl<T>(0));


template <typename T>
    auto is_reverse_iterable_impl(int)
    -> decltype (
        hydra::rbegin(std::declval<T&>()) != hydra::rend(std::declval<T&>()),
        void(), //'operator ,' overload ?
        ++std::declval<decltype(hydra::rbegin(std::declval<T&>()))&>(),
        void(*hydra::rbegin(std::declval<T&>())),
        std::true_type{});

    template <typename T>
    std::false_type is_reverse_iterable_impl(...);

template <typename T>
using is_reverse_iterable = decltype(is_reverse_iterable_impl<T>(0));


}  // namespace detail

}  // namespace hydra



#endif /* ITERABLE_TRAITS_H_ */
