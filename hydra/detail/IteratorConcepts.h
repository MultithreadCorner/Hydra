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
 * IteratorConcepts.h
 *
 *  Created on: 26/06/2026
 *      Author: Davide Brundu
 */

#ifndef ITERATORCONCEPTS_H_
#define ITERATORCONCEPTS_H_

#include <concepts>
#include <type_traits>

#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/detail/TypeTraits.h>

namespace hydra {

namespace detail {

/**
 * @brief Satisfied when @c T models a Hydra iterable, i.e. exposes @c begin()
 * and @c end() returning iterators (see hydra::detail::is_iterable).
 */
template <typename T>
concept Iterable = is_iterable<T>::value;

/**
 * @brief Satisfied when @c Ts model Hydra iterables, i.e. expose @c begin()
 * and @c end() returning iterators (see hydra::detail::are_iterables).
 */
template <typename ...Ts>
concept Iterables = are_iterables<Ts...>::value;

/**
 * @brief Satisfied when @c T models a Hydra reverse-iterable, i.e. exposes
 * @c rbegin() and @c rend() (see hydra::detail::is_reverse_iterable).
 */
template <typename T>
concept ReverseIterable = is_reverse_iterable<T>::value;

/**
 * @brief Satisfied when @c T models an iterator (see hydra::detail::is_iterator).
 */
template <typename T>
concept Iterator = is_iterator<T>::value;

/**
 * @brief Satisfied when @c Ts model iterators (see hydra::detail::are_iterators).
 */
template <typename ...Ts>
concept Iterators = are_iterators<Ts...>::value;

}  // namespace detail

}  // namespace hydra

#endif /* ITERATORCONCEPTS_H_ */
