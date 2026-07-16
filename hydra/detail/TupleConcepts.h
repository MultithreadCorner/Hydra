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
 * TupleConcepts.h
 *
 *  Created on: 28/06/2026
 *      Author: Davide Brundu
 */

#ifndef TUPLECONCEPTS_H_
#define TUPLECONCEPTS_H_

#include <concepts>
#include <type_traits>

#include <hydra/detail/ArgumentTraits.h>

namespace hydra {

namespace detail {

/**
 * @brief Satisfied when @c T (after decay) is a Hydra/Thrust tuple type
 * (see hydra::detail::is_tuple_type).
 */
template <typename T>
concept TupleType = is_tuple_type<std::decay_t<T>>::value;

/**
 * @brief Satisfied when @c T (after decay) is a tuple whose elements are all
 * Hydra function arguments (see hydra::detail::is_tuple_of_function_arguments).
 */
template <typename T>
concept TupleOfFunctionArguments = is_tuple_of_function_arguments<std::decay_t<T>>::value;

/**
 * @brief Satisfied when the pack @c T... is a valid set of call arguments for a
 * functor whose signature tuple is @c Signature
 * (see hydra::detail::is_valid_type_pack).
 */
template <typename Signature, typename... T>
concept ValidTypePack = is_valid_type_pack<Signature, T...>::value;

}  // namespace detail

}  // namespace hydra

#endif /* TUPLECONCEPTS_H_ */
