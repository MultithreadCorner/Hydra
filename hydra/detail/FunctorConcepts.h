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
 * FunctorConcepts.h
 *
 *  Created on: 26/06/2026
 *      Author: Davide Brundu
 */

#ifndef FUNCTORCONCEPTS_H_
#define FUNCTORCONCEPTS_H_

#include <concepts>
#include <type_traits>

#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>


namespace hydra {

namespace detail {

/**
 * @brief Satisfied when @c T is a Hydra functor (see hydra::detail::is_hydra_functor).
 */
template <typename T>
concept HydraFunctor = is_hydra_functor<T>::value;

/**
 * @brief Satisfied when @c T is a Hydra composite functor (see hydra::detail::is_hydra_composite_functor).
 */
template <typename T>
concept HydraCompositeFunctor = is_hydra_composite_functor<T>::value;

/**
 * @brief Satisfied when @c T is a Hydra lambda (see hydra::detail::is_hydra_lambda).
 */
template <typename T>
concept HydraLambda = is_hydra_lambda<T>::value;

/**
 * @brief Satisfied when @c T is a Hydra functor or a Hydra lambda, i.e. a
 * callable that can take part in functor arithmetic.
 */
template <typename T>
concept HydraCallable = HydraFunctor<T> || HydraLambda<T>;

/**
 * @brief Satisfied when @c T is a Hydra callable and @c U is an arithmetic type,
 * i.e. the callable can be scaled/combined with the scalar @c U. Folds the
 * recurring @c HydraCallable<T> && std::is_arithmetic_v<U> constraint used by the
 * functor-arithmetic operator overloads into a single name.
 */
template <typename T, typename U>
concept CallableScaledBy = HydraCallable<T> && std::is_arithmetic_v<U>;

}  // namespace detail

}  // namespace hydra

#endif /* FUNCTORCONCEPTS_H_ */
