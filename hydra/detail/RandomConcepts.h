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
 * RandomConcepts.h
 *
 *  Created on: 14/03/2025
 *      Author: Davide Brundu
 */

#ifndef RANDOMCONCEPTS_H_
#define RANDOMCONCEPTS_H_

#include <concepts>
#include <type_traits>

#include <hydra/detail/FormulaTraits.h>
#include <hydra/detail/RngFormula.h>
#include <hydra/detail/RandomIteratorTraits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra {

namespace detail {

/**
 * @brief Satisfied when @c T has an RngFormula.
 */
template <typename T>
concept HasRngFormula = has_rng_formula<T>::value;

/**
 * @brief Satisfied when @c FUNCTOR has an RngFormula whose result,
 * obitaned via @c Engine, is convertible to @c Iterator's value type.
 */
template <typename FUNCTOR, typename Engine, typename Iterator>
concept IsRngFormulaConvertible = requires (Engine& engine, const FUNCTOR& functor) {
    { RngFormula<FUNCTOR>().Generate(engine, functor) } -> std::convertible_to<typename hydra::thrust::iterator_traits<Iterator>::value_type>;
};

/**
 * @brief Satisfied when @c FUNCTOR has an RngFormula whose result,
 * obitaned via @c Engine, is NOT convertible to @c Iterator's value type.
 */
template <typename FUNCTOR, typename Engine, typename Iterator>
concept NotConvertibleToIteratorValue =
    !std::convertible_to<
        decltype(RngFormula<FUNCTOR>().Generate(std::declval<Engine&>(), std::declval<const FUNCTOR&>())),
        typename std::iterator_traits<Iterator>::value_type>;

/**
 * @brief Satisfied when @c FUNCTOR has an RngFormula whose result is convertible
 * to @c Iterator's value type, i.e. the functor can be sampled straight into the
 * iterator.
 */
template <typename FUNCTOR, typename Engine, typename Iterator>
concept RngFormulaFor =
    HasRngFormula<FUNCTOR> && IsRngFormulaConvertible<FUNCTOR, Engine, Iterator>;

/**
 * @brief Satisfied when @c FUNCTOR has an RngFormula but its result is not
 * convertible to @c Iterator's value type.
 */
template <typename FUNCTOR, typename Engine, typename Iterator>
concept RngFormulaResultMismatch =
    HasRngFormula<FUNCTOR> && NotConvertibleToIteratorValue<FUNCTOR, Engine, Iterator>;

namespace random {

/**
 * @brief Satisfied when @c T is accepted as an iterator argument by the random
 * generation/sampling overloads (see hydra::detail::random::is_iterator).
 */
template <typename T>
concept Iterator = is_iterator<T>::value;

/**
 * @brief Satisfied when @c T is accepted as an iterable argument by the random
 * generation/sampling overloads (see hydra::detail::random::is_iterable).
 */
template <typename T>
concept Iterable = is_iterable<T>::value;

/**
 * @brief Satisfied when @c T is a callable (functor/lambda) accepted by the
 * random generation/sampling overloads (see hydra::detail::random::is_callable).
 */
template <typename T>
concept Callable = is_callable<T>::value;

/**
 * @brief Satisfied when @c Functor has an RngFormula whose result is convertible
 * to @c Iterable's value type (see hydra::detail::random::is_matching_iterable).
 */
template <typename Engine, typename Functor, typename Iterable>
concept MatchingIterable = is_matching_iterable<Engine, Functor, Iterable>::value;

/**
 * @brief Satisfied when @c F is a random-generation callable and @c It an output
 * iterator, i.e. the pair accepted by the sample-into-iterator overloads.
 */
template <typename F, typename It>
concept SampleInto = Callable<F> && Iterator<It>;

/**
 * @brief Satisfied when @c F is a random-generation callable and @c It an output
 * iterable, i.e. the pair accepted by the sample-into-range overloads.
 */
template <typename F, typename It>
concept SampleIntoRange = Callable<F> && Iterable<It>;

}  // namespace random

}  // namespace detail

}  // namespace hydra



#endif /* RANDOMCONCEPTS_H_ */
