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
 * RandomIteratorTraits.h
 *
 *  Created on: 29/06/2026
 *      Author: Davide Brundu
 *
 *  Type traits used to classify the arguments accepted by the random
 *  generation/sampling overloads (an iterator, an iterable, a callable, or a
 *  functor whose RngFormula matches an iterable's value type). Kept separate
 *  from the engine-level hydra/detail/RandomTraits.h.
 */

#ifndef RANDOMITERATORTRAITS_H_
#define RANDOMITERATORTRAITS_H_

#include <type_traits>

#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>
#include <hydra/detail/FormulaTraits.h>
#include <hydra/detail/RngFormula.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra {

namespace detail {

namespace random {

template<typename T>
struct is_iterator: std::conditional<
        !hydra::detail::is_hydra_composite_functor<T>::value &&
		!hydra::detail::is_hydra_functor<T>::value &&
		!hydra::detail::is_hydra_lambda<T>::value &&
		!hydra::detail::is_iterable<T>::value &&
		 hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template<typename T>
struct is_iterable: std::conditional<
        !hydra::detail::is_hydra_composite_functor<T>::value &&
		!hydra::detail::is_hydra_functor<T>::value &&
		!hydra::detail::is_hydra_lambda<T>::value  &&
		 hydra::detail::is_iterable<T>::value &&
		!hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template<typename T>
struct is_callable: std::conditional<
        (hydra::detail::is_hydra_composite_functor<T>::value ||
         hydra::detail::is_hydra_functor<T>::value ||
         hydra::detail::is_hydra_lambda<T>::value ) &&
        !hydra::detail::is_iterable<T>::value &&
        !hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template< typename Engine, typename Functor, typename Iterable,
          bool Enable = hydra::detail::has_rng_formula<Functor>::value &&
                        hydra::detail::is_iterable<Iterable>::value &&
                       !hydra::detail::is_iterator<Iterable>::value >
struct rng_formula_matches_iterable: std::false_type {};

template< typename Engine, typename Functor, typename Iterable>
struct rng_formula_matches_iterable<Engine, Functor, Iterable, true>:
    std::is_convertible<
        decltype(std::declval<RngFormula<Functor>>().Generate( std::declval<Engine&>(), std::declval<Functor const&>())),
        typename hydra::thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type>{};

template< typename Engine, typename Functor, typename Iterable>
struct is_matching_iterable: std::conditional<
    (hydra::detail::is_hydra_composite_functor<Functor>::value ||
     hydra::detail::is_hydra_functor<Functor>::value ||
     hydra::detail::is_hydra_lambda<Functor>::value  ) &&
     rng_formula_matches_iterable<Engine, Functor, Iterable>::value,
    std::true_type,  std::false_type
>::type{};

}  // namespace random

}  // namespace detail

}  // namespace hydra

#endif /* RANDOMITERATORTRAITS_H_ */
