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

namespace hydra {

namespace detail {

template <typename T>
concept HasRngFormula = has_rng_formula<T>::value;

template <typename FUNCTOR, typename Engine, typename Iterator>
concept IsRngFormulaConvertible = requires (Engine& engine, const FUNCTOR& functor) {
    { RngFormula<FUNCTOR>().Generate(engine, functor) } -> std::convertible_to<typename hydra::thrust::iterator_traits<Iterator>::value_type>;
};

template <typename FUNCTOR, typename Engine, typename Iterator>
concept NotConvertibleToIteratorValue =
    !std::convertible_to<
        decltype(RngFormula<FUNCTOR>().Generate(std::declval<Engine&>(), std::declval<const FUNCTOR&>())),
        typename std::iterator_traits<Iterator>::value_type>;

}  // namespace detail

}  // namespace hydra



#endif /* RANDOMCONCEPTS_H_ */
