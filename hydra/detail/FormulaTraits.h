/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * FormulaTraits.h
 *
 *  Created on: 22/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FORMULATRAITS_H_
#define FORMULATRAITS_H_

#include<hydra/detail/RngFormula.h>
#include<hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include<type_traits>
#include<utility>

namespace hydra {

namespace detail {

template<typename Formula>
struct is_rng_formula: std::false_type{};

template<typename Formula>
struct is_rng_formula<hydra::RngFormula<Formula>>: std::true_type{};

template<typename Functor, typename T= hydra_thrust::void_t<> >
struct has_rng_formula: std::false_type{};

template<typename Functor>
struct has_rng_formula<Functor,
          hydra_thrust::void_t< typename hydra::RngFormula<Functor>::value_type > >: std::true_type{};



}  // namespace detail

}  // namespace hydra


#endif /* FORMULATRAITS_H_ */
