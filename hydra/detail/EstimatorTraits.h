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
 * EstimatorTraits.h
 *
 *  Created on: 08/03/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ESTIMATORTRAITS_H_
#define ESTIMATORTRAITS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include <type_traits>

namespace hydra {

namespace detail {

template<class Estimator, typename T= hydra_thrust::void_t<>  >
struct is_hydra_estimator: std::false_type {};

template<class T>
struct is_hydra_estimator<T,
        hydra_thrust::void_t<typename T::likelihood_estimator_type> >: std::true_type {};

}  // namespace detail

}  // namespace hydra





#endif /* ESTIMATORTRAITS_H_ */
