/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * Transform.inl
 *
 *  Created on: 11/06/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TRANSFORM_INL_
#define TRANSFORM_INL_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <utility>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/Range.h>


namespace hydra {

template<typename Iterable_Input, typename Iterable_Output, typename Functor,
typename Iterator=decltype(std::declval<Iterable_Output>().begin())>
typename std::enable_if<hydra::detail::is_iterable<Iterable_Output>::value,
Range<decltype(std::declval<Iterable_Output&>().begin())>>::type
transform(Iterable_Input& iterable_input, Iterable_Output& iterable_output,  Functor const& unary_functor){

	HYDRA_EXTERNAL_NS::thrust::transform(iterable_input.begin(), iterable_input.end(),
			iterable_output.begin(), unary_functor);

	return make_range(iterable_output.begin(), iterable_output.end());
}




}  // namespace hydra

#endif /* TRANSFORM_INL_ */
