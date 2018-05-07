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
 * ForEach.h
 *
 *  Created on: 04/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FOREACH_H_
#define FOREACH_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/for_each.h>

namespace hydra {

template<typename InputIterator, typename Functor>
InputIterator for_each(InputIterator first, InputIterator last, Functor const functor)
{
	return HYDRA_EXTERNAL_NS::thrust::for_each(first, last, functor);
}

template<detail::Backend Backend, typename InputIterator, typename Functor>
InputIterator for_each(hydra::detail::BackendPolicy<Backend> const& policy, InputIterator first,
		InputIterator last, Functor const functor)
{
	return HYDRA_EXTERNAL_NS::thrust::for_each( policy, first, last, functor);
}

}  // namespace hydra





#endif /* FOREACH_H_ */
