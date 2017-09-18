/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * Copy.h
 *
 *  Created on: 25/09/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef COPY_H_
#define COPY_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/copy.h>

namespace hydra {

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
	return HYDRA_EXTERNAL_NS::thrust::copy(first, last, result);
}

template<detail::Backend Backend, typename InputIterator, typename OutputIterator>
OutputIterator copy(hydra::detail::BackendPolicy<Backend> const& policy, InputIterator first,
		InputIterator last, OutputIterator result)
{
	return HYDRA_EXTERNAL_NS::thrust::copy( policy, first, last, result);
}

}  // namespace hydra


#endif /* COPY_H_ */
