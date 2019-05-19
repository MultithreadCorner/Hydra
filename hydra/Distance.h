/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * Distance.h
 *
 *  Created on: 21/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/distance.h>

namespace hydra {

template<typename Iterator>
inline __hydra_host__ __hydra_device__
auto distance(Iterator first, Iterator last)
->decltype( HYDRA_EXTERNAL_NS::thrust::distance<Iterator>(first,last))
{
	return HYDRA_EXTERNAL_NS::thrust::distance(first, last);
}

}  // namespace hydra

#endif /* DISTANCE_H_ */
