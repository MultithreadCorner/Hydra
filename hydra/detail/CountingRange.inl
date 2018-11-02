/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * CountingRange.inl
 *
 *  Created on: 20/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COUNTINGRANGE_INL_
#define COUNTINGRANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>

namespace hydra {

Range<HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>>
range(long int first, long int last ){

	return make_range( HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(first),
			HYDRA_EXTERNAL_NS::thrust::counting_iterator<long int>(last) );
}

}  // namespace hydra

#endif /* COUNTINGRANGE_INL_ */
