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
 * Zip.h
 *
 *  Created on: 29/06/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ZIP_H_
#define ZIP_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/Tuple.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/Range.h>


#include <utility>

namespace hydra {


template<typename ...Iterables>
typename std::enable_if< detail::all_true< detail::is_iterable<Iterables>::value...>::value,
Range< HYDRA_EXTERNAL_NS::thrust::zip_iterator<
	decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(std::declval<Iterables&>().begin()...))>>>::type
zip(Iterables&&... iterables){

	return make_range( HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(std::forward<Iterables>(iterables).begin()...)),
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(std::forward<Iterables>(iterables).end()...)) );
}



}  // namespace hydra


#endif /* ZIP_H_ */
