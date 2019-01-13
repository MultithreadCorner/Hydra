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
 * Spiline.h
 *
 *  Created on: 16/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINE_H_
#define SPILINE_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/execution_policy.h>
#include <hydra/detail/external/thrust/binary_search.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/Iterable_traits.h>

#include <math.h>
#include <algorithm>
#include <type_traits>

namespace hydra {

template<typename Iterator1, typename Iterator2,
    typename Type=typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type>
__hydra_host__ __hydra_device__
inline typename  std::enable_if<
                    std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator1>::value_type >::value &&
                    std::is_floating_point<typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type >::value,
                    Type >::type
spiline(Iterator1 first, Iterator1 last,  Iterator2 measurements, Type value);


template<typename Iterable1, typename Iterable2,
    typename Type= typename Iterable1::value_type>
__hydra_host__ __hydra_device__
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value &&
                       hydra::detail::is_iterable<Iterable2>::value &&
                       std::is_floating_point<typename Iterable1::value_type >::value &&
                       std::is_floating_point<typename Iterable2::value_type >::value,
                       Type >::type
spiline(Iterable1&& abiscissae,  Iterable2&& measurements, Type value);


} // namespace hydra

#include <hydra/detail/Spiline.inl>

#endif /* SPILINE_H_ */
