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
#include <utility>

namespace hydra {

template<typename Iterable, typename Functor>
	typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
	Range<decltype(std::declval<Iterable&>().begin())>>::type
for_each(Iterable&& iterable, Functor const& functor)
{
	HYDRA_EXTERNAL_NS::thrust::for_each( std::forward<Iterable>(iterable).begin(),
			std::forward<Iterable>(iterable).end(), functor);

	return make_range( std::forward<Iterable>(iterable).begin(),
			           std::forward<Iterable>(iterable).end());
}


}  // namespace hydra





#endif /* FOREACH_H_ */
