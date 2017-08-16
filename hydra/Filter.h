/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * Filter.h
 *
 *  Created on: 15/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FILTER_H_
#define FILTER_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Tuple.h>
//thrust
#include <thrust/partition.h>

//std


namespace hydra {

/**
 * Apply a filter to the range [first, last] and return a pair of iterators for the filtered events.
 * This function will not change the size of the original range,  [first, last], but will reorder the
 * entries to put together the accepted entries.
 * @param first Iterator pointing to the begin of the range to filter.
 * @param last  Iterator pointing to the end of the range to filter.
 * @param filter Functor returning bool.
 * @return
 */
template<typename Container, typename Functor>
hydra::pair<typename Container::iterator, typename Container::iterator>
apply_filter(Container& container, Functor const& filter);

}  // namespace hydra

#include <hydra/detail/Filter.inl>

#endif /* FILTER_H_ */
