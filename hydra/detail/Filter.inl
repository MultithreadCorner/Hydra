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
 * Filter.inl
 *
 *  Created on: 15/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FILTER_INL_
#define FILTER_INL_


namespace hydra {

template<typename Container, typename Functor>
hydra::pair<typename Container::iterator, typename Container::iterator>
apply_filter(Container& container, Functor const& filter)
{
	typename Container::iterator new_end = thrust::partition(container.begin(),container.end() , filter);
     return hydra::make_pair(container.begin(), new_end);
}

}  // namespace hydra

#endif /* FILTER_INL_ */
