/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * CollectRange.h
 *
 *  Created on: 19/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COLLECTRANGE_H_
#define COLLECTRANGE_H_

#include <hydra/detail/IteratorConcepts.h>

namespace hydra {


template<typename Iterable_Index, typename Iterable_Values>
requires (detail::Iterables<Iterable_Index, Iterable_Values>)
auto collect( Iterable_Index& indexing_scheme, Iterable_Values& collected_values)
-> Range<hydra::thrust::permutation_iterator<
		decltype(std::declval<Iterable_Values&>().begin()),
		decltype(std::declval<Iterable_Index&>().begin())>
>
{
	typedef hydra::thrust::permutation_iterator<Iterable_Values,Iterable_Index> collect_iterator;
	return make_range(collect_iterator(begin(collected_values), begin(indexing_scheme) ),
			collect_iterator(begin(collected_values), end(indexing_scheme)) );


}

}  // namespace hydra



#endif /* COLLECTRANGE_H_ */
