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
 * IteratorTraits.h
 *
 *  Created on: 14/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ITERATORTRAITS_H_
#define ITERATORTRAITS_H_

#include <utility>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>

namespace hydra {

namespace detail {

template<typename Iterator>
struct IteratorTraits
{
	static const bool is_host_iterator = HYDRA_EXTERNAL_NS::thrust::detail::is_host_iterator_category<
	typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::iterator_category>::value;

	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system_t;


	static system_t& GetTag()
	{
		using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		system_t sys;

		return  select_system(sys);
	}

};


}  // namespace detail

}  // namespace hydra

#endif /* ITERATORTRAITS_H_ */
