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
 * Placeholders.h
 *
 *  Created on: 17/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PLACEHOLDERS_H_
#define PLACEHOLDERS_H_

#include <hydra/detail/external/thrust/detail/type_traits.h>


namespace hydra {

namespace placeholders {

template<unsigned int I>
struct placeholder{
	constexpr  operator unsigned int() const { return I; }  ;
};

/*
is_placeholder<I> members

unsigned int is_placeholder<I>::value ;
typedef is_placeholder<I>::value_type ;
typedef integral_constant<unsigned int, I> type;
*/
template<typename T>
struct is_placeholder:
		public HYDRA_EXTERNAL_NS::thrust::detail::integral_constant<int, -1>{};


template<unsigned int I>
struct is_placeholder< placeholder<I> >:
		public HYDRA_EXTERNAL_NS::thrust::detail::integral_constant<unsigned int, I>{};


}  // namespace placeholders

}  // namespace hydra

#include<hydra/detail/Placeholders.inl>

#endif /* PLACEHOLDERS_H_ */
