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
 * ConstantRange.inl
 *
 *  Created on: 20/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONSTANTRANGE_INL_
#define CONSTANTRANGE_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/iterator/constant_iterator.h>

namespace hydra {

template<typename Value_Type>
Range<HYDRA_EXTERNAL_NS::thrust::constant_iterator<Value_Type>>
constant_range(const Value_Type&  value){

	return make_range( HYDRA_EXTERNAL_NS::thrust::constant_iterator<Value_Type>(value),
			HYDRA_EXTERNAL_NS::thrust::constant_iterator<Value_Type>(value) );
}

}  // namespace hydra




#endif /* CONSTANTRANGE_INL_ */
