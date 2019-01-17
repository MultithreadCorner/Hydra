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
 * Caster.h
 *
 *  Created on: 26/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CASTER_H_
#define CASTER_H_

#include <hydra/detail/Config.h>
#include <functional>


namespace hydra {

namespace detail {

template<typename TypeIn, typename TypeOut >
struct Caster:public HYDRA_EXTERNAL_NS::thrust::unary_function<TypeIn,TypeOut>{

	__hydra_host__ __hydra_device__
	inline TypeOut operator()( TypeIn & value){

		return (TypeOut) value;
	}
};

}  // namespace detail


}  // namespace name



#endif /* CASTER_H_ */
