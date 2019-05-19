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
 * Constant.h
 *
 *  Created on: 10/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef ARGUMENT_H_
#define ARGUMENT_H_


#include <type_traits>
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>

namespace hydra {

template<typename ReturnType>
struct Argument
{

public:

	//tag
	typedef void hydra_functor_tag;
	typedef ReturnType return_type;

	typedef   std::true_type is_functor;

	Argument()=default;

	__hydra_host__ __hydra_device__
	inline return_type operator()( return_type arg){ return arg;}

};

}



#endif /* ARGUMENT_H_ */
