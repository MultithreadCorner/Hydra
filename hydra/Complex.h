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
 * Complex.h
 *
 *  Created on: 18/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COMPLEX_H_
#define COMPLEX_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/complex.h>
#include <type_traits>

namespace hydra {

template<typename T,
			typename = typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_arithmetic<T>::value, void>::type>
using Complex =  HYDRA_EXTERNAL_NS::thrust::complex<T>;

}  // namespace hydra

#endif /* COMPLEX_H_ */
