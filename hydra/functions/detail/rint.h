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
 * rint.h
 *
 *  Created on: 23/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RINT_H_
#define RINT_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>
#include <cfenv>

namespace hydra {

template<typename T>
__hydra_host__ __hydra_device__
inline double rint(T x){

#ifdef 	__CUDA_ARCH__

	return ::rint(x);

#else

    std::fesetround(FE_TONEAREST);
    return std::rint(x);

#endif

}

}  // namespace hydra



#endif /* RINT_H_ */
