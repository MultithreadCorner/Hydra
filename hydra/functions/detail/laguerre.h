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
 * laguerre.h
 *
 *  Created on: 08/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LAGUERRE_H_
#define LAGUERRE_H_

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

namespace hydra {

template<typename T>
__hydra_host__ __hydra_device__
inline T laguerre(unsigned n, const T x){

	switch(n) {

	case 0:

		return 1.0;

	case 1:

		return 1 - x;

	default:

		T LL = 1.0;
		T LM = 1-x;
		T LN = static_cast<T>(0.0);

		for(unsigned m=2; m<=n; m++){

			LN = ((static_cast<T>(2)*m -static_cast<T>(1) - x)/m ) * LM -
					((static_cast<T>(m)- static_cast<T>(1))/m) * LL;
			LL = LM;    LM = LN;
		}

		return LN;
	}

}

}  // namespace hydra



#endif /* LAGUERRE_H_ */
