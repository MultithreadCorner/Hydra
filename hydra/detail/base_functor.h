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
 * base_functor.h
 *
 *  Created on: 07/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */

#ifndef BASE_FUNCTOR_H_
#define BASE_FUNCTOR_H_

#include <type_traits>
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/TypeTraits.h>

#include <hydra/detail/external/thrust/tuple.h>

namespace hydra {

namespace detail {

template<typename F1, typename F2, typename ...Fs>
struct sum_result {
	typedef common_type_t<F1, F2, Fs...> type;
};

template<typename F1, typename F2, typename ...Fs>
struct multiply_result {
	typedef common_type_t<F1, F2, Fs...> type;
};

template<typename F1, typename F2, typename ...Fs>
struct divide_result {
	typedef common_type_t<F1, F2, Fs...> type;
};

template<typename F1, typename F2, typename ...Fs>
struct minus_result {
	typedef common_type_t<F1, F2, Fs...> type;
};

}

}
#endif /* BASE_FUNCTOR_H_ */
