/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * TBBTraits.h
 *
 *  Created on: 12/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TBBTRAITS_H_
#define TBBTRAITS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendTraits.h>
#include <hydra/detail/policies/backends/TBB.h>
#include <hydra/detail/external/thrust/system/tbb/detail/par.h>

namespace hydra {

namespace detail {

template<>
struct BackendTraits<HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::par_t>
{
	typedef hydra::tbb::sys_t hydra_backend;
};

}  // namespace detail

}//namespace hydra


#endif /* TBBTRAITS_H_ */
