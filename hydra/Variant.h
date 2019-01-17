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
 * Variant.h
 *
 *  Created on: 20/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef VARIANT_H_
#define VARIANT_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/external/misc/variant.hpp>

/**
 * @todo : fix compile error in nvcc adding variants to vectors
 *
 */

namespace hydra {

namespace experimental {

using namespace detail::external::mpark;

}

}  // namespace hydra



#endif /* VARIANT_H_ */
