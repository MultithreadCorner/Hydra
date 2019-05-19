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
 * NumericalIntegral.h
 *
 *  Created on: 30/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef NUMERICALINTEGRAL_INL_
#define NUMERICALINTEGRAL_INL_

namespace hydra {

template<typename Algorithm, size_t N>
class NumericalIntegral: public Integral< Algorithm ,  N>
{

public:

	typedef void hydra_numerical_integral_tag;


};

}  // namespace hydra



#endif /* NUMERICALINTEGRAL_H_ */
