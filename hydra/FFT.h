/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * FFT.h
 *
 *  Created on: Nov 12, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FFT_H_
#define FFT_H_

namespace hydra {

template<typename FFT_Backend>
class FFT
{
	typedef FFT_Backend::complex_type complex_t;
	typedef FFT_Backend::real_type       real_t;
	typedef FFT_Backend::plan_type       plan_t;


public:

	FFT()=default;


};



}  // namespace hydra



#endif /* FFT_H_ */
