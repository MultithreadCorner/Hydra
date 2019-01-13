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
 * FFTBackend.h
 *
 *  Created on: Dec 3, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FFTBACKEND_H_
#define FFTBACKEND_H_


namespace hydra {

	namespace detail {

		enum FFTCalculator{CuFFT, FFTW};

		 template<typename Precision, FFTCalculator FFTBackend>
		 struct FFTPolicy;


	}  // namespace detail

}  // namespace hydra


#endif /* FFTBACKEND_H_ */
