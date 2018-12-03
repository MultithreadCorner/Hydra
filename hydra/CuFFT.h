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
 * CuFFT.h
 *
 *  Created on: Nov 30, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CUFFT_H_
#define CUFFT_H_

#include<hydra/detail/cufft/WrappersCuFFT.h>
#include<hydra/detail/cufft/BaseCuFFT.h>
#include<hydra/detail/cufft/ComplexToRealCuFFT.h>
#include<hydra/detail/cufft/RealToComplexCuFFT.h>
#include<hydra/detail/cufft/ComplexToComplexCuFFT.h>

namespace hydra {

template<typename T>
struct CuFFT
{
	typedef ComplexToComplexFFTW<T> C2C;
	typedef    RealToComplexFFTW<T> R2C;
	typedef    ComplexToRealFFTW<T> C2R;
};

}  // namespace hydra


#endif /* CUFFT_H_ */
