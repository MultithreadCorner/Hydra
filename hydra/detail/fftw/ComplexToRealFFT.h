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
 * ComplexToRealFFT.h
 *
 *  Created on: 16/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COMPLEXTOREALFFT_H_
#define COMPLEXTOREALFFT_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/Range.h>
#include <hydra/Tuple.h>

#include <cassert>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>
//#include <complex.h>

//FFTW3
#include <fftw3.h>

//Hydra wrappers
#include<hydra/detail/fftw/Wrappers.h>
#include<hydra/detail/fftw/BaseFFT.h>

namespace hydra {


template<typename T,
typename InputType   = typename std::conditional< std::is_same<double,T>::value, fftw_complex, fftwf_complex>::type,
typename OutputType  = T,
typename PlanType    = detail::fftw::_Planner<T> >
class ComplexToRealFFT: public BaseFFT<InputType, OutputType, PlanType >
{

public:

	ComplexToRealFFT()=delete;

	ComplexToRealFFT(int logical_size, unsigned flags=FFTW_ESTIMATE):
		BaseFFT<InputType, OutputType, PlanType >(logical_size/2 +1, logical_size ,flags )
	{}

	ComplexToRealFFT(ComplexToRealFFT<T,InputType, OutputType, PlanType>&& other):
		BaseFFT<InputType, OutputType, PlanType >(
				std::forward<BaseFFT<InputType, OutputType, PlanType >&&>(other))
	{}

	ComplexToRealFFT<T,InputType, OutputType, PlanType>&
	operator=(ComplexToRealFFT<T,InputType, OutputType, PlanType>&& other)
	{
		if(this ==&other) return *this;

		BaseFFT<InputType, OutputType, PlanType >::operator=(other);

		return *this;
	}

	~ComplexToRealFFT(){ }

};

}  // namespace hydra

#endif /* COMPLEXTOREALFFT_H_ */