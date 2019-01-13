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
 * FFTW.h
 *
 *  Created on: 13/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HYDRA_FFTW_H_
#define HYDRA_FFTW_H_

/**
 *
 */
#include <hydra/detail/FFTPolicy.h>
#include<hydra/detail/fftw/WrappersFFTW.h>
#include<hydra/detail/fftw/BaseFFTW.h>
#include<hydra/detail/fftw/ComplexToRealFFTW.h>
#include<hydra/detail/fftw/RealToComplexFFTW.h>
#include<hydra/detail/fftw/ComplexToComplexFFTW.h>
#include<hydra/host/System.h>
#include<hydra/device/System.h>

namespace hydra {

	namespace detail {

		template<typename T>
		struct FFTPolicy<T, detail::FFTW>
		{
			typedef ComplexToComplexFFTW<T> C2C;
			typedef    RealToComplexFFTW<T> R2C;
			typedef    ComplexToRealFFTW<T> C2R;
			typedef    hydra::host::sys_t host_backend_type;
#if HYDRA_DEVICE_SYSTEM!=CUDA
			typedef       hydra::device::sys_t device_backend_type;
#else
			typedef       hydra::host::sys_t device_backend_type;
#endif
		};

	}  // namespace detail


	namespace fft {

		typedef detail::FFTPolicy<double, detail::FFTW> fftw_f64_t;
		typedef detail::FFTPolicy< float, detail::FFTW> fftw_f32_t;

		static const fftw_f32_t fftw_f32=fftw_f32_t();

		static const fftw_f64_t fftw_f64=fftw_f64_t();


	}  // namespace fft

}  // namespace hydra

#endif /* FFTW_H_ */
