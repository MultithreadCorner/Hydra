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
 * Wrappers.h
 *
 *  Created on: 16/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WRAPPERSFFTW_H_
#define WRAPPERSFFTW_H_



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


namespace hydra {

	namespace detail {

		namespace fftw {

			struct _Deleter
			{

				inline void operator()(float* ptr){
					fftwf_free(ptr);
				}

				inline void operator()(double* ptr){
					fftw_free(ptr);
				}

				inline void operator()(fftw_complex* ptr){
									fftw_free(ptr);
				}

				inline void operator()(fftwf_complex* ptr){
									fftwf_free(ptr);
				}
			};

			//===========================================================================
			// Generic planner
			template<typename T>
			struct _Planner;

			template<> struct _Planner<double>
			{
				typedef fftw_plan plan_type;

				// Complex -> Complex
				inline plan_type operator()(int n, fftw_complex *in, fftw_complex *out,
						unsigned flags, int sign=0 ) {

					return fftw_plan_dft_1d(n, in, out, sign, flags);
				}

				// Real -> Complex
				inline plan_type operator()(int n, double *in, fftw_complex *out,
						unsigned flags, int sign=0 ){

					return fftw_plan_dft_r2c_1d(n, in, out, flags);
				}

				// Complex -> Real
				inline plan_type operator()(int n, fftw_complex *in, double *out,
						unsigned flags, int sign=0 ){

					return fftw_plan_dft_c2r_1d(n, in, out, flags);
				}

			};


			template<> struct _Planner<float>
			{
				typedef fftwf_plan plan_type;

				// Complex -> Complex
				inline fftwf_plan operator()(int n, fftwf_complex *in, fftwf_complex *out,
						unsigned flags, int sign=0) {

					return fftwf_plan_dft_1d(n, in, out, sign, flags);
				}

				// Real -> Complex
				inline fftwf_plan operator()(int n, float *in, fftwf_complex *out,
						unsigned flags, int sign=0 ){

					return fftwf_plan_dft_r2c_1d(n, in, out, flags);
				}

				// Complex -> Real
				inline fftwf_plan operator()(int n, fftwf_complex *in, float *out,
						unsigned flags, int sign=0 ){

					return fftwf_plan_dft_c2r_1d(n, in, out, flags);
				}

			};

			//===========================================================================

			struct _PlanDestroyer
			{
				inline void operator()(fftw_plan& plan ){

					fftw_destroy_plan(plan);
				}

				inline void operator()(fftwf_plan& plan ){

					fftwf_destroy_plan(plan);
				}

			};

			//===========================================================================

			struct _PlanExecutor
			{
				inline void operator()(fftw_plan& plan ){

					fftw_execute(plan);
				}

				inline void operator()(fftwf_plan& plan ){

					fftwf_execute(plan);
				}
			};


		}  // namespace fftw

	}  // namespace detail


}  // namespace hydra


#endif /* WRAPPERSFFTW_H_ */
