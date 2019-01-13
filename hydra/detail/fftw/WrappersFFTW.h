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
#include <hydra/Complex.h>

#include <cassert>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>

//FFTW3
#include <fftw3.h>


namespace hydra {

	namespace detail {

		namespace fftw {

		/**
		 * this function will delete the ptr if the reallocation is successfull,
		 *  othewise will return null
		 * @param ptr
		 * @param new_size
		 */
		void* reallocate(void* ptr, size_t new_size){
			void* new_ptr =fftw_malloc(new_size );

			if(new_ptr!=NULL){
				fftwf_free(ptr);
			}
			return new_ptr;
		}

			struct _Deleter
			{

				inline void operator()(float* ptr){
					fftwf_free(ptr);
				}

				inline void operator()(double* ptr){
					fftw_free(ptr);
				}

				inline void operator()(hydra::complex<double>* ptr ){
									fftw_free(reinterpret_cast<fftw_complex*>(ptr) );
				}

				inline void operator()(hydra::complex<float>* ptr){
									fftwf_free(reinterpret_cast<fftwf_complex*>(ptr) );
				}
			};

			//===========================================================================
			// Generic planner
			struct _Planner
			{
				typedef hydra::complex<double> complex_double;
				typedef hydra::complex<float>  complex_float;
				typedef double  real_double;
				typedef float   real_float;

				typedef fftw_plan  plan_type;
				typedef fftwf_plan planf_type;

				// Complex -> Complex
				inline plan_type operator()(int n, complex_double *in, complex_double *out,
						unsigned flags, int sign=0 ) {

					return fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex*>(in),
							reinterpret_cast<fftw_complex*>(out), sign, flags);
				}

				// Real -> Complex
				inline plan_type operator()(int n, double *in, complex_double *out,
						unsigned flags, int sign=0 ){

					return fftw_plan_dft_r2c_1d(n, in, reinterpret_cast<fftw_complex*>(out), flags);
				}

				// Complex -> Real
				inline plan_type operator()(int n, complex_double *in, double *out,
						unsigned flags, int sign=0 ){

					return fftw_plan_dft_c2r_1d(n, reinterpret_cast<fftw_complex*>(in), out, flags);
				}

				// Complex -> Complex
				inline fftwf_plan operator()(int n, complex_float *in, complex_float *out,
						unsigned flags, int sign=0) {

					return fftwf_plan_dft_1d(n, reinterpret_cast<fftwf_complex*>(in),
							reinterpret_cast<fftwf_complex*>(out), sign, flags);
				}

				// Real -> Complex
				inline fftwf_plan operator()(int n, float *in, complex_float *out,
						unsigned flags, int sign=0 ){

					return fftwf_plan_dft_r2c_1d(n, in, reinterpret_cast<fftwf_complex*>(out), flags);
				}

				// Complex -> Real
				inline fftwf_plan operator()(int n, complex_float *in, float *out,
						unsigned flags, int sign=0 ){

					return fftwf_plan_dft_c2r_1d(n, reinterpret_cast<fftwf_complex*>(in), out, flags);
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
