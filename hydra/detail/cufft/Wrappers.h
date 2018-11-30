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

#ifndef WRAPPERS_H_
#define WRAPPERS_H_



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
#include <cufft.h>


namespace hydra {

	namespace detail {

		namespace cufft {

			struct _Deleter
			{

				template<T>
				inline void operator()(T* ptr){
					hydra::cuda::free(ptr);
				}

			};

			//===========================================================================
			// Generic planner
			template<cufftType Type>
			struct _Planner
			{
				typedef cufftHandle plan_type;


				inline plan_type operator()(int nx, int batch) {

					plan_type plan;

					cufftPlan1d(&plan, n, Type{}, batch);
					return plan;
				}


			};

			struct _PlanDestroyer
			{
				inline void operator()( cufftHandle& plan ){

					cufftDestroy(plan);

				}

			};

			//===========================================================================

			struct _PlanExecutor
			{
				inline void operator()(cufftHandle& plan, cufftReal* input, cufftComplex* output){

					cufftExecR2C(plan, input, output);
					cudaDeviceSynchronize();
				}

				inline void operator()(cufftHandle& plan, cufftDoubleReal* input, cufftDoubleComplex* output){

					cufftExecD2Z(plan, input, output);
					cudaDeviceSynchronize();
				}

				//------

				inline void operator()(cufftHandle& plan, cufftComplex* input, cufftReal* output){

					cufftExecC2R(plan, input, output);
					cudaDeviceSynchronize();
				}


				inline void operator()(cufftHandle& plan, cufftDoubleComplex* input, cufftDoubleReal* output){

					cufftExecZ2D(plan, input, output);
					cudaDeviceSynchronize();
				}

				//------

				inline void operator()(cufftHandle& plan, cufftComplex* input, cufftComplex* output,
						int direction ){

					cufftExecC2C(plan, input, output, direction);
					cudaDeviceSynchronize();
				}


				inline void operator()(cufftHandle& plan, cufftDoubleComplex* input, cufftDoubleComplex* output,
						int direction ){

					cufftExecZ2Z(plan, input, output, direction);
					cudaDeviceSynchronize();
				}
			};


		}  // namespace cufft

	}  // namespace detail


}  // namespace hydra


#endif /* WRAPPERS_H_ */
