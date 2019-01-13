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

#ifndef WRAPPERSCUFFT_H_
#define WRAPPERSCUFFT_H_



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


//CuFFT
#include <cufft.h>

namespace hydra {

	namespace detail {

		namespace cufft {

			void* malloc( size_t size )
			{
					void *ptr;
					if (cudaMalloc(&ptr, size) == cudaSuccess) return ptr;
					return 0;
			}

			void free( void* ptr ){
				 cudaFree( ptr);
			}

			void* reallocate(void* ptr, size_t new_size){
						void* new_ptr ;
						if (cudaMalloc(&new_ptr, new_size) == cudaSuccess){

							cudaFree( ptr);
						}

						return new_ptr;
					}


			void* memset( void* dest, int ch, size_t count  )
			{
					if (cudaMemset(dest, ch, count) == cudaSuccess) return dest;
					return 0;
			}

			void* memcpy( void* dest, const void* src, size_t count ){
				 if( cudaMemcpy( dest, src, count, cudaMemcpyDeviceToDevice) == cudaSuccess) return dest;
				 return 0;
			}

			struct _Deleter
			{

				template<typename T>
				inline void operator()(T* ptr){
					hydra::detail::cufft::free(ptr);
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

					cufftPlan1d(&plan, nx, Type, batch);
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
				typedef hydra::complex<double> complex_double;
				typedef hydra::complex<float>  complex_float;
				typedef double  real_double;
				typedef float   real_float;


				inline void operator()(cufftHandle& plan, real_float* input, complex_float* output,
						int direction=1){

					cufftExecR2C(plan, reinterpret_cast<cufftReal*>(input), reinterpret_cast<cufftComplex*>(output));
					cudaDeviceSynchronize();
				}

				inline void operator()(cufftHandle& plan, real_double* input, complex_double* output,
						int direction=1){

					cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(input), reinterpret_cast<cufftDoubleComplex*>(output));
					cudaDeviceSynchronize();
				}

				//------

				inline void operator()(cufftHandle& plan, complex_float* input, real_float* output,
						int direction=1){

					cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(input), reinterpret_cast<cufftReal*>(output));
					cudaDeviceSynchronize();
				}


				inline void operator()(cufftHandle& plan, complex_double* input, real_double* output,
						int direction=1){

					cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(input), reinterpret_cast<cufftDoubleReal*>(output));
					cudaDeviceSynchronize();
				}

				//------

				inline void operator()(cufftHandle& plan, complex_float* input, complex_float* output,
						int direction=1){

					cufftExecC2C(plan,  reinterpret_cast<cufftComplex*>(input),reinterpret_cast<cufftComplex*>(output), direction);
					cudaDeviceSynchronize();
				}


				inline void operator()(cufftHandle& plan, complex_double* input, complex_double* output,
						int direction=1){

					cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(input), reinterpret_cast<cufftDoubleComplex*>(output)
							, direction);
					cudaDeviceSynchronize();
				}
			};


		}  // namespace cufft

	}  // namespace detail

}  // namespace hydra


#endif /* WRAPPERSCUFFT_H_ */
