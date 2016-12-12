/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * ProcessPoints.h
 *
 *  Created on: 20/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */


#ifndef PROCESSCALLSVEGAS_H_
#define PROCESSCALLSVEGAS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include <curand_kernel.h>
#endif

#include <mutex>


namespace hydra{

namespace detail {

struct ResultVegas
{

	GReal_t integral;
	GReal_t tss;

	__host__ __device__
	ResultVegas():
	integral(0),
	tss(0)
	{}
};

struct ProcessBoxesVegas
		:public thrust::binary_function< ResultVegas const&, ResultVegas const& , ResultVegas >
{

	__host__ __device__ inline
	ResultVegas operator()(ResultVegas const& x, ResultVegas const& y)
	{
		ResultVegas result;

		result.integral = x.integral + y.integral ;
		result.tss      = x.tss + y.tss ;

		return result;

	}
};

template<typename FUNCTOR, size_t NDimensions, typename Precision, typename GRND=thrust::random::default_random_engine>
struct ProcessCallsVegas
{
	ProcessCallsVegas( size_t NBins,
			size_t NBoxes,
			size_t NBoxesPerDimension,
			size_t NCallsPerBox,
			GReal_t Jacobian,
			GInt_t Seed, /*GUInt_t* Bins,*/
			GReal_t* Xi,
			GReal_t* XLow,
			GReal_t* DeltaX,
			Precision* Distribution/*FunctionCalls*/,
			GInt_t Mode,
			std::mutex *Mutex,
			FUNCTOR const& functor):
		fMode(Mode),
		fSeed(Seed),
		fNBins(NBins),
		fNBoxes(NBoxes),
		fNBoxesPerDimension(NBoxesPerDimension),
		fNCallsPerBox(NCallsPerBox),
		fJacobian(Jacobian),
		fXi(Xi),
		fXLow(XLow),
		fDeltaX(DeltaX),
		fDistribution(Distribution),
		fFunctor(functor),
		fMutex(Mutex)
	{}

	__host__ __device__
	ProcessCallsVegas( ProcessCallsVegas<FUNCTOR,NDimensions,Precision, GRND> const& other):
	fMode(other.fMode),
	fSeed(other.fSeed),
	fNBins(other.fNBins),
	fNBoxes(other.fNBoxes),
	fNBoxesPerDimension(other.fNBoxesPerDimension),
	fNCallsPerBox(other.fNCallsPerBox),
	fJacobian(other.fJacobian),
	fXi(other.fXi),
	fXLow(other.fXLow),
	fDeltaX(other.fDeltaX),
	fDistribution(other.fDistribution),
	fFunctor(other.fFunctor),
	fMutex(other.fMutex)
	{

	}

	__host__ __device__
	~ProcessCallsVegas(){

	};

	__host__ __device__
	inline GInt_t GetBoxCoordinate(GInt_t idx, GInt_t dim, GInt_t nboxes, GInt_t j)
	{
		GInt_t _idx = idx;
		GInt_t _dim = dim - 1;
		GInt_t _coordinate;

		do {
			_coordinate = _idx % (nboxes);
			_idx /= (nboxes);
			_dim--;
		} while (_dim >= j);

		return _coordinate;
	}

	__host__   __device__ inline
	size_t hash(size_t a, size_t b)
	{
		//Matthew Szudzik pairing
		//http://szudzik.com/ElegantPairing.pdf

		size_t  A = 2 * a ;
		size_t  B = 2 * b ;
		size_t  C = ((A >= B ? A * A + A + B : A + B * B) / 2);
		return  C ;
	}
	__host__ __device__
	inline ResultVegas operator()( size_t box)
	{

		GReal_t volume = 1.0;
		GReal_t x[NDimensions];
		GInt_t bin[NDimensions];
		ResultVegas result;

#ifdef __CUDA_ARCH__

		curandStateSobol64_t state;
		curandDirectionVectors64_t dvector;
		curand_init(dvector, fSeed+box, &state);


#else

		GRND randEng( hash(fSeed,box));
		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);
#endif


		GReal_t m    = 0.0, q = 0;
		GReal_t mean = 0.0;
		GReal_t f_sq_sum = 0.0;
		GReal_t m2=0.0;
		for (size_t call = 0; call < fNCallsPerBox; call++)
		{
//size_t call = box%fNCallsPerBox;


			for (size_t j = 0; j < NDimensions; j++) {


#ifdef __CUDA_ARCH__
				x[j] = 	curand_uniform_double(&state);
#else

				randEng.discard(call +call*j);
				x[j] = uniDist(randEng);
#endif

				GInt_t b = GetBoxCoordinate(box, NDimensions, fNBoxesPerDimension, j);


				GReal_t z = ((b + x[j]) / fNBoxesPerDimension) * fNBins;

				GInt_t k = static_cast<GInt_t>(z);

				GReal_t y = 0;
				GReal_t bin_width = 0;

				bin[j] = k;

				bin_width = fXi[(k + 1)*NDimensions + j]
				                - (k != 0) * fXi[k*NDimensions + j];

				y = (k != 0) * fXi[k*NDimensions + j] + (z - k) * bin_width;

				x[j] = fXLow[j] + y * fDeltaX[j];

				volume *= bin_width;
				//printf("bin[j]=%d x[j]=%f box=%d  fNCallsPerBox = %f\n", bin[j],x[j],b ,double(call)	);
			}

			GReal_t fval = fJacobian*volume*fFunctor( detail::arrayToTuple<GReal_t, NDimensions>(x));


			GReal_t  delta  =  fval - mean;
			mean +=  delta / (double(call) + 1.0);
			GReal_t  delta2  =  fval - mean;
			m2 +=  delta * delta2 ;

			if (fMode != MODE_STRATIFIED)
			{
				for (GUInt_t j = 0; j < NDimensions; j++) {
#ifdef __CUDA_ARCH__

					atomicAdd( (fDistribution + bin[j]* NDimensions + j) , static_cast<Precision>(fval*fval));

#else
					std::lock_guard<std::mutex> lock(*fMutex);

					*(fDistribution  + bin[j]* NDimensions + j) += static_cast<Precision>(fval*fval);
#endif
				}
			}


		}

		/*
		result.integral += m*fNCallsPerBox;
		f_sq_sum = q*fNCallsPerBox;
		result.tss += f_sq_sum;
		*/
		result.integral = mean*fNCallsPerBox;
		result.tss = m2*fNCallsPerBox;

		if (fMode == MODE_STRATIFIED) {
			for (GUInt_t j = 0; j < NDimensions; j++) {
#ifdef __CUDA_ARCH__

				atomicAdd((fDistribution + bin[j]*NDimensions+j), static_cast<Precision>(m2));

#else
				std::lock_guard<std::mutex> lock(*fMutex);
//#pragma omp atomic
				*(fDistribution  + bin[j]* NDimensions + j) += static_cast<Precision>(m2);
#endif
			}
		}

		return result;

	}

private:

	size_t  fNBins;
	size_t  fNBoxes;
	size_t  fNBoxesPerDimension;
	size_t  fNCallsPerBox;
    GReal_t fJacobian;
    GInt_t  fSeed;
    GInt_t  fMode;

   GReal_t*   __restrict__ fXi;
   GReal_t*   __restrict__ fXLow;
   GReal_t*   __restrict__ fDeltaX;
   Precision*  __restrict__ fDistribution;
   std::mutex *fMutex;

    FUNCTOR fFunctor;

};

}// namespace detail

}// namespace hydra


#endif
