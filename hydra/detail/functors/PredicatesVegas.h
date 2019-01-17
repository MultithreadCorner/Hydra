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
 * PredicatesVegas.h
 *
 *  Created on: 13/12/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PREDICATESVEGAS_H_
#define PREDICATESVEGAS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/random.h>

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include <curand_kernel.h>
#endif


namespace hydra{

namespace detail {

template<typename FUNCTOR, size_t NDimensions, typename Precision, typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
struct ProcessCallsVegas
{
	ProcessCallsVegas( size_t NBins,
			size_t NBoxes,
			size_t NBoxesPerDimension,
			size_t NCallsPerBox,
			size_t Seed,
			GReal_t Jacobian,
			GReal_t* Xi,
			GReal_t* XLow,
			GReal_t* DeltaX,
			GReal_t* fval,
			GInt_t *bins[N],
			FUNCTOR const& functor):
		fSeed(Seed),
		fNBins(NBins),
		fNBoxes(NBoxes),
		fNBoxesPerDimension(NBoxesPerDimension),
		fNCallsPerBox(NCallsPerBox),
		fJacobian(Jacobian),
		fXi(Xi),
		fXLow(XLow),
		fDeltaX(DeltaX),
		fFVal(fval),
		fFunctor(functor)
	{
		for(size_t i=0; i<N; i++)
			fBins[i]=bins[i];

	}

	__hydra_host__ __hydra_device__
	ProcessCallsVegas( ProcessCallsVegas<FUNCTOR,NDimensions,Precision, GRND> const& other):
	fSeed(other.fSeed),
	fNBins(other.fNBins),
	fNBoxes(other.fNBoxes),
	fNBoxesPerDimension(other.fNBoxesPerDimension),
	fNCallsPerBox(other.fNCallsPerBox),
	fJacobian(other.fJacobian),
	fXi(other.fXi),
	fXLow(other.fXLow),
	fDeltaX(other.fDeltaX),
	fFVal(fval),
	fFunctor(other.fFunctor)
	{
		for(size_t i=0; i<N; i++)
		fBins[i]=bins[i];
	}

	__hydra_host__ __hydra_device__
	~ProcessCallsVegas(){

	};

	__hydra_host__ __hydra_device__
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

	__hydra_host__   __hydra_device__ inline
	size_t hash(size_t a, size_t b)
	{
		//Matthew Szudzik pairing
		//http://szudzik.com/ElegantPairing.pdf

		size_t  A = 2 * a ;
		size_t  B = 2 * b ;
		size_t  C = ((A >= B ? A * A + A + B : A + B * B) / 2);
		return  C ;
	}

#ifdef __CUDA_ARCH__
	__hydra_device__ inline
	void GetPointSobol(size_t index, GReal_t &volume, GReal_t (&point)[NDimensions], GUInt_t bin[NDimensions] )
	{
		skipahead(index*NDimensions, &fSobolState);

		for (size_t j = 0; j < NDimensions; j++) {

			x[j] = 	curand_uniform_double(&fSobolState);

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
	}
#endif

	__hydra_host__ __hydra_device__ inline
	void GetPointRandom(size_t index, GReal_t &volume, GReal_t (&point)[NDimensions], GUInt_t bin[NDimensions] )
	{
		fRandonEngine.discard(index*NDimensions);

		for (size_t j = 0; j < NDimensions; j++) {

			x[j] = fUniformDistribution(fRandonEngine );

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
	}


	__hydra_host__ __hydra_device__	inline
	void 	operator()( size_t index)
	{
		GReal_t volume = 1.0;
		GReal_t x[NDimensions];
		GInt_t bin[NDimensions];

#ifdef __CUDA_ARCH__
		GetPointSobol(index, volume, x, bin);
#else
		GetPointRandom(index, volume, x, bin);
#endif

		GReal_t fval = fJacobian*volume*fFunctor( detail::arrayToTuple<GReal_t, NDimensions>(x));
		fFVal[index] = fval;
		for(size_t i=0; i<N; i++)
				fBins[i][index]=bin[i];

	}

private:

   GReal_t*   __restrict__ fFVal;
   GInt_t*    __restrict__ fBins[N];

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
   curandStateSobol64_t state;
   curandDirectionVectors64_t dvector;
#endif

   GRND fRandonEngine;
   HYDRA_EXTERNAL_NS::thrust::uniform_real_distribution<GReal_t> fUniformDistribution;

	size_t  fNBins;
	size_t  fNBoxes;
	size_t  fNBoxesPerDimension;
	size_t  fNCallsPerBox;
    GReal_t fJacobian;
    GInt_t  fSeed;

   GReal_t*   __restrict__ fXi;
   GReal_t*   __restrict__ fXLow;
   GReal_t*   __restrict__ fDeltaX;

    FUNCTOR fFunctor;

};

}// namespace detail

}// namespace hydra


#endif /* PREDICATESVEGAS_H_ */
