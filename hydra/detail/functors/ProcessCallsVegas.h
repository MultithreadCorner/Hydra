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
/*
template<size_t N>
struct ResultVegas
{

	GReal_t integral;
	GReal_t tss;
	GReal_t fDistribution[N*BINS_MAX];

	__host__ __device__
	ResultVegas():
	integral(0),
	tss(0)
	{
		for(size_t i=0; i< N*BINS_MAX; i++)
		fDistribution[i]=0;
	}

	__host__ __device__
	ResultVegas(ResultVegas<N> const& other):
	integral(other.integral),
	tss(other.tss)
	{
		for(size_t i=0; i< N*BINS_MAX; i++)
			fDistribution[i]=other.fDistribution[i];
	}


};
*/

template<size_t N>
struct ResultVegas
{


    size_t  fN;
    GReal_t fMin;
    GReal_t fMax;
    GReal_t fMean;
    GReal_t fM2;
   // GReal_t fDistribution[N*BINS_MAX];

    __host__ __device__
    ResultVegas():
    fN(0),
    fMean(0),
    fM2(0),
    fMin(std::numeric_limits<GReal_t>::min() ),
    fMax(std::numeric_limits<GReal_t>::max() )
    {
    	/*
    	for(size_t i=0; i< N*BINS_MAX; i++)
    		fDistribution[i]=0;
    		*/
    }

    __host__ __device__
    ResultVegas( ResultVegas<N> const& other):
    fN(other.fN),
    fMean(other.fMean ),
    fM2(other.fM2 ),
    fMin(other.fMin  ),
    fMax(other.fMax  )
    {
    	/*
    	for(size_t i=0; i< N*BINS_MAX; i++)
    		fDistribution[i]=other.fDistribution[i];
    		*/
    }



    __host__ __device__ inline
    GReal_t variance()   { return fM2 / (fN - 1); }

    __host__ __device__ inline
    GReal_t variance_n() { return fM2 / fN; }

};



template<size_t N>
struct ProcessBoxesVegas
		:public thrust::binary_function< ResultVegas<N> const&, ResultVegas<N> const& , ResultVegas<N> >
{


    __host__ __device__ inline
    ResultVegas<N> operator()(ResultVegas<N> const& x, ResultVegas<N> const& y)
    {
    	ResultVegas<N> result;

        // precompute some common subexpressions
        size_t n  = x.fN + y.fN;

        GReal_t delta  = y.fMean - x.fMean;
        GReal_t delta2 = delta  * delta;

        //Basic number of samples (n), min, and max
        result.fN   = n;
        result.fMin = thrust::min(x.fMin, y.fMin);
        result.fMax = thrust::max(x.fMax, y.fMax);

        result.fMean = x.fMean + delta * y.fN / n;

        result.fM2  = x.fM2 + y.fM2;
        result.fM2 += delta2 * x.fN * y.fN / n;

        /*
        for(size_t i=0; i< N*BINS_MAX; i++)
        			result.fDistribution[i] += x.fDistribution[i] + y.fDistribution[i];
        */
        return result;
    }

};

template<typename FUNCTOR, size_t NDimensions, typename Precision,  typename GRND=thrust::random::default_random_engine>
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
	ProcessCallsVegas( ProcessCallsVegas<FUNCTOR,NDimensions, Precision, GRND> const& other):
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
	{}

	__host__ __device__
	~ProcessCallsVegas(){ };

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

	__host__   __device__ inline
	void fill_distribution(size_t bin, size_t dimension, GReal_t fval )
	{
#ifdef __CUDA_ARCH__

#if __CUDA_ARCH__ >= 600
					atomicAdd((fDistribution + bin*NDimensions + dimension) ,  static_cast<Precision>(fval*fval));
#else
					atomicAdd((fDistribution + bin*NDimensions + dimension) , static_cast<Precision>(fval*fval));
#endif

#else
					std::lock_guard<std::mutex> lock(*fMutex);
					*(fDistribution + bin* NDimensions + dimension) += static_cast<Precision>(fval*fval);
#endif
	}

	__host__ __device__
	inline ResultVegas<NDimensions> operator()( size_t box)
	{


		GReal_t volume = 1.0;
		GReal_t x[NDimensions];
		GInt_t bin[NDimensions];
		ResultVegas<NDimensions> result;

		GRND randEng( hash(fSeed,box) );
		randEng.discard(NDimensions + box*fNCallsPerBox);
		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);


		//	for (size_t call = 0; call < fNCallsPerBox; call++)
		//	{

		for (size_t j = 0; j < NDimensions; j++) {

			x[j] = uniDist(randEng);

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

			//	printf("index=%f call=%f x[j]=%f \n",  double(box) , double(call), x[j] );

		}


		GReal_t fval = fJacobian*volume*fFunctor( detail::arrayToTuple<GReal_t, NDimensions>(x));


		//	if (fMode != MODE_STRATIFIED){

		for (GUInt_t j = 0; j < NDimensions; j++) {
					fill_distribution( bin[j], j,  fval );
		}

		//	}

		result.fN    = 1;
		result.fMin  = fval;
		result.fMax  = fval;
		result.fMean = fval;
		result.fM2   = 0;

		//	}




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

   Precision*   fDistribution;
   GReal_t*   __restrict__ fXi;
   GReal_t*   __restrict__ fXLow;
   GReal_t*   __restrict__ fDeltaX;

   std::mutex *fMutex;
   FUNCTOR fFunctor;

};

}// namespace detail

}// namespace hydra


#endif
