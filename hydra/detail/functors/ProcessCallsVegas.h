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
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/random.h>
#include <hydra/VegasState.h>


namespace hydra{

namespace detail {

struct ResultVegas
{
	GReal_t fN;
    GReal_t fMean;
    GReal_t fM2;

};



struct ProcessBoxesVegas
		:public HYDRA_EXTERNAL_NS::thrust::binary_function< ResultVegas const&, ResultVegas const& , ResultVegas >
{


    __hydra_host__ __hydra_device__ inline
    ResultVegas operator()(ResultVegas const& x, ResultVegas const& y)
    {
    	ResultVegas result;

        GReal_t n  = x.fN + y.fN;

        GReal_t delta  = y.fMean - x.fMean;
        GReal_t delta2 = delta  * delta;

        result.fN   = n;

        //result.fMean = x.fMean + delta * y.fN / n;
        result.fMean = (x.fMean*x.fN + y.fMean*y.fN)/n;
        result.fM2   = x.fM2   +  y.fM2;
        result.fM2  += delta2 * x.fN * y.fN / n;

        return result;
    }

};


template<typename FUNCTOR, size_t NDimensions, typename  BACKEND,
typename IteratorBackendReal, typename IteratorBackendUInt,
typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
struct ProcessCallsVegas;

template<typename FUNCTOR, size_t NDimensions,  hydra::detail::Backend  BACKEND,
typename IteratorBackendReal, typename IteratorBackendUInt, typename GRND>
struct ProcessCallsVegas<FUNCTOR,  NDimensions, hydra::detail::BackendPolicy<BACKEND>,
IteratorBackendReal,  IteratorBackendUInt, GRND>
{

	typedef   ProcessCallsVegas<FUNCTOR,  NDimensions, hydra::detail::BackendPolicy<BACKEND>,
			IteratorBackendReal,  IteratorBackendUInt, GRND> this_t;

	typedef  hydra::VegasState<NDimensions,hydra::detail::BackendPolicy<BACKEND>> state_t;

public :

	ProcessCallsVegas( size_t NBoxes, state_t& fState,	IteratorBackendUInt begin_bins,
			IteratorBackendReal begin_real,  FUNCTOR const& functor):
				fNBoxes( NBoxes ),
				fSeed(fState.GetItNum()),
				fNBins(fState.GetNBins()),
				fNBoxesPerDimension(fState.GetNBoxes()),
				fNCallsPerBox(fState.GetCallsPerBox()),
				fJacobian( fState.GetJacobian() ),
				fXi(fState.GetBackendXi().begin() ),
				fXLow( fState.GetBackendXLow().begin() ),
				fDeltaX( fState.GetBackendDeltaX().begin() ),
				fGlobalBin( begin_bins ),
				fFVals( begin_real ),
				fFunctor(functor)
				{}

	__hydra_host__ __hydra_device__
	ProcessCallsVegas( this_t const& other):
	fSeed(other.fSeed),
	fNBins(other.fNBins),
	fNBoxes(other.fNBoxes),
	fNBoxesPerDimension(other.fNBoxesPerDimension),
	fNCallsPerBox(other.fNCallsPerBox),
	fJacobian(other.fJacobian),
	fXi(other.fXi),
	fXLow(other.fXLow),
	fDeltaX(other.fDeltaX),
	fGlobalBin(other.fGlobalBin),
	fFVals(other.fFVals),
	fFunctor(other.fFunctor)
	{}


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


	__hydra_host__   __hydra_device__ inline
	void get_point(const size_t  index, GReal_t &volume, GInt_t (&bin)[NDimensions], GReal_t (&x)[NDimensions] )
	{

		size_t box = index/fNCallsPerBox;

		GRND randEng( hash(fSeed,index) );
		//randEng.discard(index);
		HYDRA_EXTERNAL_NS::thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		for (size_t j = 0; j < NDimensions; j++)
		{
			x[j] = uniDist(randEng);

			GInt_t b = fNBoxesPerDimension > 1? GetBoxCoordinate(box, NDimensions, fNBoxesPerDimension, j):box;

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

			//	printf("j=%d k=%d z=%f y=%f box=%f  x[j]=%f \n", j, k, z,y, double(b) ,  x[j] );
}

	}

	__hydra_host__ __hydra_device__ inline
	size_t GetDistributionIndex(size_t index,  const GUInt_t dim) const
	{ return index*NDimensions + dim; }

	__hydra_host__ __hydra_device__ inline
	GUInt_t GetDistributionKey( const GUInt_t bin, const GUInt_t dim) const
	{ return bin * NDimensions + dim; }


	__hydra_host__ __hydra_device__ inline
	ResultVegas operator()( size_t index)
	{

		GReal_t volume = 1.0;
		GReal_t x[NDimensions];
		GInt_t bin[NDimensions];
		ResultVegas result;

		get_point( index, volume, bin, x );

		GReal_t fval = fJacobian*volume*fFunctor( detail::arrayToTuple<GReal_t, NDimensions>(x));

		for (GUInt_t j = 0; j < NDimensions; j++)
		{

		    fGlobalBin[ GetDistributionIndex(index, j ) ] = GetDistributionKey(bin[j], j);
			fFVals[GetDistributionIndex(index, j )]=fval*fval;

		}

		result.fN    = 1.0;
		result.fMean = fval;
		result.fM2   = 0.0;

		return result;

	}

private:

	size_t  fNBins;
	size_t  fNBoxes;
	size_t  fNBoxesPerDimension;
	size_t  fNCallsPerBox;

	GReal_t fJacobian;
	GInt_t  fSeed;
	IteratorBackendUInt fGlobalBin;
	IteratorBackendReal fFVals;
	IteratorBackendReal  fXi;
	IteratorBackendReal  fXLow;
	IteratorBackendReal  fDeltaX;

	FUNCTOR fFunctor;



};

}// namespace detail

}// namespace hydra


#endif
