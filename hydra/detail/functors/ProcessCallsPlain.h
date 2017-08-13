
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
 * ProcessCallsPlain.h
 *
 *  Created on: 30/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */


#ifndef PROCESSCALLSPLAIN_H_
#define PROCESSCALLSPLAIN_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/PlainState.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <thrust/random.h>

namespace hydra {

namespace detail {

// ProcessCallsPlainUnary is a functor that takes in a value x and
// returns a PlainState whose mean value is initialized to f(x).
template <typename FUNCTOR, size_t N, typename GRND=thrust::random::default_random_engine>
struct ProcessCallsPlainUnary
{

	ProcessCallsPlainUnary(GReal_t* XLow, GReal_t  *DeltaX, size_t seed, FUNCTOR const& functor):
		fSeed(seed),
		fXLow(XLow),
		fDeltaX(DeltaX),
		fFunctor(functor)
	{}

	__host__ __device__ inline
	ProcessCallsPlainUnary( ProcessCallsPlainUnary<FUNCTOR,N, GRND> const& other):
	fSeed(other.fSeed),
	fXLow(other.fXLow),
	fDeltaX(other.fDeltaX),
	fFunctor(other.fFunctor)
	{}

	template<typename GRND2>
	 __host__ __device__ inline
	ProcessCallsPlainUnary( ProcessCallsPlainUnary<FUNCTOR,N, GRND2> const& other):
	    fSeed(other.fSeed),
		fXLow(other.fXLow),
		fDeltaX(other.fDeltaX),
		fFunctor(other.fFunctor)
	{}



	__host__ __device__ inline
	PlainState operator()(size_t index)
	 {

		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		GReal_t x[N];

		for (size_t j = 0; j < N; j++) {
			GReal_t r =  uniDist(randEng);
			x[j] = fXLow[j] + r*fDeltaX[j];
		}

		GReal_t fval = fFunctor( detail::arrayToTuple<GReal_t, N>(x));

		PlainState result;
		result.fN    = 1;
		result.fMin  = fval;
		result.fMax  = fval;
		result.fMean = fval;
		result.fM2   = 0;

		return result;
	}

	size_t fSeed;
	FUNCTOR fFunctor;
	GReal_t* __restrict__ fXLow;
	GReal_t* __restrict__ fDeltaX;
};



// ProcessCallsPlainBinary is a functor that accepts two PlainState
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for
// all values that have been agregated so far
struct ProcessCallsPlainBinary
    : public thrust::binary_function<PlainState const&,
                                     PlainState const&,
                                     PlainState >
{
    __host__ __device__ inline
    PlainState operator()(const PlainState& x, const PlainState& y)
    {
    	PlainState result;

        // precompute some common subexpressions
        size_t n  = x.fN + y.fN;

        GReal_t delta  = y.fMean - x.fMean;
        GReal_t delta2 = delta  * delta;

        //Basic number of samples (n), min, and max
        result.fN   = n;
        result.fMin = thrust::min(x.fMin, y.fMin);
        result.fMax = thrust::max(x.fMax, y.fMax);

        result.fMean = (x.fMean* x.fN +  y.fMean* y.fN) / n;

        result.fM2  = x.fM2 + y.fM2;
        result.fM2 += delta2 * x.fN * y.fN / n;

        return result;
    }
};

}// namespace detail

}// namespace hydra

#endif /* PROCESSCALLSPLAIN_H_ */
