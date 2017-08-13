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
 * RandomUtils.h
 *
 *  Created on: 07/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup random
 */


#ifndef RANDOMUTILS_H_
#define RANDOMUTILS_H_

#include <hydra/detail/Config.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <hydra/detail/utility/Utility_Tuple.h>

namespace hydra{

namespace detail {

template<typename GRND, typename FUNCTOR>
struct RndCDF{

	RndCDF(FUNCTOR const& functor, size_t seed):
		fSeed(seed),
		fFunctor(functor)
	{}
	__host__ __device__
	RndCDF( RndCDF<GRND, FUNCTOR> const& other):
		fSeed(other.fSeed),
		fFunctor(other.fFunctor)
	{}

	__host__ __device__
	inline GReal_t operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<GReal_t> dist(0.0, 1.0);
		GReal_t x = dist(randEng);
		return fFunctor(x);
	}

	size_t fSeed;
	FUNCTOR fFunctor;
};

template<typename T, typename GRND>
struct RndGauss{

	/**
	 * \warning: the implementation of thrust::random::normal_distribution
	 * is different between nvcc and gcc. Do not expect the same
	 * numbers event by event.
	 * Possible: implement myself ? (que inferno! :0)
	 * Refs: thrust/random/detail/normal_distribution_base.h
	 * ```
	 * template<typename RealType>
  	 * struct normal_distribution_base
	 * {
	 *	#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  	 *  typedef normal_distribution_nvcc<RealType> type;
	 *	#else
  	 *  typedef normal_distribution_portable<RealType> type;
	 *  #endif
	 * };
	 *	```
	 *
	 */
	RndGauss(size_t seed, T mean, T  sigma):
		fSeed(seed),
		fMean(mean),
		fSigma(sigma)
	{}
	__host__ __device__
	RndGauss( RndGauss<T, GRND> const& other):
		fSeed(other.fSeed),
		fMean(other.fMean),
		fSigma(other.fSigma)
	{}

	__host__ __device__
	inline T operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::random::normal_distribution<T> dist(fMean, fSigma);
		T x = dist(randEng);
		//printf("Gauss %f\n",x);
		return x;
	}

	size_t fSeed;
	T fMean;
	T fSigma;
};

template<typename T,typename GRND>
struct RndUniform{

	RndUniform(size_t seed, T min, T max):
		fSeed(seed),
		fMin(min),
		fMax(max)
	{}
	__host__ __device__
	RndUniform( RndUniform<T,GRND> const& other):
		fSeed(other.fSeed),
		fMin(other.fMin),
		fMax(other.fMax)
	{}


	__host__ __device__
	inline T operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<T>  dist(fMin, fMax);
		return dist(randEng);
	}

	size_t fSeed;
	T fMin;
	T fMax;
};


template<typename T,typename GRND>
struct RndExp{

	RndExp(size_t seed, T tau):
		fSeed(seed),
		fTau(tau)
	{}
	__host__ __device__
	RndExp( RndExp<T, GRND> const& other):
		fSeed(other.fSeed),
		fTau(other.fTau)
	{}

	__host__ __device__
	inline T operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<T>  dist(0.0, 1.0);
		return  -fTau*log(dist(randEng));
	}

	size_t fSeed;
	T fTau;
};

template<typename T,typename GRND>
struct RndBreitWigner{

	RndBreitWigner(size_t seed, T mean, T gamma):
		fSeed(seed),
		fMean(mean),
		fGamma(gamma)
	{}

	__host__ __device__
	RndBreitWigner( RndBreitWigner<T, GRND> const& other):
		fSeed(other.fSeed),
		fMean(other.fMean),
		fGamma(other.fGamma)
	{}

	__host__ __device__
	inline T operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);

		thrust::uniform_real_distribution<T>  dist(0.0, 1.0);
		T rval  = dist(randEng);
		T x = fMean + 0.5*fGamma*tan(PI*(rval-0.5));
		return  x;
	}

	size_t fSeed;
	T fMean;
	T fGamma;

};

template<typename T,typename GRND>
struct RndFlag{

	RndFlag(const size_t seed, const T max_value, const T* values ):
		fSeed(seed),
		fValMax(max_value),
		fVals(values)
	{}

	__host__ __device__
	RndFlag(RndFlag<T, GRND> const& other):
		fSeed(other.fSeed),
		fValMax(other.fValMax),
		fVals(other.fVals)
	{}

	__host__ __device__
	inline GBool_t operator()(size_t index)
	{
		GRND randEng(fSeed*2);
		randEng.discard(index);
		thrust::uniform_real_distribution<T>  dist(0.0, fValMax);

		return (fVals[index] > dist(randEng)) ;
	}

	size_t  fSeed;
	T fValMax;
	const T* __restrict__ fVals;
};


template<typename T,typename GRND, typename FUNCTOR, size_t N>
struct RndTrial{

	RndTrial(size_t seed, FUNCTOR const& functor,
			std::array<T,N>const& min,
			std::array<T,N>const& max):
				fFunctor(functor),
				fSeed(seed)
	{
		for(size_t i=0; i<N; i++){
			fMin[i] = min[i];
			fMax[i] = max[i];
		}
	}

	__host__ __device__
	RndTrial(RndTrial<T, GRND,FUNCTOR, N> const& other):
		fFunctor(other.fFunctor),
		fSeed(other.fSeed)
	{
		for(size_t i=0; i<N; i++){
			fMin[i] = other.fMin[i];
			fMax[i] = other.fMax[i];
		}
	}


	template<typename Tuple>
	__host__ __device__
	inline T operator()(size_t index, Tuple& t)
	{
		T* x[N];
		detail::set_ptrs_to_tuple(t, &x[0]);

		GRND randEng(fSeed);
		randEng.discard(index);

		for (size_t j = 0; j < N; j++)
		{
			thrust::uniform_real_distribution<T>  dist(fMin[j], fMax[j]);
			*(x[j]) = dist(randEng);
		}

		return  fFunctor(t);
	}

	FUNCTOR fFunctor;
	size_t  fSeed;
	T fMin[N];
	T fMax[N];
};


template<typename T,typename GRND, typename FUNCTOR>
struct RndTrial<T,GRND, FUNCTOR, 1>{

	RndTrial(size_t seed, FUNCTOR const& functor, GReal_t min, GReal_t max):
		fFunctor(functor),
		fSeed(seed),
		fMin(min),
		fMax(max)
	{}

	__host__ __device__
	RndTrial(RndTrial<T, GRND,FUNCTOR, 1> const& other):
		fFunctor(other.fFunctor),
		fSeed(other.fSeed),
		fMin(other.fMin),
		fMax(other.fMax)
	{}


	__host__ __device__
	inline GReal_t operator()(size_t index, T& t)
	{

		GRND randEng(fSeed);
		randEng.discard(index);
    	thrust::uniform_real_distribution<T>  dist(fMin, fMax);
		t = dist(randEng);

		return  fFunctor(t);
	}

	FUNCTOR fFunctor;
	size_t  fSeed;
	GReal_t fMin;
	GReal_t fMax;
};

} // namespace detail

}// namespace hydra

#endif /* RANDOMUTILS_H_ */
