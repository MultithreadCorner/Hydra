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

#ifndef RANDOMUTILS_H_
#define RANDOMUTILS_H_

#include <hydra/detail/Config.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <hydra/detail/utility/Utility_Tuple.h>

namespace hydra{

template<typename GRND, typename FUNCTOR>
struct HyRndCDF{

	HyRndCDF(FUNCTOR const& functor, size_t seed):
		fSeed(seed),
		fFunctor(functor)
	{}
	__host__ __device__
	HyRndCDF( HyRndCDF<GRND, FUNCTOR> const& other):
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

template<typename GRND>
struct HyRndGauss{

	HyRndGauss(size_t seed, GReal_t mean, GReal_t  sigma):
		fSeed(seed),
		fMean(mean),
		fSigma(sigma)
	{}
	__host__ __device__
	HyRndGauss( HyRndGauss<GRND> const& other):
		fSeed(other.fSeed),
		fMean(other.fMean),
		fSigma(other.fSigma)
	{}

	__host__ __device__
	inline GReal_t operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::random::normal_distribution<GReal_t> dist(fMean, fSigma);
		GReal_t x = dist(randEng);
		//printf("Gauss %f\n",x);
		return x;
	}

	size_t fSeed;
	GReal_t fMean;
	GReal_t fSigma;
};

template<typename GRND>
struct HyRndUniform{

	HyRndUniform(size_t seed, GReal_t min, GReal_t max):
		fSeed(seed),
		fMin(min),
		fMax(max)
	{}
	__host__ __device__
	HyRndUniform( HyRndUniform<GRND> const& other):
		fSeed(other.fSeed),
		fMin(other.fMin),
		fMax(other.fMax)
	{}


	__host__ __device__
	inline GReal_t operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<GReal_t>  dist(fMin, fMax);
		return dist(randEng);
	}

	size_t fSeed;
	GReal_t fMin;
	GReal_t fMax;
};


template<typename GRND>
struct HyRndExp{

	HyRndExp(size_t seed, GReal_t tau):
		fSeed(seed),
		fTau(tau)
	{}
	__host__ __device__
	HyRndExp( HyRndExp<GRND> const& other):
		fSeed(other.fSeed),
		fTau(other.fTau)
	{}

	__host__ __device__
	inline GReal_t operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);
		thrust::uniform_real_distribution<GReal_t>  dist(0, 1);
		return  -fTau*log(dist(randEng));
	}

	size_t fSeed;
	GReal_t fTau;
};

template<typename GRND>
struct HyRndBreitWigner{

	HyRndBreitWigner(size_t seed, GReal_t mean, GReal_t gamma):
		fSeed(seed),
		fMean(mean),
		fGamma(gamma)
	{}

	__host__ __device__
	HyRndBreitWigner( HyRndBreitWigner<GRND> const& other):
		fSeed(other.fSeed),
		fMean(other.fMean),
		fGamma(other.fGamma)
	{}

	__host__ __device__
	inline GReal_t operator()(size_t index)
	{
		GRND randEng(fSeed);
		randEng.discard(index);

		thrust::uniform_real_distribution<GReal_t>  dist(0.0, 1.0);
		GReal_t rval  = dist(randEng);
		GReal_t x = fMean + 0.5*fGamma*tan(PI*(rval-0.5));
		return  x;
	}

	size_t fSeed;
	GReal_t fMean;
	GReal_t fGamma;

};

template<typename GRND>
struct HyRndFlag{

	HyRndFlag(const size_t seed, const GReal_t max_value):
		fSeed(seed),
		fValMax(max_value)
	{}

	__host__ __device__
	HyRndFlag(HyRndFlag<GRND> const& other):
		fSeed(other.fSeed),
		fValMax(other.fValMax)
	{}

	__host__ __device__
	inline GBool_t operator()(size_t index, GReal_t x)
	{
		GRND randEng(fSeed*2);
		randEng.discard(index);
		thrust::uniform_real_distribution<GReal_t>  dist(0.0, fValMax);

		return (x > dist(randEng)) ;
	}

	size_t  fSeed;
	GReal_t fValMax;

};


template<typename GRND, typename FUNCTOR, size_t N>
struct HyRndTrial{

	HyRndTrial(size_t seed, FUNCTOR const& functor, std::array<GReal_t,N> min, std::array<GReal_t,N> max):
		fSeed(seed),
		fFunctor(functor)
	{
		for(size_t i=0; i<N; i++){
			fMin[i] = min[i];
			fMax[i] = max[i];
		}
	}

	__host__ __device__
	HyRndTrial(HyRndTrial<GRND,FUNCTOR, N> const& other):
		fSeed(other.fSeed),
		fFunctor(other.fFunctor)
	{
		for(size_t i=0; i<N; i++){
			fMin[i] = other.fMin[i];
			fMax[i] = other.fMax[i];
		}
	}

	__hydra_exec_check_disable__
	__host__ __device__
	~HyRndTrial(){};

	template<typename T>
	__host__ __device__
	inline GReal_t operator()(size_t index, T& t)
	{
		GReal_t* x[N];
		detail::set_ptrs_to_tuple(t, &x[0]);

		GRND randEng(fSeed);
		randEng.discard(index);

		for (size_t j = 0; j < N; j++)
		{
			thrust::uniform_real_distribution<GReal_t>  dist(fMin[j], fMax[j]);
			*(x[j]) = dist(randEng);
		}

		return  fFunctor(t);
	}

	FUNCTOR fFunctor;
	size_t  fSeed;
	GReal_t fMin[N];
	GReal_t fMax[N];
};



}
#endif /* RANDOMUTILS_H_ */
