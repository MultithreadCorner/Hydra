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
 * Random.h
 *
 *  Created on: 07/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup random
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <array>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/functors/RandomUtils.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Containers.h>
#include <hydra/PointVector.h>
//
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>


namespace hydra{

template<typename GRND=thrust::random::default_random_engine>
class Random{

public:
	Random(GUInt_t seed):
		fSeed(seed)
{}

	Random( Random const& other):
		fSeed(other.fSeed)
	{}

	~Random(){};

	GUInt_t GetSeed() const {
		return fSeed;
	}

	void SetSeed(GUInt_t seed) {
		fSeed = seed;
	}

	/**
	 * \warning{ the implementation of thrust::random::normal_distribution
	 * is different between nvcc and gcc. Do not expect the same
	 * numbers event by event.
	 * Possible: implement myself ? (que inferno! :0)
	 * Refs: see in thrust/random/detail/normal_distribution_base.h
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
	 *}
	 */
	template<typename Iterator>
	void Gauss(GReal_t mean, GReal_t sigma, Iterator begin, Iterator end ) ;//-> decltype(*begin);

	template<typename Iterator>
	void Exp(GReal_t tau, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	template<typename Iterator>
	void BreitWigner(GReal_t mean, GReal_t gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	template<typename Iterator>
	void Uniform(GReal_t min, GReal_t max, Iterator begin, Iterator end) ;// -> decltype(*begin);

	template<typename FUNCTOR, typename Iterator>
	void InverseCDF(FUNCTOR const& invcdf, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	template<typename BACKEND, typename FUNCTOR, size_t N>
	auto Sample(BACKEND&, FUNCTOR const& functor, std::array<GReal_t,N> min,
			std::array<GReal_t,N> max, size_t trials)
	->	typename BACKEND::container;

	template<typename BACKEND, typename FUNCTOR, size_t N >
	void Sample(FUNCTOR const& functor, std::array<GReal_t,N> min, std::array<GReal_t,N> max,
			PointVector<BACKEND, GReal_t, N, false, false>& result, size_t trials);

private:
	GUInt_t fSeed;

};




}
#endif /* RANDOM_H_ */
#include <hydra/detail/Random.inl>
