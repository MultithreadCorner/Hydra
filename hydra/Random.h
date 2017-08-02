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
#include <thrust/partition.h>

namespace hydra{

/**
 * @ingroup random
 * @brief This class implements functionalities associated to random number generation and pdf sampling.
 *
 * hydra::Random instances can sample multidimensional hydra::Pdf and fill ranges with data corresponding to
 * gaussian, exponential, uniform and Breit-Wigner distributions.
 *
 * @tparam GRND underlying random number generator.
 *
 */
template<typename GRND=thrust::random::default_random_engine>
class Random{

public:
	Random():
			fSeed(7895123)
	{}

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

    /**
     * @brief Fill the range [begin, end] with numbers distributed according a Gaussian distribution.
     * @param mean \f$\mu\f$ of the Gaussian distribution.
     * @param sigma \f$\sigma\f$ of the Gaussian distribution.
     * @param begin Iterator pointing to the begin of the range.
     * @param end Iterator pointing to the end of the range.
     */
	template<typename Iterator>
	void Gauss(GReal_t mean, GReal_t sigma, Iterator begin, Iterator end ) ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Exp(GReal_t tau, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void BreitWigner(GReal_t mean, GReal_t gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Uniform(GReal_t min, GReal_t max, Iterator begin, Iterator end) ;// -> decltype(*begin);


	template<typename FUNCTOR, typename Iterator>
	void InverseCDF(FUNCTOR const& invcdf, Iterator begin, Iterator end)  ;//-> decltype(*begin);


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min GReal_t min with lower limit of sampling region.
	 * @param max GReal_t max with upper limit of sampling region.
	 * @param trials number of trials.
	 * @return a hydra::backend::vector<tuple<T1,T2...>>
	 */
	template<typename ITERATOR, typename FUNCTOR>
	ITERATOR Sample(ITERATOR begin, ITERATOR end , GReal_t min, GReal_t max,
			FUNCTOR const& functor);

	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min std::array<GReal_t,N> min with lower limit of sampling region.
	 * @param max  std::array<GReal_t,N> min with upper limit of sampling region.
	 * @param trials number of trials.
	 * @return a hydra::backend::vector<tuple<T1,T2...>>
	 */
	template<typename ITERATOR, typename FUNCTOR, size_t N >
	ITERATOR Sample(ITERATOR begin, ITERATOR end , std::array<GReal_t,N>const& min, std::array<GReal_t,N>const& max,
			FUNCTOR const& functor);



	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min std::array<GReal_t,N> min with lower limit of sampling region.
	 * @param max  std::array<GReal_t,N> min with upper limit of sampling region.
	 * @param result
	 * @param trials
	 */
	/*
	template<hydra::detail::Backend BACKEND, typename FUNCTOR, size_t N >
	void Sample(hydra::detail::BackendPolicy<BACKEND>const&  policy, FUNCTOR const& functor, std::array<GReal_t,N> min, std::array<GReal_t,N> max,
			PointVector< Point<GReal_t, N, false, false>, BACKEND >& result, size_t trials);
			*/
private:
	GUInt_t fSeed;

};




}
#endif /* RANDOM_H_ */
#include <hydra/detail/Random.inl>
