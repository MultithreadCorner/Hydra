/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
#include <hydra/GenericRange.h>

//
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/random.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/partition.h>

namespace hydra{


/*! \typedef default_random_engine
 *  \brief An implementation-defined "default" random number engine.
 *  \note \p default_random_engine is currently an alias for \p minstd_rand, and may change
 *        in a future version.
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::default_random_engine default_random_engine;


/*! \typedef minstd_rand0
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand0
 *        shall produce the value \c 1043618065 .
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::minstd_rand0 minstd_rand0;

/*! \typedef minstd_rand
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand
 *        shall produce the value \c 399268537 .
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::minstd_rand minstd_rand;


/*! \typedef ranlux24
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-3 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux24
 *        shall produce the value \c 9901578 .
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::ranlux24	ranlux24;

/*! \typedef ranlux48
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-4 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux48
 *        shall produce the value \c 88229545517833 .
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::ranlux48	ranlux48;

/*! \typedef taus88
 *  \brief A random number engine with predefined parameters which implements
 *         L'Ecuyer's 1996 three-component Tausworthe random number generator.
 *
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p taus88
 *        shall produce the value \c 3535848941 .
 */
typedef HYDRA_EXTERNAL_NS::thrust::random::taus88 	taus88;

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
template<typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
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
	 * \warning{ the implementation of HYDRA_EXTERNAL_NS::thrust::random::normal_distribution
	 * is different between nvcc and gcc. Do not expect the same
	 * numbers event by event.
	 * Possible: implement myself ? (que inferno! :0)
	 * Refs: see in hydra/detail/external/thrust/random/detail/normal_distribution_base.h
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
	void Gauss(typename Iterator::value_type mean,typename Iterator::value_type sigma,
			Iterator begin, Iterator end ) ;
	 /**
	     * @brief Fill the range [begin, end] with numbers distributed according a Gaussian distribution.
	     * @param mean \f$\mu\f$ of the Gaussian distribution.
	     * @param sigma \f$\sigma\f$ of the Gaussian distribution.
	     * @param begin Iterator pointing to the begin of the range.
	     * @param end Iterator pointing to the end of the range.
	     */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Gauss( hydra::detail::BackendPolicy<BACKEND> const& policy, typename Iterator::value_type mean,typename Iterator::value_type sigma,
				Iterator begin, Iterator end ) ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Exp(typename Iterator::value_type tau, Iterator begin, Iterator end)  ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Exp(hydra::detail::BackendPolicy<BACKEND> const& policy, typename Iterator::value_type tau, Iterator begin, Iterator end)  ;


	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void BreitWigner(typename Iterator::value_type mean,
			typename Iterator::value_type gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void BreitWigner(hydra::detail::BackendPolicy<BACKEND> const& policy, typename Iterator::value_type mean,
			typename Iterator::value_type gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Uniform(typename Iterator::value_type min,
			typename Iterator::value_type max, Iterator begin, Iterator end) ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Uniform(hydra::detail::BackendPolicy<BACKEND> const& policy, typename Iterator::value_type min,
			typename Iterator::value_type max, Iterator begin, Iterator end) ;


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min GReal_t min with lower limit of sampling region.
	 * @param max GReal_t max with upper limit of sampling region.
	 * @param trials number of trials.
	 * @return a hydra::backend::vector<tuple<T1,T2...>>
	 */
	template<typename T, typename Iterator, typename FUNCTOR>
	GenericRange<Iterator> Sample(Iterator begin, Iterator end ,
			T min, T max, FUNCTOR const& functor);

	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min GReal_t min with lower limit of sampling region.
	 * @param max GReal_t max with upper limit of sampling region.
	 * @param trials number of trials.
	 * @return a hydra::backend::vector<tuple<T1,T2...>>
	 */
	template<hydra::detail::Backend  BACKEND, typename T, typename Iterator, typename FUNCTOR>
	GenericRange<Iterator> Sample(hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end ,
			T min, T max, FUNCTOR const& functor);


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param functor hydra::Pdf instance that will be sampled.
	 * @param min std::array<GReal_t,N> min with lower limit of sampling region.
	 * @param max  std::array<GReal_t,N> min with upper limit of sampling region.
	 * @param trials number of trials.
	 * @return a hydra::backend::vector<tuple<T1,T2...>>
	 */
	template<typename T, typename Iterator, typename FUNCTOR, size_t N >
	GenericRange<Iterator> Sample(Iterator begin, Iterator end ,
			std::array<T,N>const& min,
			std::array<T,N>const& max,
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
	template<hydra::detail::Backend  BACKEND, typename T, typename Iterator, typename FUNCTOR, size_t N >
	GenericRange<Iterator> Sample(hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end ,
			std::array<T,N>const& min,
			std::array<T,N>const& max,
			FUNCTOR const& functor);


private:
	GUInt_t fSeed;

};




}
#endif /* RANDOM_H_ */
#include <hydra/detail/Random.inl>
