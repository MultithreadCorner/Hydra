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
 * Random.h
 *
 *  Created on: 07/08/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef RANDOM_H_
#define RANDOM_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/functors/RandomUtils.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Containers.h>
#include <hydra/Range.h>

//
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/random.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/partition.h>

#include <array>
#include <utility>

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
	void Gauss(double mean,double sigma,Iterator begin, Iterator end ) ;

	 /**
	     * @brief Fill the range [begin, end] with numbers distributed according a Gaussian distribution.
	     * @param mean \f$\mu\f$ of the Gaussian distribution.
	     * @param sigma \f$\sigma\f$ of the Gaussian distribution.
	     * @param begin Iterator pointing to the begin of the range.
	     * @param end Iterator pointing to the end of the range.
	     */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Gauss( hydra::detail::BackendPolicy<BACKEND> const& policy,
			double mean,double sigma, Iterator begin, Iterator end );

    /**
     * @brief Fill the range [begin, end] with numbers distributed according a Gaussian distribution.
     * @param mean \f$\mu\f$ of the Gaussian distribution.
     * @param sigma \f$\sigma\f$ of the Gaussian distribution.
     * @param output range to store the generated values.
     */
	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
					 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Gauss(double mean, double sigma,Iterable&& output) ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Exp(double tau, Iterator begin, Iterator end)  ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Exp(hydra::detail::BackendPolicy<BACKEND> const& policy, double tau, Iterator begin, Iterator end)  ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Exponential distribution.
	 * @param tau \f$\tau\f$ of the Exponential distribution
	 * @param output range to store the generated values.
	 */
	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
						 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Exp(double tau, Iterable&& output );

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void BreitWigner(double mean,double gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void BreitWigner(hydra::detail::BackendPolicy<BACKEND> const& policy,
			double mean, double gamma, Iterator begin, Iterator end)  ;//-> decltype(*begin);

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Breit-Wigner distribution.
	 * @param mean \f$\mu\f$ of the Breit-Wigner distribution.
	 * @param gamma \f$\\Gamma\f$ of the Breit-Wigner distribution.
	 * @param output range to store the generated values.
	 */
	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
							 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	BreitWigner(double mean, double gamma, Iterable&& output)  ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<typename Iterator>
	void Uniform(double min, double max, Iterator begin, Iterator end) ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param begin Iterator pointing to the begin of the range.
	 * @param end Iterator pointing to the end of the range.
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator>
	void Uniform(hydra::detail::BackendPolicy<BACKEND> const& policy,
			double min, double max, Iterator begin, Iterator end) ;

	/**
	 * @brief Fill the range [begin, end] with numbers distributed according a Uniform distribution.
	 * @param min minimum
	 * @param max maximum
	 * @param output range to store the generated values.
	 */
	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
	hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Uniform(double min,	double max, Iterable&& output) ;


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param begin beginning of the range storing the generated values
	 * @param end ending of the range storing the generated values
	 * @param min lower limit of sampling region
	 * @param max upper limit of sampling region.
	 * @param functor distribution to be sampled
	 * @return range with the generated values
	 */
	template<typename Iterator, typename FUNCTOR>
	Range<Iterator> Sample(Iterator begin, Iterator end , double min, double max, FUNCTOR const& functor);

	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param begin beginning of the range storing the generated values
	 * @param end ending of the range storing the generated values
	 * @param min lower limit of sampling region
	 * @param max upper limit of sampling region.
	 * @param functor distribution to be sampled
	 * @return range with the generated values
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator, typename FUNCTOR>
	Range<Iterator> Sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
			Iterator begin, Iterator end ,	double min, double max, FUNCTOR const& functor);


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param output range storing the generated values
	 * @param min lower limit of sampling region
	 * @param max upper limit of sampling region.
	 * @param functor distribution to be sampled
	 * @return range with the generated values
	 */
	template< typename Iterable, typename FUNCTOR>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
		hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Sample(Iterable&& output, double min, double max, FUNCTOR const& functor);

	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param begin beginning of the range storing the generated values
	 * @param end ending of the range storing the generated values
	 * @param min array of lower limits of sampling region
	 * @param max array of upper limits of sampling region.
	 * @param functor distribution to be sampled
	 * @return range with the generated values
	 */
	template<typename Iterator, typename FUNCTOR, size_t N >
	Range<Iterator> Sample(Iterator begin, Iterator end ,
			std::array<double,N>const& min,
			std::array<double,N>const& max,
			FUNCTOR const& functor);


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param policy backend to perform the calculation.
	 * @param begin beginning of the range storing the generated values
	 * @param end ending of the range storing the generated values
	 * @param min array of lower limits of sampling region
	 * @param max array of upper limits of sampling region.
	 * @param functor distribution to be sampled
	 */
	template<hydra::detail::Backend  BACKEND, typename Iterator, typename FUNCTOR, size_t N >
	Range<Iterator> Sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
			Iterator begin, Iterator end ,
			std::array<double,N>const& min,
			std::array<double,N>const& max,
			FUNCTOR const& functor);


	/**
	 * @brief Fill a range with numbers distributed according a user defined distribution.
	 * @param output range storing the generated values
	 * @param min array of lower limits of sampling region
	 * @param max array of upper limits of sampling region.
	 * @param functor distribution to be sampled
	 * @return output range with the generated values
	 */
	template<typename Iterable, typename FUNCTOR, size_t N >
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
			hydra::Range<decltype(std::declval<Iterable>().begin())>>::type Sample(Iterable&& output ,
			std::array<double,N>const& min,
			std::array<double,N>const& max,
			FUNCTOR const& functor);

private:
	GUInt_t fSeed;

};

/**
 * \ingroup random
 *
 * This functions reorder a dataset to put produce a unweighted sample according to the weights
 * [wbegin, wend]. The length of the range [wbegin, wend] should be equal or greater than
 * the dataset size.
 *
 * @param policy parallel backend to perform the unweighting
 * @param wbegin iterator pointing to the begin of the range of weights
 * @param wend  iterator pointing to the begin of the range of weights
 * @param begin iterator pointing to the begin of the range of data
 * @return
 */
template<hydra::detail::Backend  BACKEND, typename Iterator1, typename Iterator2>
Range<Iterator2> unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator1 wbegin, Iterator1 wend , Iterator2 begin);



/**
 * \ingroup random
 *
 * This functions reorder a dataset to put produce a unweighted sample according to @param functor .
 *
 * @param policy
 * @param begin
 * @param end
 * @param functor
 * @return the index of the last entry of the unweighted event.
 */
template<hydra::detail::Backend  BACKEND, typename Functor, typename Iterator>
typename std::enable_if< hydra::detail::is_hydra_functor<Functor>::value, Range<Iterator>>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end, Functor const& functor);





}
#endif /* RANDOM_H_ */
#include <hydra/detail/Random.inl>
