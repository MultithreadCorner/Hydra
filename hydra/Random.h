/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
#include <hydra/detail/functors/DistributionSampler.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/ArgumentTraits.h>
#include <hydra/detail/PRNGTypedefs.h>

#include <hydra/Range.h>

//
#include <hydra/detail/external/hydra_thrust/copy.h>
#include <hydra/detail/external/hydra_thrust/tabulate.h>
#include <hydra/detail/external/hydra_thrust/random.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/partition.h>

#include <array>
#include <utility>

namespace hydra{

namespace detail {

namespace random {

template<typename T>
struct is_iterator: std::conditional<
        !hydra::detail::is_hydra_composite_functor<T>::value &&
		!hydra::detail::is_hydra_functor<T>::value &&
		!hydra::detail::is_hydra_lambda<T>::value &&
		!hydra::detail::is_iterable<T>::value &&
		 hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template<typename T>
struct is_iterable: std::conditional<
        !hydra::detail::is_hydra_composite_functor<T>::value &&
		!hydra::detail::is_hydra_functor<T>::value &&
		!hydra::detail::is_hydra_lambda<T>::value  &&
		 hydra::detail::is_iterable<T>::value &&
		!hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template<typename T>
struct is_callable: std::conditional<
        (hydra::detail::is_hydra_composite_functor<T>::value ||
         hydra::detail::is_hydra_functor<T>::value ||
         hydra::detail::is_hydra_lambda<T>::value ) &&
        !hydra::detail::is_iterable<T>::value &&
        !hydra::detail::is_iterator<T>::value,
         std::true_type,
         std::false_type >::type {};

template< typename Engine, typename Functor, typename Iterable>
struct is_matching_iterable: std::conditional<
     hydra::detail::is_iterable<Iterable>::value &&
    !hydra::detail::is_iterator<Iterable>::value &&
    (hydra::detail::is_hydra_composite_functor<Functor>::value ||
     hydra::detail::is_hydra_functor<Functor>::value ||
     hydra::detail::is_hydra_lambda<Functor>::value  ) &&
     hydra::detail::has_rng_formula<Functor>::value &&
     std::is_convertible<
    decltype(std::declval<RngFormula<Functor>>().Generate( std::declval<Engine&>(), std::declval<Functor const&>())),
    typename hydra_thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type>::value,
    std::true_type,  std::false_type
>::type{};
}  // namespace random

}  // namespace detail


/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce a unweighted sample according to the weights
 * [wbegin, wend]. The length of the range [wbegin, wend] should be equal or greater than
 * the dataset size.
 *
 * @param policy parallel backend to perform the unweighting
 * @param data_begin iterator pointing to the begin of the range of weights
 * @param data_end iterator pointing to the begin of the range of weights
 * @param weights_begin iterator pointing to the begin of the range of data
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename DerivedPolicy, typename IteratorData, typename IteratorWeight>
typename std::enable_if<
detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
Range<IteratorData> >::type
unweight( hydra_thrust::detail::execution_policy_base<DerivedPolicy>  const& policy,
		IteratorData data_begin, IteratorData data_end, IteratorWeight weights_begin,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce a unweighted sample according to the weights
 * [wbegin, wend]. The length of the range [wbegin, wend] should be equal or greater than
 * the dataset size.
 *
 * @param policy parallel backend to perform the unweighting
 * @param data_begin iterator pointing to the begin of the range of weights
 * @param data_end iterator pointing to the begin of the range of weights
 * @param weights_begin iterator pointing to the begin of the range of data
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename IteratorData, typename IteratorWeight, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
Range<IteratorData> >::type
unweight( detail::BackendPolicy<BACKEND> const& policy, IteratorData data_begin, IteratorData data_end, IteratorWeight weights_begin,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce a unweighted sample according to the weights
 * [wbegin, wend]. The length of the range [wbegin, wend] should be equal or greater than
 * the dataset size.
 *
 * @param data_begin iterator pointing to the begin of the range of weights
 * @param data_end iterator pointing to the begin of the range of weights
 * @param weights_begin iterator pointing to the begin of the range of data
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename IteratorData, typename IteratorWeight>
typename std::enable_if<
	detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
	Range<IteratorData>
>::type
unweight(IteratorData data_begin, IteratorData data_end , IteratorData weights_begin,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce a unweighted sample according to a weights.
 * The length of the range @param weights should be equal or greater than
 * the  @param data size.
 *
 * @param policy parallel backend to perform the unweighting
 * @param weights the range of weights
 * @param data the range corresponding dataset
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename IterableData, typename IterableWeight, hydra::detail::Backend BACKEND>
typename std::enable_if<
detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
Range< decltype(std::declval<IterableData>().begin())> >::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy,  IterableData&& data, IterableWeight&& weights,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to a weights.
 * The length of the range @param weights should be equal or greater than
 * the  @param data size.
 *
 * @param weights the range of weights
 * @param data the range corresponding dataset
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename IterableData, typename IterableWeight>
typename std::enable_if<
	detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
	Range< decltype(std::declval<IterableData>().begin())>
>::type
unweight( IterableData&& data, IterableWeight&& weights,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0 );


/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param policy
 * @param begin
 * @param end
 * @param functor
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, typename DerivedPolicy>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
	    	Iterator begin, Iterator end, Functor const& functor,
			double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0 );


/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param policy
 * @param begin
 * @param end
 * @param functor
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end, Functor const& functor,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param begin
 * @param end
 * @param functor
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( Iterator begin, Iterator end, Functor const& functor,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param iterable
 * @param functor
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
	Range< decltype(std::declval<Iterable>().begin())>
>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterable&& iterable, Functor const& functor,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0  );

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param iterable
 * @param functor
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
unweight( Iterable&& iterable, Functor const& functor,
		double max_pdf=-1.0, std::size_t rng_seed=0x8ec74d321e6b5a27, std::size_t rng_jump=0 );


/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end, double min, double max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename DerivedPolicy, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		Iterator begin, Iterator end, double min, double max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end , double min, double max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param output range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample(Iterable&& output, double min, double max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, std::size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end , std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min tuple of lower limits of sampling region
 * @param max tuple of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value  &&
detail::random::is_iterator<Iterator>::value &&
detail::is_tuple_type< decltype(*std::declval<Iterator>())>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end ,
		typename Functor::argument_type const& min, typename Functor::argument_type const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @param functor distribution to be sampled
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND, std::size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N>const& min,	std::array<double,N>const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );
/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @param functor distribution to be sampled
 */
template<typename RNG=default_random_engine, typename DerivedPolicy, typename Functor, typename Iterator, std::size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra_thrust::detail::execution_policy_base<DerivedPolicy>  const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N>const& min,	std::array<double,N>const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param output range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return output range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable, std::size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample( Iterable&& output ,
		std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param output range storing the generated values
 * @param min tuple of lower limits of sampling region
 * @param max tuple of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 * @return output range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value  &&
detail::random::is_iterable<Iterable>::value &&
detail::is_tuple_type< decltype(*std::declval<Iterable>().begin())>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample( Iterable&& output ,
		typename Functor::argument_type const& min,typename Functor::argument_type  const& max,
		Functor const& functor, std::size_t seed=0xb56c4feeef1b, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine,  hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< hydra::detail::has_rng_formula<FUNCTOR>::value && std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename hydra_thrust::iterator_traits<Iterator>::value_type
>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine =hydra::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< hydra::detail::has_rng_formula<FUNCTOR>::value && std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename hydra_thrust::iterator_traits<Iterator>::value_type
>::value, void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
typename std::enable_if< detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterable&& iterable, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, typename Iterable, typename FUNCTOR >
typename std::enable_if< detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value,
void>::type
fill_random(Iterable&& iterable, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula is not implemented for the requested functor
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< !hydra::detail::has_rng_formula<FUNCTOR>::value , void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula is not implemented for the requested functor
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< !hydra::detail::has_rng_formula<FUNCTOR>::value , void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< !std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename std::iterator_traits<Iterator>::value_type
>::value && hydra::detail::has_rng_formula<FUNCTOR>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& funct, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< !std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename std::iterator_traits<Iterator>::value_type
>::value && hydra::detail::has_rng_formula<FUNCTOR>::value, void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if the argument is not an Iterable or if itis not convertible to the Functor return value
 * @param policy backend to perform the calculation.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
typename std::enable_if< !(detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value), void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterable&& iterable, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );

/**
 * \ingroup random
 *
 * @brief Fall back function if the argument is not an Iterable or if itis not convertible to the Functor return value
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 * @param max_pdf maximum pdf value for accept-reject method. If no value is set, the maximum value in the sample is used.
 * @param rng_seed seed for the underlying pseudo-random number generator
 * @param rng_jump sequence offset for the underlying pseudo-random number generator
 */
template< typename Engine = hydra::default_random_engine, typename Iterable, typename FUNCTOR >
typename std::enable_if<!(detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value),void>::type
fill_random(Iterable&& iterable, FUNCTOR const& functor, std::size_t seed=0x254a0afcf7da74a2, std::size_t rng_jump=0 );



}

#include <hydra/detail/Random.inl>
#include <hydra/detail/RandomFill.inl>

#endif /* RANDOM_H_ */

