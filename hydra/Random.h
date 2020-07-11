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
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/random/philox.h>
#include <hydra/detail/random/threefry.h>
#include <hydra/detail/random/ars.h>
#include <hydra/detail/random/squares3.h>
#include <hydra/detail/random/squares4.h>

#include <hydra/Range.h>

//
#include <hydra/detail/external/hydra_thrust/copy.h>
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



/*! \typedef default_random_engine
 *  \brief An implementation-defined "default" random number engine.
 *  \note \p default_random_engine is currently an alias for \p hydra::random::squares3, and may change
 *        in a future version.
 */

typedef typename hydra::random::squares3 default_random_engine;

//typedef hydra_thrust::random::default_random_engine default_random_engine;
//typedef hydra::random::philox default_random_engine;
//typedef hydra::random::threefry default_random_engine;
//typedef hydra::random::ars default_random_engine;
//typedef hydra::random::squares3 default_random_engine;
//typedef hydra::random::squares4 default_random_engine;

/*! \typedef minstd_rand0
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand0
 *        shall produce the value \c 1043618065 .
 */
typedef hydra_thrust::random::minstd_rand0 minstd_rand0;

/*! \typedef minstd_rand
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand
 *        shall produce the value \c 399268537 .
 */
typedef hydra_thrust::random::minstd_rand minstd_rand;


/*! \typedef ranlux24
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-3 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux24
 *        shall produce the value \c 9901578 .
 */
typedef hydra_thrust::random::ranlux24	ranlux24;

/*! \typedef ranlux48
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-4 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux48
 *        shall produce the value \c 88229545517833 .
 */
typedef hydra_thrust::random::ranlux48	ranlux48;

/*! \typedef taus88
 *  \brief A random number engine with predefined parameters which implements
 *         L'Ecuyer's 1996 three-component Tausworthe random number generator.
 *
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p taus88
 *        shall produce the value \c 3535848941 .
 */
typedef hydra_thrust::random::taus88 	taus88;


/*! \typedef philox
 *  \brief The Philox family of counter-based RNGs use integer multiplication, xor and permutation of W-bit words
 *         to scramble its N-word input key.  Philox is a mnemonic for Product HI LO Xor).
 *
 */
typedef hydra::random::philox philox;

/*! \typedef threefry
 *  \brief Threefry uses integer addition, bitwise rotation, xor and permutation of words to randomize its output.
 *
 */
typedef hydra::random::threefry threefry;

/*! \typedef ars
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::ars ars;

/*! \typedef squares3
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::squares3 squares3;

/*! \typedef squares4
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::squares4 squares4;



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
 * @return
 */
template<typename RNG=default_random_engine, typename IteratorData, typename IteratorWeight, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
Range<IteratorData> >::type
unweight( detail::BackendPolicy<BACKEND> const& policy, IteratorData data_begin, IteratorData data_end, IteratorWeight weights_begin);

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
 * @return
 */
template<typename RNG=default_random_engine, typename IteratorData, typename IteratorWeight>
typename std::enable_if<
	detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
	Range<IteratorData>
>::type
unweight(IteratorData data_begin, IteratorData data_end , IteratorData weights_begin);

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
 * @return
 */
template<typename RNG=default_random_engine, typename IterableData, typename IterableWeight, hydra::detail::Backend BACKEND>
typename std::enable_if<
detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
Range< decltype(std::declval<IterableData>().begin())> >::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy,  IterableData&& data, IterableWeight&& weights);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to a weights.
 * The length of the range @param weights should be equal or greater than
 * the  @param data size.
 *
 * @param weights the range of weights
 * @param data the range corresponding dataset
 * @return
 */
template<typename RNG=default_random_engine, typename IterableData, typename IterableWeight>
typename std::enable_if<
	detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
	Range< decltype(std::declval<IterableData>().begin())>
>::type
unweight( IterableData data, IterableWeight weights);


/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param policy
 * @param begin
 * @param end
 * @param functor
 * @return the index of the last entry of the unweighted event.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end, Functor const& functor);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param begin
 * @param end
 * @param functor
 * @return the index of the last entry of the unweighted event.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( Iterator begin, Iterator end, Functor const& functor);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param iterable
 * @param functor
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
	Range< decltype(std::declval<Iterable>().begin())>
>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterable&& iterable, Functor const& functor);

/**
 * \ingroup random
 *
 * This functions reorder a dataset to produce an unweighted sample according to @param functor .
 *
 * @param iterable
 * @param functor
 * @return hydra::Range object pointing unweighted sample.
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
unweight( Iterable&& iterable, Functor const& functor);





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
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end, double min, double max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

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
template<typename RNG=default_random_engine, typename DerivedPolicy, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		Iterator begin, Iterator end, double min, double max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end , double min, double max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param output range storing the generated values
 * @param min lower limit of sampling region
 * @param max upper limit of sampling region.
 * @param functor distribution to be sampled
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample(Iterable&& output, double min, double max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @return range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end , std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND, size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N>const& min,	std::array<double,N>const& max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);
/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 */
template<typename RNG=default_random_engine, typename DerivedPolicy, typename Functor, typename Iterator, size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra_thrust::detail::execution_policy_base<DerivedPolicy>  const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N>const& min,	std::array<double,N>const& max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);

/**
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param output range storing the generated values
 * @param min array of lower limits of sampling region
 * @param max array of upper limits of sampling region.
 * @param functor distribution to be sampled
 * @return output range with the generated values
 */
template<typename RNG=default_random_engine, typename Functor, typename Iterable, size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample( Iterable&& output ,
		std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, size_t seed=0xb56c4feeef1b);




}

#include <hydra/detail/Random.inl>

#endif /* RANDOM_H_ */

