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
 * RandomFill.h
 *
 *  Created on: 26/02/2020
 *      Author: Davide Brundu
 */


#ifndef RANDOMFILL_H_
#define RANDOMFILL_H_

#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/FormulaTraits.h>
#include <hydra/detail/RngFormula.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/Distribution.h>

#include <hydra/detail/external/hydra_thrust/random.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/tabulate.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <array>
#include <utility>



namespace hydra{


/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine,  hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< hydra::detail::has_rng_formula<FUNCTOR>::value && std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename hydra_thrust::iterator_traits<Iterator>::value_type
>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine =hydra_thrust::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< hydra::detail::has_rng_formula<FUNCTOR>::value && std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename hydra_thrust::iterator_traits<Iterator>::value_type
>::value, void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param policy backend to perform the calculation.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
typename std::enable_if< hydra::detail::is_iterable<Iterable>::value && std::is_convertible<
decltype(*std::declval<Iterable>().begin()), typename FUNCTOR::return_type
>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterable&& iterable, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





/**
 * \ingroup random
 *
 * @brief Fill a range with numbers distributed according a user defined distribution.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, typename Iterable, typename FUNCTOR >
typename std::enable_if< hydra::detail::is_iterable<Iterable>::value && std::is_convertible<
decltype(*std::declval<Iterable>().begin()), typename FUNCTOR::return_type
>::value, void>::type
fill_random(Iterable&& iterable, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula is not implemented for the requested functor
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< !hydra::detail::has_rng_formula<FUNCTOR>::value , void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);




/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula is not implemented for the requested functor
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< !hydra::detail::has_rng_formula<FUNCTOR>::value , void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);




/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
 * @param policy backend to perform the calculation.
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
typename std::enable_if< !std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename std::iterator_traits<Iterator>::value_type
>::value && hydra::detail::has_rng_formula<FUNCTOR>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterator begin, Iterator end, FUNCTOR const& funct, size_t seed=0x254a0afcf7da74a2);




/**
 * \ingroup random
 *
 * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
 * @param begin beginning of the range storing the generated values
 * @param end ending of the range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, typename Iterator, typename FUNCTOR >
typename std::enable_if< !std::is_convertible<
decltype(std::declval<RngFormula<FUNCTOR>>().Generate( std::declval<Engine&>(),  std::declval<FUNCTOR const&>())),
typename std::iterator_traits<Iterator>::value_type
>::value && hydra::detail::has_rng_formula<FUNCTOR>::value, void>::type
fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





/**
 * \ingroup random
 *
 * @brief Fall back function if the argument is not an Iterable or if itis not convertible to the Functor return value
 * @param policy backend to perform the calculation.
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
typename std::enable_if< !hydra::detail::is_iterable<Iterable>::value || !std::is_convertible<
decltype(*std::declval<Iterable>().begin()), typename FUNCTOR::return_type
>::value, void>::type
fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
            Iterable&& iterable, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);




/**
 * \ingroup random
 *
 * @brief Fall back function if the argument is not an Iterable or if itis not convertible to the Functor return value
 * @param iterable range storing the generated values
 * @param functor distribution to be sampled
 */
template< typename Engine = hydra_thrust::default_random_engine, typename Iterable, typename FUNCTOR >
typename std::enable_if< !hydra::detail::is_iterable<Iterable>::value || !std::is_convertible<
decltype(*std::declval<Iterable>().begin()), typename FUNCTOR::return_type
>::value, void>::type
fill_random(Iterable&& iterable, FUNCTOR const& functor, size_t seed=0x254a0afcf7da74a2);





}
#endif /* RANDOMFILL_H_ */

#include <hydra/detail/RandomFill.inl>
