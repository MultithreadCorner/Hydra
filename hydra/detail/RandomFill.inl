/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * RandomFill.inl
 *
 *  Created on: 26/02/2020
 *      Author: Davide Brundu
 *
 *  Modified on: 28/07/2020
 *       Author: Antonio Augusto Alves Junior
 *          Log: Moved detail::Sampler class DistributionSampler.h
 */




#ifndef RANDOMFILL_INL_
#define RANDOMFILL_INL_


namespace hydra{

    /**
     * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
     * @param policy backend to perform the calculation.
     * @param begin beginning of the range storing the generated values
     * @param end ending of the range storing the generated values
     * @param functor distribution to be sampled
     */
    template <typename Engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR>
    requires hydra::detail::HasRngFormula<FUNCTOR> &&
             hydra::detail::IsRngFormulaConvertible<FUNCTOR, Engine, Iterator>
    void fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
                Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {
        using hydra::thrust::system::detail::generic::select_system;
        typedef typename hydra::thrust::iterator_system<Iterator>::type system_t;
        typedef typename hydra::detail::BackendPolicy<BACKEND>::execution_policy_type policy_t;
        system_t system;
        policy_t _policy;

        typedef  typename hydra::thrust::detail::remove_reference<
                    decltype(select_system( system, _policy ))>::type common_system_type;

        hydra::thrust::tabulate( common_system_type(), begin, end, detail::Sampler<FUNCTOR,Engine>(functor, seed, rng_jump) );

    }

    /**
     * @brief Fill a range with numbers distributed according a user defined distribution using a RNG analytical formula
     * @param begin beginning of the range storing the generated values
     * @param end ending of the range storing the generated values
     * @param functor distribution to be sampled
     */
    template< typename Engine, typename Iterator, typename FUNCTOR >
    requires hydra::detail::HasRngFormula<FUNCTOR> &&
             hydra::detail::IsRngFormulaConvertible<FUNCTOR, Engine, Iterator>
    void fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {
        using hydra::thrust::system::detail::generic::select_system;
        typedef typename hydra::thrust::iterator_system<Iterator>::type system_t;
        system_t system;

        hydra::thrust::tabulate(select_system(system), begin, end, detail::Sampler<FUNCTOR,Engine>(functor, seed, rng_jump) );
    }

    /**
     * @brief Fill a range with numbers distributed according a user defined distribution.
     * @param policy backend to perform the calculation.
     * @param iterable range storing the generated values
     * @param functor distribution to be sampled
     */
    template< typename Engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
    typename std::enable_if< detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value, void>::type
    fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
                Iterable&& iterable, FUNCTOR const& functor, size_t seed, size_t rng_jump){

        fill_random(policy, std::forward<Iterable>(iterable).begin(),
                    std::forward<Iterable>(iterable).end(), functor, seed, rng_jump);

    }

    /**
     * @brief Fill a range with numbers distributed according a user defined distribution.
     * @param iterable range storing the generated values
     * @param functor distribution to be sampled
     */
    template< typename Engine, typename Iterable, typename FUNCTOR >
    typename std::enable_if< detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value,
     void>::type
    fill_random(Iterable&& iterable, FUNCTOR const& functor, size_t seed, size_t rng_jump){

        fill_random(std::forward<Iterable>(iterable).begin(),
                    std::forward<Iterable>(iterable).end(), functor, seed, rng_jump);

    }

    /**
     * @brief Fall back function if RngFormula is not implemented for the requested functor
     */
    template< typename Engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
    requires (!hydra::detail::HasRngFormula<FUNCTOR>)
    void fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
                Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {

        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " The functor passed to hydra::fill_random() function has no \n "
                " rng analytical formula implemented. Please use hydra::sample \n"
                " which will deploy different strategy \n.")

    }

    /**
     * @brief Fall back function if RngFormula is not implemented for the requested functor
     */
    template< typename Engine, typename Iterator, typename FUNCTOR >
    requires (!hydra::detail::HasRngFormula<FUNCTOR>)
    void fill_random(Iterator begin, Iterator end, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {

        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " The functor passed to hydra::fill_random() function has no \n "
                " rng analytical formula implemented. Please use hydra::sample \n"
                " which will deploy different strategy ")

    }

    /**
     * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
     */
    template< typename Engine, hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
    requires hydra::detail::HasRngFormula<FUNCTOR> &&
             hydra::detail::NotConvertibleToIteratorValue<FUNCTOR, Engine, Iterator>
    void fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
                Iterator begin, Iterator end, FUNCTOR const& funct, size_t seed, size_t rng_jump)
    {
        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " Generated objects can't be stored in this container. " )
    }

    /**
     * @brief Fall back function if RngFormula::Generate() return value is not convertible to functor return value
     */
    template< typename Engine, typename Iterator, typename FUNCTOR >
    requires hydra::detail::HasRngFormula<FUNCTOR> &&
             hydra::detail::NotConvertibleToIteratorValue<FUNCTOR, Engine, Iterator>
    void fill_random(Iterator begin, Iterator end, FUNCTOR const& funct, size_t seed, size_t rng_jump)
    {
        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " Generated objects can't be stored in this container. " )
    }

    /**
     * @brief Fall back function if the argument is not an Iterable or if it is not convertible to the Functor return value
     */
    template< typename Engine, hydra::detail::Backend BACKEND, typename Iterable, typename FUNCTOR >
    typename std::enable_if< !(detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value), void>::type
    fill_random(hydra::detail::BackendPolicy<BACKEND> const& policy,
                Iterable&& iterable, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {
        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " Generated objects can't be stored in this container. \n"
                " The container is not iterable, or it is not convertible \n"
                " from/to the functor return value. " )
    }

    /**
     * @brief Fall back function if the argument is not an Iterable or if it is not convertible to the Functor return value
     */
    template< typename Engine, typename Iterable, typename FUNCTOR >
    typename std::enable_if<!detail::random::is_matching_iterable<Engine, FUNCTOR, Iterable>::value, void>::type
    fill_random(Iterable&& iterable, FUNCTOR const& functor, size_t seed, size_t rng_jump)
    {
        HYDRA_STATIC_ASSERT( int(std::is_class<Engine>::value) ==-1 ,
                " Generated objects can't be stored in this container. \n"
                " The container is not iterable, or it is not convertible \n"
                " from/to the functor return value. " )
    }



}//namespace hydra



#endif /* RANDOMFILL_INL_ */
