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
 * Random.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup random
 */


#ifndef RANDOM_INL_
#define RANDOM_INL_

#include <hydra/detail/external/hydra_thrust/memory.h>

namespace hydra{

template<typename RNG, typename DerivedPolicy, typename IteratorData, typename IteratorWeight>
typename std::enable_if<
	detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
	Range<IteratorData>
>::type
unweight( hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy, IteratorData data_begin, IteratorData data_end, IteratorWeight weights_begin)
{

	typedef typename IteratorWeight::value_type value_type;
	typedef detail::RndFlag<value_type, IteratorWeight, RNG> flagger_type;

	size_t ntrials = hydra_thrust::distance( data_begin, data_end );

	IteratorWeight weights_end = weights_begin + ntrials;

	//get the maximum value
	value_type max_value = *( hydra_thrust::max_element(policy,  weights_begin, weights_end ) );

	// create iterators
	hydra_thrust::counting_iterator<size_t> first(0);
	hydra_thrust::counting_iterator<size_t> last = first + ntrials;

	IteratorData r = hydra_thrust::partition(policy, data_begin, data_end, first,
			flagger_type(ntrials, max_value, weights_begin) );

	return  make_range(begin , r);
}


template<typename RNG, typename IteratorData, typename IteratorWeight, hydra::detail::Backend BACKEND>
typename std::enable_if<
	detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
	Range<IteratorData>
>::type
unweight( detail::BackendPolicy<BACKEND> const& policy, IteratorData data_begin, IteratorData data_end, IteratorWeight weights_begin)
{


	return  unweight<RNG>(policy.backend, data_begin, data_end, weights_begin);
}

template< typename RNG, typename IteratorData, typename IteratorWeight>
typename std::enable_if<
detail::random::is_iterator<IteratorData>::value && detail::random::is_iterator<IteratorWeight>::value,
Range<IteratorData> >::type
unweight(IteratorData data_begin, IteratorData data_end , IteratorData weights_begin)
{
	using hydra_thrust::system::detail::generic::select_system;
	typedef  typename hydra_thrust::iterator_system<IteratorData>::type   system_data_type;
	typedef  typename hydra_thrust::iterator_system<IteratorWeight>::type system_weight_type;

	typedef  typename hydra_thrust::detail::remove_reference<
			 decltype(select_system(std::declval<system_data_type>(), std::declval<system_weight_type>()))
			 >::type common_system_type;

	return unweight<RNG>(common_system_type(), data_begin, data_end , weights_begin);

}

template<typename RNG, typename IterableData, typename IterableWeight, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
Range< decltype(std::declval<IterableData>().begin())> >::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy,  IterableData&& data, IterableWeight&& weights)
{
	return unweight<RNG>( policy, std::forward<IterableData>(data).begin(), std::forward<IterableData>(data).end(),
			std::forward<IterableWeight>(weights).begin());
}

template< typename RNG, typename IterableData, typename IterableWeight>
typename std::enable_if<detail::random::is_iterable<IterableData>::value && detail::random::is_iterable<IterableWeight>::value,
Range< decltype(std::declval<IterableData>().begin())>>::type
unweight( IterableData&&  data, IterableWeight&&  weights)
{

	using hydra_thrust::system::detail::generic::select_system;
	typedef  typename hydra_thrust::iterator_system<decltype(data.begin())>::type   system_data_type;
	typedef  typename hydra_thrust::iterator_system<decltype(weights.begin())>::type system_weight_type;

	typedef  typename hydra_thrust::detail::remove_reference<
			 decltype(select_system(std::declval<system_data_type>(), std::declval<system_weight_type>()))
			 >::type common_system_type;

	return unweight<RNG>(common_system_type(), std::forward<IterableData>(data).begin(), std::forward<IterableData>(data).end(),
			std::forward<IterableWeight>(weights).begin());

}


template< typename RNG, typename Functor, typename Iterator, typename DerivedPolicy>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight(hydra_thrust::detail::execution_policy_base<DerivedPolicy>  const& policy, Iterator begin, Iterator end, Functor const& functor)
{

	typedef typename Functor::return_type value_type;

	typedef hydra_thrust::pointer<value_type,
			typename hydra_thrust::detail::execution_policy_base<DerivedPolicy>::tag_type> pointer_type;

	typedef detail::RndFlag<value_type,pointer_type, RNG > flagger_type;

    size_t ntrials = hydra_thrust::distance( begin, end);

    auto values = hydra_thrust::get_temporary_buffer<value_type>(policy, ntrials);

	// create iterators
	hydra_thrust::counting_iterator<size_t> first(0);
	hydra_thrust::counting_iterator<size_t> last = first + ntrials;

	//calculate the functor values
	hydra_thrust::transform(policy, begin, end, values.first, functor);

	//get the maximum value
	value_type max_value = *( hydra_thrust::max_element(policy,  values.first, values.first + values.second) );

	Iterator r = hydra_thrust::partition(policy, begin, end, first, flagger_type(ntrials, max_value, values.first ) );

	// deallocate storage with hydra_thrust::return_temporary_buffer
	hydra_thrust::return_temporary_buffer(policy, values.first);

	return  make_range(begin , r);

}


template< typename RNG, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( detail::BackendPolicy<BACKEND> const& policy, Iterator begin, Iterator end, Functor const& functor)
{

	return  unweight<RNG>(policy.backend, begin, end, functor );

}

template<typename RNG, typename Functor, typename Iterator>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
unweight( Iterator begin, Iterator end, Functor const& functor)
{
	typedef  typename hydra_thrust::iterator_system< Iterator>::type   system_data_type;
	return unweight<RNG>( system_data_type(), begin, end, functor );
}


template<typename RNG, typename Functor, typename Iterable, hydra::detail::Backend  BACKEND>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
	Range< decltype(std::declval<Iterable>().begin())>
>::type
unweight( hydra::detail::BackendPolicy<BACKEND> const& policy, Iterable&& iterable, Functor const& functor)
{

	return hydra::unweight<RNG>(policy, std::forward<Iterable>(iterable).begin(),
			std::forward<Iterable>(iterable).end(), functor);
}

template<typename RNG, typename Functor, typename Iterable>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
	Range< decltype(std::declval<Iterable>().begin())>
>::type
unweight( Iterable&& iterable, Functor const& functor)
{

	return hydra::unweight<RNG>(std::forward<Iterable>(iterable).begin(),
				std::forward<Iterable>(iterable).end(), functor);
}

//---------------------------------------------------------------
//
//---------------------------------------------------------------
template<typename RNG, typename DerivedPolicy, typename Functor, typename Iterator>
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator> >::type
sample(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy, Iterator begin, Iterator end, double min, double max,
				Functor const& functor, size_t seed)
{
	typedef double value_type;

	typedef detail::RndTrial<value_type,RNG,Functor,1> sampler_type;

	typedef hydra_thrust::pointer<value_type, DerivedPolicy> pointer_type;

	typedef detail::RndFlag<value_type, pointer_type,RNG> flagger_type;

    size_t ntrials = hydra_thrust::distance( begin, end);

    auto values = hydra_thrust::get_temporary_buffer<value_type>( policy, ntrials);

	// create iterators
	hydra_thrust::counting_iterator<size_t> first(0);
	hydra_thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	hydra_thrust::transform(policy, first, last, begin, values.first.get(),
			sampler_type(seed, functor, min, max));

	//get the maximum value
	value_type max_value = *( hydra_thrust::max_element(policy,values.first, values.first+ values.second) );

	Iterator r = hydra_thrust::partition(policy, begin, end, first,
			flagger_type(seed+ntrials, max_value, values.first) );

	// deallocate storage with hydra_thrust::return_temporary_buffer
	hydra_thrust::return_temporary_buffer( policy, values.first);

	return make_range(begin , r);
}

template<typename RNG, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end, double min, double max,
		Functor const& functor, size_t seed)
{
	return	sample<RNG>( policy.backend, begin, end, min, max, functor, seed );

}

template<typename RNG, typename Functor, typename Iterator>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(Iterator begin, Iterator end , double min, double max,
		Functor const& functor, size_t seed)
{
	typedef  typename hydra_thrust::iterator_system<Iterator>::type   system_type;

	return	sample<RNG>(system_type(), begin, end, min, max, functor, seed );

}

template<typename RNG, typename Functor, typename Iterable>
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
Range< decltype(std::declval<Iterable>().begin())>>::type
sample(Iterable&& output, double min, double max, Functor const& functor, size_t seed)
{
	return	sample<RNG>(std::forward<Iterable>(output).begin(), std::forward<Iterable>(output).end(),
			min, max, functor, seed );
}


template<typename RNG, typename DerivedPolicy, typename Functor, typename Iterator, size_t N >
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
sample( hydra_thrust::detail::execution_policy_base<DerivedPolicy>  const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N> const& min, std::array<double,N> const& max,
		Functor const& functor, size_t seed)
{
	typedef double value_type;

	typedef hydra_thrust::pointer<value_type,  DerivedPolicy> pointer_type;

	typedef detail::RndFlag<value_type, pointer_type, RNG> flagger_type;

	typedef detail::RndTrial<value_type, RNG, Functor, N> sampler_type;

    size_t ntrials = hydra_thrust::distance( begin, end);


    auto values = hydra_thrust::get_temporary_buffer<value_type>(policy, ntrials);

	// create iterators
	hydra_thrust::counting_iterator<size_t> first(0);
	hydra_thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	hydra_thrust::transform(policy, first, last, begin, values.first,
			sampler_type(seed, functor, min, max));

	//get the maximum value
	value_type max_value = *( hydra_thrust::max_element(policy,values.first, values.first+ values.second) );

	Iterator r = hydra_thrust::partition(policy, begin, end, first,
			flagger_type(seed+ntrials, max_value, values.first) );

	// deallocate storage with hydra_thrust::return_temporary_buffer
	hydra_thrust::return_temporary_buffer(policy, values.first);

	return  make_range(begin , r);
}

template<typename RNG, typename Functor, typename Iterator, hydra::detail::Backend  BACKEND, size_t N >
typename std::enable_if<
detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
Range<Iterator> >::type
sample(hydra::detail::BackendPolicy<BACKEND> const& policy,
		Iterator begin, Iterator end ,
		std::array<double,N>const& min,	std::array<double,N>const& max,
		Functor const& functor, size_t seed)
{

	return sample<RNG>(policy.backend, begin, end, min, max, functor, seed );
}

template<typename RNG, typename Functor, typename Iterator, size_t N >
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterator<Iterator>::value,
	Range<Iterator>
>::type
sample(Iterator begin, Iterator end , std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, size_t seed)
{
	typedef  typename hydra_thrust::iterator_system<Iterator>::type   system_type;

	return	sample<RNG>(system_type(), begin, end, min, max, functor, seed );

}

template<typename RNG, typename Functor, typename Iterable, size_t N >
typename std::enable_if<
	detail::random::is_callable<Functor>::value && detail::random::is_iterable<Iterable>::value ,
	Range< decltype(std::declval<Iterable>().begin())>
>::type
sample(Iterable&& output , std::array<double,N>const& min, std::array<double,N>const& max,
		Functor const& functor, size_t seed)
{
	return	sample<RNG>(std::forward<Iterable>(output).begin(), std::forward<Iterable>(output).end(),
			min, max, functor, seed );
}



}//namespace hydra



#endif /* RANDOM_INL_ */
