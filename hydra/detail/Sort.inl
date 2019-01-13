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
 * Sort.inl
 *
 *  Created on: 18/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SORT_INL_
#define SORT_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <utility>
#include <hydra/detail/external/thrust/sort.h>
#include <hydra/Range.h>


namespace hydra {

template<typename Iterable, typename Iterator=decltype(std::declval<Iterable>().begin())>
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
Range<decltype(std::declval<Iterable&>().begin())>>::type
sort(Iterable& iterable){

	HYDRA_EXTERNAL_NS::thrust::sort(iterable.begin(), iterable.end() );

	return make_range(iterable.begin(), iterable.end());
}

template<typename Iterable, typename Functor,
typename Iterator=decltype(std::declval<Iterable>().begin())>
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
Range<decltype(std::declval<Iterable&>().begin())>>::type
sort(Iterable& iterable, Functor const& comparator){

	HYDRA_EXTERNAL_NS::thrust::sort(iterable.begin(), iterable.end(), comparator);

	return make_range(iterable.begin(), iterable.end());
}

template<typename Iterable,typename Iterable_Key,
typename Iterator=decltype(std::declval<Iterable>().begin()),
typename Iterator_Key=decltype(std::declval<Iterable_Key>().begin()),
typename Value_Key=decltype(*std::declval<Iterator_Key>().begin()) >
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
Range<decltype(std::declval<Iterable&>().begin())>>::type
sort_by_key(Iterable& iterable, Iterable_Key& keys){

	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator_Key>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(system1, system2 ))>::type common_system_t;

	HYDRA_EXTERNAL_NS::thrust::sort_by_key(iterable.begin(), iterable.end(), keys.begin());

	return make_range(iterable.begin(), iterable.end());
}

template<typename Iterable,typename Iterator_Key, typename Functor,
typename Iterator=decltype(std::declval<Iterable>().begin()),
typename Value_Key=decltype(*std::declval<Range<Iterator_Key,Functor>>().begin()) >
typename std::enable_if<hydra::detail::is_iterable<Iterable>::value,
Range<decltype(std::declval<Iterable&>().begin())>>::type
sort_by_key(Iterable& iterable, Range<Iterator_Key,Functor>&& keys){

	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator_Key>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(system1, system2 ))>::type common_system_t;

	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<Value_Key>(common_system_t(), iterable.size());
	HYDRA_EXTERNAL_NS::thrust::copy(common_system_t(), keys.begin(), keys.end(), key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort_by_key(iterable.begin(), iterable.end(), key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

	return make_range(iterable.begin(), iterable.end());
}

}  // namespace hydra


#endif /* SORT_INL_ */
