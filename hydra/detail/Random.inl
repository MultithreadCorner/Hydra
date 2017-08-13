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

#include <thrust/memory.h>

namespace hydra{

template<typename GRND>
template<typename FUNCTOR, typename Iterator>
void Random<GRND>::InverseCDF(FUNCTOR const& invcdf, Iterator begin, Iterator end )
{
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin , end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin,
			detail::RndCDF<GRND, FUNCTOR >(invcdf, fSeed ));


}


template<typename GRND>
template<typename Iterator>
void  Random<GRND>::Gauss(typename Iterator::value_type mean, typename Iterator::value_type sigma,
		Iterator begin, Iterator end )
{
	typedef typename Iterator::value_type value_type;
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin,
			detail::RndGauss<value_type, GRND>(fSeed,  mean, sigma));

}


/**
 * Fill the range (begin, end) with a uniform distribution between [min, max]
 */
template<typename GRND>
template<typename Iterator>
void Random<GRND>::Uniform(typename Iterator::value_type min, typename Iterator::value_type max,
		Iterator begin, Iterator end)
{
	typedef typename Iterator::value_type value_type;
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system),  first, last, begin,
			detail::RndUniform<value_type,GRND>(fSeed+1, min, max));

}

/**
 * Fill the range (begin, end) with an exponential distribution
 */
template<typename GRND>
template<typename Iterator>
void  Random<GRND>::Exp(typename Iterator::value_type tau,  Iterator begin, Iterator end)
{
	typedef typename Iterator::value_type value_type;
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin,
			detail::RndExp<value_type,GRND>(fSeed+2, tau));

}

/**
 * Fill the range (begin, end) with an exponential distribution
 */
template<typename GRND>
template<typename Iterator>
void  Random<GRND>::BreitWigner(typename Iterator::value_type mean, typename Iterator::value_type gamma,
		Iterator begin, Iterator end)
{
	typedef typename Iterator::value_type value_type;
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin,
			detail::RndBreitWigner<value_type,GRND>(fSeed+3,  mean, gamma));

}


template<typename GRND>
template<typename T, typename Iterator, typename FUNCTOR>
Iterator Random<GRND>::Sample(Iterator begin, Iterator end ,
		T min, T max,FUNCTOR const& functor)
{
	typedef T value_type;
	using thrust::system::detail::generic::select_system;
	typedef  typename thrust::iterator_system<Iterator>::type system_t;

    size_t ntrials = thrust::distance( begin, end);

    auto values = thrust::get_temporary_buffer<value_type>(system_t(), ntrials);

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	thrust::transform( system_t(), first, last, begin, values.first.get(),
			detail::RndTrial<value_type,GRND,FUNCTOR,1>(fSeed+4, functor, min, max));

	//get the maximum value
	value_type max_value = *( thrust::max_element(system_t(),values.first, values.first+ values.second) );

	Iterator r = thrust::partition(begin, end, first, detail::RndFlag<value_type,GRND>(fSeed+ntrials, max_value, values.first.get()) );

	// deallocate storage with thrust::return_temporary_buffer
	thrust::return_temporary_buffer(system_t(), values.first);

	return r;
}

template<typename GRND>
template<typename T, typename Iterator, typename FUNCTOR, size_t N >
Iterator Random<GRND>::Sample(Iterator begin, Iterator end ,
		std::array<T,N> const& min,
		std::array<T,N> const& max,
		FUNCTOR const& functor)
{
	typedef T value_type;

	using thrust::system::detail::generic::select_system;
	typedef  typename thrust::iterator_system<Iterator>::type system_t;

    size_t ntrials = thrust::distance( begin, end);

    auto values = thrust::get_temporary_buffer<value_type>(system_t(), ntrials);

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	thrust::transform( system_t(), first, last, begin, values.first.get(),
			detail::RndTrial<value_type, GRND,FUNCTOR,N>(fSeed+4, functor, min, max));

	//get the maximum value
	value_type max_value = *( thrust::max_element(system_t(),values.first, values.first+ values.second) );

	Iterator r = thrust::partition(begin, end, first,
			detail::RndFlag<value_type, GRND>(fSeed+ntrials, max_value, values.first.get()) );
   
	// deallocate storage with thrust::return_temporary_buffer
	thrust::return_temporary_buffer(system_t(), values.first);

	return r;
}



}



#endif /* RANDOM_INL_ */
