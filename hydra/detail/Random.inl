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
void  Random<GRND>::Gauss(GReal_t mean, GReal_t sigma, Iterator begin, Iterator end )
{
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin,
			detail::RndGauss<GRND>(fSeed,  mean, sigma));

}


/**
 * Fill the range (begin, end) with a uniform distribution between [min, max]
 */
template<typename GRND>
template<typename Iterator>
void Random<GRND>::Uniform(GReal_t min, GReal_t max, Iterator begin, Iterator end)
{
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system),  first, last, begin, detail::RndUniform<GRND>(fSeed+1, min, max));

}

/**
 * Fill the range (begin, end) with an exponential distribution
 */
template<typename GRND>
template<typename Iterator>
void  Random<GRND>::Exp(GReal_t tau,  Iterator begin, Iterator end)
{
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin, detail::RndExp<GRND>(fSeed+2, tau));

}

/**
 * Fill the range (begin, end) with an exponential distribution
 */
template<typename GRND>
template<typename Iterator>
void  Random<GRND>::BreitWigner(GReal_t mean, GReal_t gamma, Iterator begin, Iterator end)
{
	using thrust::system::detail::generic::select_system;
	typedef typename thrust::iterator_system<Iterator>::type System;
	System system;

	size_t fNEvents=thrust::distance(begin ,end );

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + fNEvents;

	thrust::transform(select_system(system), first, last, begin, detail::RndBreitWigner<GRND>(fSeed+3,  mean, gamma));

}


template<typename GRND>
template<typename ITERATOR, typename FUNCTOR>
ITERATOR Random<GRND>::Sample(ITERATOR begin, ITERATOR end ,
		GReal_t min, GReal_t max,FUNCTOR const& functor)
{

	using thrust::system::detail::generic::select_system;
	typedef  typename thrust::iterator_system<ITERATOR>::type system_t;

    size_t ntrials = thrust::distance( begin, end);

    auto values = thrust::get_temporary_buffer<GReal_t>(system_t(), ntrials);
    std::cout<< "----------------------------" << std::endl;
     std::cout<< values.second << std::endl;

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	thrust::transform( system_t(), first, last, begin, values.first.get(),
			detail::RndTrial<GRND,FUNCTOR,1>(fSeed+4, functor, min, max));

	//get the maximum value
	GReal_t max_value = *( thrust::max_element(system_t(),values.first, values.first+ values.second) );

	auto r = thrust::partition(begin, end, first, detail::RndFlag<GRND>(fSeed+ntrials, max_value, values.first.get()) );

	// deallocate storage with thrust::return_temporary_buffer
	thrust::return_temporary_buffer(system_t(), values.first);

	return r;
}

template<typename GRND>
template<typename ITERATOR, typename FUNCTOR, size_t N >
ITERATOR Random<GRND>::Sample(ITERATOR begin, ITERATOR end ,
		std::array<GReal_t,N> const& min, std::array<GReal_t,N> const& max,
		FUNCTOR const& functor)
{

	using thrust::system::detail::generic::select_system;
	typedef  typename thrust::iterator_system<ITERATOR>::type system_t;

    size_t ntrials = thrust::distance( begin, end);

    auto values = thrust::get_temporary_buffer<GReal_t>(system_t(), ntrials);
    std::cout<< "----------------------------" << std::endl;
     std::cout<< values.second << std::endl;

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + ntrials;


	//calculate the functor values
	thrust::transform( system_t(), first, last, begin, values.first.get(),
			detail::RndTrial<GRND,FUNCTOR,N>(fSeed+4, functor, min, max));

	//get the maximum value
	GReal_t max_value = *( thrust::max_element(system_t(),values.first, values.first+ values.second) );

	auto r = thrust::partition(begin, end, first, detail::RndFlag<GRND>(fSeed+ntrials, max_value, values.first.get()) );

	// deallocate storage with thrust::return_temporary_buffer
	thrust::return_temporary_buffer(system_t(), values.first);

	return r;
}



}



#endif /* RANDOM_INL_ */
