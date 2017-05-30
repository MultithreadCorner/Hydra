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
template<hydra::detail::Backend BACKEND, typename FUNCTOR, size_t N >
auto Random<GRND>::Sample(hydra::detail::BackendPolicy<BACKEND>const&, FUNCTOR const& functor, std::array<GReal_t,N> min,
		std::array<GReal_t,N> max, size_t trials)
->	typename hydra::detail::BackendPolicy<BACKEND>::template container<
typename  detail::tuple_type<N,GReal_t>::type>
{
	typedef typename detail::tuple_type<N,GReal_t>::type tuple_t;

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;


    typedef typename system_t::template container<tuple_t> vector_tuple_t;
    typedef typename system_t::template container<GReal_t> vector_real_t;
    typedef typename system_t::template container<GBool_t> vector_bool_t;

	vector_real_t values(trials);
	vector_bool_t flags(trials);
	vector_tuple_t points(trials);

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + trials;


	thrust::transform( system_t(), first, last, points.begin(), values.begin(),
			detail::RndTrial<GRND,FUNCTOR,N>(fSeed+4, functor, min, max));

	GReal_t max_value = *( thrust::max_element( system_t(),values.begin(), values.end()) );

	thrust::transform( system_t(),first, last, values.begin(), flags.begin(), detail::RndFlag<GRND>(fSeed+trials, max_value) );

	size_t count = thrust::count( system_t(), flags.begin(), flags.end(), kTrue);

	vector_tuple_t result(count);

	thrust::copy_if( system_t(), points.begin(), points.end(), flags.begin(), result.begin(), thrust::identity<GBool_t>());

	return result;
}

template<typename GRND>
template<hydra::detail::Backend BACKEND, typename FUNCTOR, size_t N >
void Random<GRND>::Sample(hydra::detail::BackendPolicy<BACKEND>const&, FUNCTOR const& functor, std::array<GReal_t,N> min, std::array<GReal_t,N> max,
		PointVector<Point<GReal_t, N, false, false>, BACKEND >& result,
		size_t trials)
{
	typedef typename detail::tuple_type<N,GReal_t>::type tuple_t;

    typedef hydra::detail::BackendPolicy<BACKEND> system_t;


    typedef typename system_t::template container<tuple_t> vector_tuple_t;
    typedef typename system_t::template container<GReal_t> vector_real_t;
    typedef typename system_t::template container<GBool_t> vector_bool_t;


	vector_real_t values(trials);
	vector_bool_t flags(trials);
	vector_tuple_t points(trials);

	// create iterators
	thrust::counting_iterator<size_t> first(0);
	thrust::counting_iterator<size_t> last = first + trials;


	thrust::transform(system_t(),  first, last, points.begin(), values.begin(),
			detail::RndTrial<GRND,FUNCTOR,N>(fSeed+4, functor, min, max));

	GReal_t max_value = *( thrust::max_element(system_t(), values.begin(), values.end()) );

	thrust::transform(system_t(), first, last, values.begin(), flags.begin(), detail::RndFlag<GRND>(fSeed+trials, max_value) );

	size_t count = thrust::count(system_t(), flags.begin(), flags.end(), kTrue);

	result.GetPoints().resize(count);

	thrust::copy_if(system_t(), points.begin(), points.end(), flags.begin(), result.begin(), thrust::identity<GBool_t>());

	}

}



#endif /* RANDOM_INL_ */
