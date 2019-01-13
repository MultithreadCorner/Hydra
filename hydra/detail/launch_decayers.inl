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
 * launch_decayer
 *
 *  Created on: Jun 16, 2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */


#ifndef LAUNCH_DECAYER_INC
#define LAUNCH_DECAYER_INC

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
//#include <hydra/Events.h>
#include <hydra/detail/functors/DecayMother.h>
#include <hydra/detail/functors/DecayMothers.h>
#include <hydra/detail/functors/EvalMother.h>
#include <hydra/detail/functors/EvalMothers.h>
#include <hydra/detail/functors/AverageMother.h>
#include <hydra/detail/functors/AverageMothers.h>

#include <hydra/detail/utility/Utility_Tuple.h>

#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/sequence.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/transform_reduce.h>

namespace hydra {


namespace detail {

	//-------------------------------

	template<size_t N,typename FUNCTOR, typename ...FUNCTORS, typename GRND, typename Iterator>
	inline void launch_evaluator(Iterator begin, Iterator end,
			detail::EvalMother<N, GRND,FUNCTOR, FUNCTORS...> const& evaluator) {

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
				HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
				HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		HYDRA_EXTERNAL_NS::thrust::transform( first, last, begin, evaluator);

	}


	template<size_t N, typename FUNCTOR,typename ...FUNCTORS, typename GRND,
	                   typename IteratorMother, typename Iterator>
	inline void launch_evaluator( IteratorMother mbegin, IteratorMother mend, Iterator begin,
			 detail::EvalMothers<N, GRND,FUNCTOR, FUNCTORS...> const& evaluator) {

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(mbegin, mend);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		HYDRA_EXTERNAL_NS::thrust::transform( HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(first, mbegin),
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(last, mend), begin, evaluator );

	}


	//-------------------------------

	template<size_t N, hydra::detail::Backend BACKEND, typename FUNCTOR, typename GRND, typename Iterator>
	inline StatsPHSP launch_reducer(hydra::detail::BackendPolicy<BACKEND>const& policy,
			Iterator begin, Iterator end,	detail::AverageMother<N, GRND,FUNCTOR> const& evaluator)
	{
		typedef hydra::detail::BackendPolicy<BACKEND> system_t;

		StatsPHSP init = StatsPHSP();

		StatsPHSP result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(policy , begin, end,
				evaluator, init,detail::AddStatsPHSP() );

		return result;
	}


	template<size_t N, typename FUNCTOR, typename GRND, typename Iterator>
	inline StatsPHSP launch_reducer(Iterator begin, Iterator end,
			detail::AverageMothers<N, GRND,FUNCTOR> const& evaluator)
	{
		using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type System;
		System system;

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		StatsPHSP init = StatsPHSP();

		StatsPHSP result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(select_system(system),
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(first, begin),
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(last, end),
				evaluator, init,detail::AddStatsPHSP() );

		return result;
	}

	//-------------------------------

	template<size_t N, typename GRND, typename Iterator>
    inline void launch_decayer(Iterator begin, Iterator end, DecayMother<N, GRND> const& decayer)
	{

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		auto begin_weights = HYDRA_EXTERNAL_NS::thrust::get<0>(begin.get_iterator_tuple());

		auto begin_temp = hydra::detail::dropFirst( begin.get_iterator_tuple() );

		auto begin_particles = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_temp);

		HYDRA_EXTERNAL_NS::thrust::transform(first, last, begin_particles, begin_weights, decayer);

		return;
	}

	template<size_t N, typename GRND, typename Iterator, hydra::detail::Backend BACKEND>
	inline void launch_decayer( hydra::detail::BackendPolicy<BACKEND> const& exec_policy ,Iterator begin, Iterator end, DecayMother<N, GRND> const& decayer)
	{

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		auto begin_weights = HYDRA_EXTERNAL_NS::thrust::get<0>(begin.get_iterator_tuple());

		auto begin_temp = hydra::detail::dropFirst( begin.get_iterator_tuple() );

		auto begin_particles = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_temp);

		HYDRA_EXTERNAL_NS::thrust::transform(exec_policy ,first, last, begin_particles, begin_weights, decayer);

		return;
	}

	//-------------------------------

	template<size_t N, typename GRND,	typename IteratorMother, typename IteratorDaughter>
	inline	void launch_decayer(IteratorMother begin, IteratorMother end
			, IteratorDaughter begin_daugters, DecayMothers<N, GRND> const& decayer)
	{

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		auto begin_weights = HYDRA_EXTERNAL_NS::thrust::get<0>(begin_daugters.get_iterator_tuple());

		auto begin_temp = hydra::detail::changeFirst(  begin, begin_daugters.get_iterator_tuple() );

		auto begin_particles = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_temp);

		HYDRA_EXTERNAL_NS::thrust::transform(first, last, begin_particles, begin_weights, decayer);

		return;
	}


	template<size_t N, typename GRND,	typename IteratorMother, typename IteratorDaughter, hydra::detail::Backend BACKEND>
	inline	void launch_decayer( hydra::detail::BackendPolicy<BACKEND> const& exec_policy , IteratorMother begin, IteratorMother end
			, IteratorDaughter begin_daugters, DecayMothers<N, GRND> const& decayer)
	{

		size_t nevents = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + nevents;

		auto begin_weights = HYDRA_EXTERNAL_NS::thrust::get<0>(begin_daugters.get_iterator_tuple());

		auto begin_temp = hydra::detail::changeFirst(  begin, begin_daugters.get_iterator_tuple() );

		auto begin_particles = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_temp);

		HYDRA_EXTERNAL_NS::thrust::transform(exec_policy , first, last, begin_particles, begin_weights, decayer);

		return;
	}



}// namespace detail


}// namespace hydra



#endif /* LAUNCH_DECAYER_INC */
