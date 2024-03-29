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

//#include <hydra/Events.h>
#include <hydra/detail/functors/DecayMother.h>
#include <hydra/detail/functors/DecayMothers.h>
#include <hydra/detail/functors/EvalMother.h>
#include <hydra/detail/functors/EvalMothers.h>
#include <hydra/detail/functors/AverageMother.h>
#include <hydra/detail/functors/AverageMothers.h>

#include <hydra/detail/utility/Utility_Tuple.h>

#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/sequence.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/tabulate.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>

namespace hydra {


namespace detail {

	//-------------------------------

	template<size_t N,typename FUNCTOR, typename ...FUNCTORS, typename GRND, typename Iterator>
	inline void launch_evaluator(Iterator begin, Iterator end,
			detail::EvalMother<N, GRND,FUNCTOR, FUNCTORS...> const& evaluator) {

		hydra::thrust::tabulate(begin, end, evaluator);

	}


	template<size_t N, typename FUNCTOR,typename ...FUNCTORS, typename GRND,
	                   typename IteratorMother, typename Iterator>
	inline void launch_evaluator( IteratorMother mbegin, IteratorMother mend, Iterator begin,
			 detail::EvalMothers<N, GRND,FUNCTOR, FUNCTORS...> const& evaluator) {

		size_t nevents = hydra::thrust::distance(mbegin, mend);
		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + nevents;

		hydra::thrust::transform(
				hydra::thrust::make_zip_iterator(first, mbegin),
				hydra::thrust::make_zip_iterator(last, mend),
				begin, evaluator );

	}


	//-------------------------------

	template<size_t N, hydra::detail::Backend BACKEND, typename FUNCTOR, typename GRND, typename Iterator>
	inline StatsPHSP launch_reducer(hydra::detail::BackendPolicy<BACKEND>const& policy,
			Iterator begin, Iterator end,	detail::AverageMother<N, GRND,FUNCTOR> const& evaluator)
	{
		typedef hydra::detail::BackendPolicy<BACKEND> system_t;

		StatsPHSP init = StatsPHSP();

		StatsPHSP result = hydra::thrust::transform_reduce(policy , begin, end,
				evaluator, init,detail::AddStatsPHSP() );

		return result;
	}


	template<size_t N, typename FUNCTOR, typename GRND, typename Iterator>
	inline StatsPHSP launch_reducer(Iterator begin, Iterator end,
			detail::AverageMothers<N, GRND,FUNCTOR> const& evaluator)
	{
		using hydra::thrust::system::detail::generic::select_system;
		typedef typename hydra::thrust::iterator_system<Iterator>::type System;
		System system;

		size_t nevents = hydra::thrust::distance(begin, end);
		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + nevents;

		StatsPHSP init = StatsPHSP();

		StatsPHSP result = hydra::thrust::transform_reduce(select_system(system),
				hydra::thrust::make_zip_iterator(first, begin),
				hydra::thrust::make_zip_iterator(last, end),
				evaluator, init,detail::AddStatsPHSP() );

		return result;
	}

	//-------------------------------

	template<size_t N, typename GRND, typename Iterator>
    inline void launch_decayer(Iterator begin, Iterator end, DecayMother<N, GRND> const& decayer)
	{

		hydra::thrust::tabulate(begin, end, decayer);
		return;
	}

	template<size_t N, typename GRND, typename Iterator, hydra::detail::Backend BACKEND>
	inline void launch_decayer( hydra::detail::BackendPolicy<BACKEND> const& exec_policy ,
			Iterator begin, Iterator end, DecayMother<N, GRND> const& decayer)
	{

		hydra::thrust::tabulate(exec_policy , begin, end, decayer);
		return;

	}

	//-------------------------------

	template<size_t N, typename GRND,	typename IteratorMother, typename IteratorDaughter>
	inline	void launch_decayer(IteratorMother begin_mothers, IteratorMother end_mothers
			, IteratorDaughter begin_daugters, DecayMothers<N, GRND> const& decayer)
	{

		size_t nevents    = hydra::thrust::distance(begin_mothers, end_mothers);

		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + nevents;

		hydra::thrust::transform(first, last, begin_mothers,	begin_daugters, decayer);

		return;
	}


	template<size_t N, typename GRND,	typename IteratorMother, typename IteratorDaughter, hydra::detail::Backend BACKEND>
	inline	void launch_decayer( hydra::detail::BackendPolicy<BACKEND> const& exec_policy,
			  IteratorMother begin_mothers, IteratorMother end_mothers,
			  IteratorDaughter begin_daugters, DecayMothers<N, GRND> const& decayer)
	{
		size_t nevents    = hydra::thrust::distance(begin_mothers, end_mothers);

		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + nevents;

		hydra::thrust::transform(exec_policy, first, last, begin_mothers, begin_daugters, decayer);

		return;
	}



}// namespace detail


}// namespace hydra



#endif /* LAUNCH_DECAYER_INC */
