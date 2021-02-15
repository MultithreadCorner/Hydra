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
 * DalitzSampler.inl
 *
 *  Created on: 20/12/2020
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef DALITZPHASESPACE_INL_
#define DALITZPHASESPACE_INL_

#include <thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/functors/DalitzAverage.h>
#include <hydra/detail/functors/DalitzSampler.h>
#include <hydra/detail/functors/DalitzEvaluator.h>
#include <hydra/detail/functors/StatsPHSP.h>

namespace hydra {



template <typename GRND>
template<typename Iterator, hydra::detail::Backend BACKEND>
inline void  DalitzPhaseSpace<GRND>::Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy,
		Iterator begin, Iterator end){

	Generate(exec_policy.backend, begin, end);

}

template<typename GRND>
template<typename Iterator>
inline void  DalitzPhaseSpace<GRND>::Generate( Iterator begin, Iterator end){

	using hydra_thrust::system::detail::generic::select_system;
	typedef typename hydra_thrust::iterator_system<Iterator>::type System;
	System system{};

	Generate( select_system(system), begin, end );

}

template<typename GRND>
template<typename Iterator, typename FUNCTOR>
inline void  DalitzPhaseSpace<GRND>::Generate( Iterator begin, Iterator end, FUNCTOR const& function){

	using hydra_thrust::system::detail::generic::select_system;
	typedef typename hydra_thrust::iterator_system<Iterator>::type System;
	System system{};

	Generate( select_system(system), begin, end, function);

}


template<typename GRND>
template<typename Iterable, typename FUNCTOR>
inline typename std::enable_if<
hydra::detail::is_iterable<Iterable>::value &&
hydra::detail::random::is_callable<FUNCTOR>::value
>::type
DalitzPhaseSpace<GRND>::Generate( Iterable&& events, FUNCTOR const& function ){

	Generate( std::forward<Iterable>(events).begin(), std::forward<Iterable>(events).end(), function );
}


template<typename GRND>
template<typename Iterable>
inline typename std::enable_if<hydra::detail::is_iterable<Iterable>::value>::type
DalitzPhaseSpace<GRND>::Generate( Iterable&& events ){

	Generate( std::forward<Iterable>(events).begin(), std::forward<Iterable>(events).end() );
}


template<typename GRND>
template< hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
inline void DalitzPhaseSpace<GRND>::Evaluate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy,
		Iterator begin, Iterator end, FUNCTOR const& functor) {

	Evaluate(exec_policy.backend, begin, end, functor);

}

template<typename GRND>
template<typename Iterator, typename FUNCTOR >
inline void DalitzPhaseSpace<GRND>::Evaluate( Iterator begin, Iterator end, FUNCTOR const& functor){

	using hydra_thrust::system::detail::generic::select_system;
	typedef typename hydra_thrust::iterator_system<Iterator>::type System;
	System system{};

	Evaluate(select_system(system), begin, end, functor);

}

template<typename GRND>
template< typename Iterable, typename FUNCTOR >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value>::type
DalitzPhaseSpace<GRND>::Evaluate(Iterable&& result, FUNCTOR const& functor){

	Evaluate( std::forward<Iterable>(result).begin(),
			        std::forward<Iterable>(result).end(), functor);
}

template<typename GRND>
template<typename FUNCTOR, hydra::detail::Backend BACKEND>
inline std::pair<GReal_t, GReal_t>
DalitzPhaseSpace<GRND>::AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy, FUNCTOR const& functor, std::size_t n) {

	return Average( policy, functor, n) ;


}

template<typename GRND>
template<typename Iterator, typename DerivedPolicy>
inline void DalitzPhaseSpace<GRND>::Generate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		Iterator begin, Iterator end){

	size_t nentries = hydra_thrust::distance(begin, end);

	auto first = hydra_thrust::make_counting_iterator(size_t(0));
	auto  last = hydra_thrust::make_counting_iterator(nentries);

	detail::DalitzSampler<GRND> sampler( fMotherMass, fDaughterMass, fSeed);

	hydra_thrust::transform(policy, first, last, begin, sampler);

}

template<typename GRND>
template<typename Iterator, typename Functor, typename DerivedPolicy>
inline void DalitzPhaseSpace<GRND>::Generate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		Iterator begin, Iterator end, Functor const& functor){

	size_t nentries = hydra_thrust::distance(begin, end);

	auto first = hydra_thrust::make_counting_iterator(size_t(0));
	auto  last = hydra_thrust::make_counting_iterator(nentries);

	detail::DalitzSampler<GRND, Functor> sampler( fMotherMass, fDaughterMass,
			fSeed, functor);

	hydra_thrust::transform(policy, first, last, begin, sampler);

}


template<typename GRND>
template<typename Iterator, typename FUNCTOR, typename DerivedPolicy>
inline void DalitzPhaseSpace<GRND>::Evaluate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		Iterator begin, Iterator end, FUNCTOR const& functor){

	size_t nevents = hydra_thrust::distance(begin, end);

	hydra_thrust::counting_iterator<size_t> first(size_t(0));
	hydra_thrust::counting_iterator<size_t> last = first + nevents;

	detail::DalitzEvaluator<FUNCTOR,GRND> sampler( fMotherMass, fDaughterMass, functor, fSeed);

	hydra_thrust::transform(policy, first, last, begin, sampler);


}

template<typename GRND>
template<typename FUNCTOR, typename DerivedPolicy>
inline std::pair<GReal_t, GReal_t>
DalitzPhaseSpace<GRND>::Average(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
		FUNCTOR const& functor, size_t n){

	detail::StatsPHSP init = detail::StatsPHSP();

	hydra_thrust::counting_iterator<std::size_t> first(0);
	hydra_thrust::counting_iterator<std::size_t> last = first + n;

	detail::DalitzAverager<FUNCTOR,GRND> evaluator( fMotherMass, fDaughterMass, functor, fSeed);

	detail::StatsPHSP result = hydra_thrust::transform_reduce(policy, first, last, evaluator,  init, detail::AddStatsPHSP() );

	return std::make_pair(result.fMean, ::sqrt(result.fM2)/result.fW);
}

}  // namespace hydra


#endif /* DALITZPHASESPACE_INL_ */
