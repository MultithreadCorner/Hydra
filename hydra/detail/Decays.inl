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
 * Decays.inl
 *
 *  Created on: 05/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DECAYS_INL_
#define DECAYS_INL_

/*
 template<size_t N, detail::Backend BACKEND>
 Decays<N, detail::BackendPolicy<BACKEND> >::

 */
namespace hydra {

namespace detail {

template<size_t N, typename Functor, typename ArgType>
struct EvalOnDaugthers: public HYDRA_EXTERNAL_NS::thrust::unary_function<
		ArgType, GReal_t> {
	EvalOnDaugthers(Functor const& functor) :
			fFunctor(functor) {	}

	__hydra_host__  __hydra_device__
	EvalOnDaugthers(EvalOnDaugthers<N, Functor, ArgType> const&other) :
			fFunctor(other.fFunctor) {
	}

	//template<typename T>
	__hydra_host__  __hydra_device__
	GReal_t operator()(ArgType& value) {

		typename detail::tuple_type<N,Vector4R>::type particles= detail::dropFirst(value);
		return HYDRA_EXTERNAL_NS::thrust::get<0>(value)
				* (fFunctor( particles));

	}

	Functor fFunctor;
};

template<size_t N>
struct FlagDaugthers: public HYDRA_EXTERNAL_NS::thrust::unary_function<size_t,
		bool> {

	FlagDaugthers(GReal_t max, GReal_t* iterator) :
			fVals(iterator), fMax(max) {
	}

	__hydra_host__  __hydra_device__
	FlagDaugthers(FlagDaugthers<N> const&other) :
			fVals(other.fVals), fMax(other.fMax) {
	}
	__hydra_host__  __hydra_device__
	bool operator()(size_t idx) {
		HYDRA_EXTERNAL_NS::thrust::default_random_engine randEng(159753654);
		randEng.discard(idx);
		HYDRA_EXTERNAL_NS::thrust::uniform_real_distribution<GReal_t> uniDist(
				0.0, 1.0);

		return fVals[idx] / fMax > uniDist(randEng);

	}

	GReal_t* fVals;
	GReal_t fMax;
};

}  // namespace detail




template<size_t N, detail::Backend BACKEND>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::Unweight(GUInt_t scale) {
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<
			typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator>::type system_t;

	//number of events to trial
	size_t ntrials = this->size();

	//create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > last = first + ntrials;

	//get the maximum value
	GReal_t max_value = *(HYDRA_EXTERNAL_NS::thrust::max_element(fWeights.begin(), fWeights.end()));

	//raw pointer to weights
	GReal_t* weights_ptr = HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(fWeights.data());

	//says if an event passed or not
	detail::FlagDaugthers<N> predicate(scale * max_value, weights_ptr);

	//re-sort the container to build up un-weighted sample
	auto middle = HYDRA_EXTERNAL_NS::thrust::stable_partition(this->begin(),
			this->end(), first, predicate);

	//done!
	return HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle);
}


template<size_t N, detail::Backend BACKEND>
template<typename FUNCTOR>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::Unweight(
		FUNCTOR const& functor, GUInt_t scale) {

	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<
			typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator>::type system_t;

	//number of events to trial
	size_t ntrials = this->size();

	auto values = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer <GReal_t> (system_t(), ntrials);

	//create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > last = first + ntrials;

	detail::EvalOnDaugthers<N, FUNCTOR,
		typename Decays<N, detail::BackendPolicy<BACKEND> >::value_type> predicate1(functor);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->begin(), predicate1),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->end(),predicate1),
			values.first);

	GReal_t max_value = *(HYDRA_EXTERNAL_NS::thrust::max_element(values.first,
			values.first + values.second));

	//says if an event passed or not
	detail::FlagDaugthers<N> predicate2(scale * max_value, values.first.get());

	//weight 1.0 all events
	HYDRA_EXTERNAL_NS::thrust::constant_iterator<GReal_t> iter_weight(1.0);

	//re-sort the container to build up un-weighted sample

	auto middle = HYDRA_EXTERNAL_NS::thrust::stable_partition(this->begin(), this->end(), first, predicate2);

	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), values.first);
	//done!
	return (size_t) HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle);

}

template<size_t N, detail::Backend BACKEND>
template<typename FUNCTOR>
void Decays<N, detail::BackendPolicy<BACKEND> >::Reweight(
		FUNCTOR const& functor) {

	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<
			typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator>::type system_t;

	detail::EvalOnDaugthers<N, FUNCTOR,
			typename Decays<N, detail::BackendPolicy<BACKEND> >::value_type> predicate1(
			functor);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->begin(),
					predicate1),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->end(),
					predicate1), fWeights.begin());

	return;

}

//=======================

template<size_t N1, hydra::detail::Backend BACKEND1, size_t N2,
		hydra::detail::Backend BACKEND2>
bool operator==(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
		const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs) {

	bool is_same_type = (N1 == N2)
			&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<
					hydra::detail::BackendPolicy<BACKEND1>,
					hydra::detail::BackendPolicy<BACKEND2> >::value
			&& lhs.size() == rhs.size();
	bool result = 1;

	auto comp = []__hydra_host__ __hydra_device__(HYDRA_EXTERNAL_NS::thrust::tuple<
			typename Decays<N1, hydra::detail::BackendPolicy<BACKEND1>>::value_type,
			typename Decays<N2, hydra::detail::BackendPolicy<BACKEND2>>::value_type> const& values) {
		return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	if (is_same_type) {
		result = HYDRA_EXTERNAL_NS::thrust::all_of(
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(),
						rhs.begin()),
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(),
						rhs.end()), comp);
	}
	return result && is_same_type;

}

template<size_t N1, hydra::detail::Backend BACKEND1, size_t N2,
		hydra::detail::Backend BACKEND2>
bool operator!=(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
		const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs) {

	bool is_same_type = (N1 == N2)
			&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<
					hydra::detail::BackendPolicy<BACKEND1>,
					hydra::detail::BackendPolicy<BACKEND2> >::value
			&& lhs.size() == rhs.size();
	bool result = 1;

	auto comp = []__hydra_host__ __hydra_device__(HYDRA_EXTERNAL_NS::thrust::tuple<
			typename Decays<N1, hydra::detail::BackendPolicy<BACKEND1>>::value_type,
			typename Decays<N2, hydra::detail::BackendPolicy<BACKEND2>>::value_type> const& values) {
		return (HYDRA_EXTERNAL_NS::thrust::get<0>(values) == HYDRA_EXTERNAL_NS::thrust::get<1>(values));

	};

	if (is_same_type) {
		result = HYDRA_EXTERNAL_NS::thrust::all_of(
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(),
						rhs.begin()),
				HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(),
						rhs.end()), comp);
	}
	return (!result) && is_same_type;
}

}  // namespace hydra

#endif /* DECAYS_INL_ */
