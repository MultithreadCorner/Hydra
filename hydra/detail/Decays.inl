/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
			fFunctor(functor) {
	}

	__host__  __device__
	EvalOnDaugthers(EvalOnDaugthers<N, Functor, ArgType> const&other) :
			fFunctor(other.fFunctor) {
	}

	template<typename T>
	__host__  __device__
	GReal_t operator()(T& value) {
		auto particles = detail::dropFirst(value);
		Vector4R Particles[N];
		hydra::detail::assignTupleToArray(particles, Particles);
		return hydra::get<0>(value)
				* (fFunctor((unsigned int) N, &Particles[0]));
	}

	Functor fFunctor;
};

template<size_t N>
struct FlagDaugthers: public HYDRA_EXTERNAL_NS::thrust::unary_function<size_t,
		bool> {
	FlagDaugthers(GReal_t max, GReal_t* iterator) :
			fVals(iterator), fMax(max) {
	}

	__host__  __device__
	FlagDaugthers(FlagDaugthers<N> const&other) :
			fVals(other.fVals), fMax(other.fMax) {
	}
	__host__  __device__
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
void Decays<N, detail::BackendPolicy<BACKEND> >::pop_back() {
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].pop_back();
	this->fWeights.pop_back();

}


template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::resize(size_t size) {
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].resize(size);
	this->fWeights.resize(size);
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::clear() {
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].clear();
	this->fWeights.clear();

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::shrink_to_fit() {
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].shrink_to_fit();
	this->fWeights.shrink_to_fit();

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::reserve(size_t size) {
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].reserve(size);
	this->fWeights.reserve(size);

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		size_type n, const value_type &x) {
	size_t pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
	auto particles = detail::dropFirst(x);
	std::array<Vector4R, N> arr { };
	detail::tupleToArray(particles, arr);

	for (size_t i = 0; i < N; i++)
		this->fDecays[i].insert(this->fDecays[i].begin() + pos, n, arr[i]);

	this->fWeights.insert(fWeights.begin() + pos, n, get<0>(x));
}

template<size_t N, detail::Backend BACKEND>
template<typename InputIterator>
void Decays<N, detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		InputIterator first, InputIterator last) {
	size_t pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
	do_insert(pos, first, last);

	this->fWeights.insert(fWeights.begin() + pos,
			get<0>(first.get_iterator_tuple()),
			get<0>(last.get_iterator_tuple()));
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator Decays<N,
		detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		const Decays<N, detail::BackendPolicy<BACKEND> >::value_type &x) {
	size_t pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);

	tuple_particles_iterator_type output_particle_iterator_tuple;

	do_insert(pos, output_particle_iterator_tuple, x);

	auto output_head = this->fWeights.insert(fWeights.begin() + pos, get<0>(x));

	auto output_iterator_tuple = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
			HYDRA_EXTERNAL_NS::thrust::make_tuple(output_head),
			output_particle_iterator_tuple);

	return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(output_iterator_tuple);

}

template<size_t N, detail::Backend BACKEND>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::size() const {
	return this->fWeights.size();
}

template<size_t N, detail::Backend BACKEND>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::capacity() const {
	return this->fWeights.capacity();
}

template<size_t N, detail::Backend BACKEND>
bool Decays<N, detail::BackendPolicy<BACKEND> >::empty() const {
	return this->fWeights.empty();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator Decays<N,
		detail::BackendPolicy<BACKEND> >::erase(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position) {
	size_t pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
	for (size_t i = 0; i < N; i++)
		this->fDecays[i].erase(this->fDecays[i].begin() + pos);

	this->fWeights.erase(this->fWeights.begin() + pos);

	return this->begin() + pos;
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator Decays<N,
		detail::BackendPolicy<BACKEND> >::erase(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator first,
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator last) {
	size_t pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), first);

	for (auto el = first; el != last; el++)
		this->erase(el);

	return this->begin() + pos;
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reference Decays<N,
		detail::BackendPolicy<BACKEND> >::front() {
	return this->begin()[0];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reference Decays<N,
		detail::BackendPolicy<BACKEND> >::front() const {
	return this->cbegin()[0];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reference Decays<N,
		detail::BackendPolicy<BACKEND> >::back() {
	this->begin()[this->size() - 1];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reference Decays<N,
		detail::BackendPolicy<BACKEND> >::back() const {
	this->cbegin()[this->size() - 1];
}


template<size_t N, detail::Backend BACKEND>
//hydra::pair<typename Decays<N, detail::BackendPolicy<BACKEND> >::accpeted_iterator,
//typename Decays<N, detail::BackendPolicy<BACKEND> >::accpeted_iterator >
size_t Decays<N, detail::BackendPolicy<BACKEND> >::Unweight(GUInt_t scale) {
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<
			typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator>::type system_t;

	//number of events to trial
	size_t ntrials = this->size();

	//create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > last = first
			+ ntrials;

	//get the maximum value
	GReal_t max_value = *(HYDRA_EXTERNAL_NS::thrust::max_element(
			fWeights.begin(), fWeights.end()));

	//raw pointer to weights
	GReal_t* weights_ptr = HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(
			fWeights.data());

	//says if an event passed or not
	detail::FlagDaugthers<N> predicate(scale * max_value, weights_ptr);

	//weight 1.0 all events
	HYDRA_EXTERNAL_NS::thrust::constant_iterator<GReal_t> iter_weight(1.0);

	//re-sort the container to build up un-weighted sample
	auto middle = HYDRA_EXTERNAL_NS::thrust::stable_partition(this->begin(),
			this->end(), first, predicate);

	//unpack zip-iterator
	auto begin_tuple = this->begin().get_iterator_tuple();

	//change first index
	auto begin_tpl = detail::changeFirst(iter_weight, begin_tuple);

	//done!
	return HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle);
	/*hydra::pair< typename Decays<N, detail::BackendPolicy<BACKEND> >::accpeted_iterator,
	 typename Decays<N, detail::BackendPolicy<BACKEND> >::accpeted_iterator >
	 (HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_tpl), HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_tpl)+HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle ));*/
}

template<size_t N, detail::Backend BACKEND>
template<typename FUNCTOR>
//hydra::pair<typename Decays<N, detail::BackendPolicy<BACKEND> >::decays_trans_iterator,
//typename Decays<N, detail::BackendPolicy<BACKEND> >::decays_trans_iterator >
size_t Decays<N, detail::BackendPolicy<BACKEND> >::Unweight(
		FUNCTOR const& functor, GUInt_t scale) {
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<
			typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator>::type system_t;

	//number of events to trial
	size_t ntrials = this->size();

	auto values = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer < GReal_t
			> (system_t(), ntrials);

	//create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator < size_t > last = first
			+ ntrials;

	detail::EvalOnDaugthers<N, FUNCTOR,
			typename Decays<N, detail::BackendPolicy<BACKEND> >::value_type> predicate1(
			functor);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->begin(),
					predicate1),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(this->end(),
					predicate1), values.first);

	GReal_t max_value = *(HYDRA_EXTERNAL_NS::thrust::max_element(values.first,
			values.first + values.second));

	//says if an event passed or not
	detail::FlagDaugthers<N> predicate2(scale * max_value, values.first.get());

	//weight 1.0 all events
	HYDRA_EXTERNAL_NS::thrust::constant_iterator<GReal_t> iter_weight(1.0);

	//re-sort the container to build up un-weighted sample
	auto middle = HYDRA_EXTERNAL_NS::thrust::stable_partition(this->begin(),
			this->end(), first, predicate2);

	//unpack zip-iterator
	auto begin_tuple = this->begin().get_iterator_tuple();

	//change first index
	auto begin_tpl = detail::changeFirst(iter_weight, begin_tuple);

	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(),
			values.first);

	//done!
	return (size_t) HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle);
	/*hydra::pair<
	 typename Decays<N, detail::BackendPolicy<BACKEND> >::decays_trans_iterator,
	 typename Decays<N, detail::BackendPolicy<BACKEND> >::decays_trans_iterator
	 >(  this->ptbegin(), this->ptbegin()+HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle ) );*/
	//(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_tpl), HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(begin_tpl)+HYDRA_EXTERNAL_NS::thrust::distance(begin(), middle ));
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
					predicate1), this->wbegin());

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

	auto comp = []__host__ __device__(HYDRA_EXTERNAL_NS::thrust::tuple<
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

	auto comp = []__host__ __device__(HYDRA_EXTERNAL_NS::thrust::tuple<
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
