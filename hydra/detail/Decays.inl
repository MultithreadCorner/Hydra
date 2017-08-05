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

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::pop_back()
{
this->fDecays.pop_back();
this->fWeights.pop_back();

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::push_back(GReal_t weight,
		const Decays<N, detail::BackendPolicy<BACKEND> >::particle_tuple& particles)
{
	std::array<Vector4R, N> arr{};
	detail::tupleToArray(particles, arr );
	for(size_t i=0; i<N; i++)
		this->fDecays[i].push_back( arr[i] );
	this->fWeights.push_back(weight);
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::push_back(GReal_t weight, Vector4R const (&particles)[N])
{
	for(size_t i=0; i<N; i++)
		this->fDecays[i].push_back( particles[i] );
	this->fWeights.push_back(weight);
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::push_back(GReal_t weight, std::initializer_list<Vector4R>const& particles)
{
	for(size_t i=0; i<N; i++)
		this->fDecays[i].push_back( *(particles.begin()+i) );
	this->fWeights.push_back(weight);
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::push_back(value_type const& value)
{
	auto particles = detail::dropFirst(value);

	std::array<Vector4R, N> arr{};

	detail::tupleToArray(particles, arr );

	for(size_t i=0; i<N; i++)
		this->fDecays[i].push_back( arr[i] );
	this->fWeights.push_back(get<0>(value));
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::resize(size_t size)
{
	for(size_t i=0; i<N; i++)
		this->fDecays[i].resize( size );
	this->fWeights.resize( size );
}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::clear()
{
	for(size_t i=0; i<N; i++)
		this->fDecays[i].clear();
	this->fWeights.clear( );

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::shrink_to_fit()
{
	for(size_t i=0; i<N; i++)
			this->fDecays[i].shrink_to_fit();
		this->fWeights.shrink_to_fit( );

}

template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::reserve(size_t size)
{
	for(size_t i=0; i<N; i++)
		this->fDecays[i].reserve(size);
	this->fWeights.reserve(size);

}


template<size_t N, detail::Backend BACKEND>
void Decays<N, detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		size_type n, const value_type &x)
{
	size_t pos = thrust::distance(this->begin(), position);
	auto particles = detail::dropFirst(x);
	std::array<Vector4R, N> arr{};
	detail::tupleToArray(particles, arr );

	for(size_t i=0; i<N; i++)
			this->fDecays[i].insert( this->fDecays[i].begin()+pos, n, arr[i] );

	this->fWeights.insert(fWeights.begin()+pos, n, get<0>(x));
}

template<size_t N, detail::Backend BACKEND>
template<typename InputIterator>
void Decays<N, detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		InputIterator first, InputIterator last)
{
	size_t pos = thrust::distance(this->begin(), position);

	auto tail_first = detail::dropFirst( first );
	std::array<typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator, N> arr_first{};
	detail::tupleToArray( tail_first, arr_first);

	auto tail_last = detail::dropFirst( last );
	std::array<typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator, N> arr_last{};
	detail::tupleToArray( tail_last, arr_last);

	for(size_t i=0; i<N; i++)
		this->fDecays[i].insert(this->fDecays[i].begin()+pos,
				arr_first[i], arr_last[i]);

	auto head_first = get<0>(first);
	auto head_last  = get<0>(last);

	this->fWeights.insert(fWeights.begin()+pos, head_first, head_last);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator
Decays<N, detail::BackendPolicy<BACKEND> >::insert(
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position,
		const Decays<N, detail::BackendPolicy<BACKEND> >::value_type &x)
{
	size_t pos = thrust::distance(this->begin(), position);

	auto head = get<0>(x);
	auto tail = detail::dropFirst( x );

	std::array<typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator, N> array_tail{};
	std::array<typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator, N> output_tail{};
	detail::tupleToArray( tail, array_tail);
	for(size_t i=0; i<N; i++)
		output_tail[i]=this->fDecays[i].insert(this->fDecays[i].begin()+pos, array_tail[i]);

	auto output_tail_tuple = detail::tupleToArray( output_tail);
	auto output_head = this->fWeights.insert(fWeights.begin()+pos,head);

	auto output_tuple = thrust::tuple_cat( thrust::make_tuple( output_head ), output_tail_tuple);

	return thrust::make_zip_iterator(output_tuple);

}

template<size_t N, detail::Backend BACKEND>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::size() const
{
 return this->fWeights.size();
}

template<size_t N, detail::Backend BACKEND>
size_t Decays<N, detail::BackendPolicy<BACKEND> >::capacity() const
{
	return this->fWeights.capacity();
}

template<size_t N, detail::Backend BACKEND>
bool Decays<N, detail::BackendPolicy<BACKEND> >::empty() const
{
	return this->fWeights.empty();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator
Decays<N, detail::BackendPolicy<BACKEND> >::erase(typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator position)
{
	size_t pos = thrust::distance(this->begin(), position);
	for(size_t i=0; i<N; i++)
		this->fDecays[i].erase(this->fDecays[i].begin()+pos);

	return this->fWeights.erase(this->fWeights.begin()+pos);

	return this->begin() + pos;
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator
Decays<N, detail::BackendPolicy<BACKEND> >::erase(typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator first,
		typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator last)
{
	size_t pos = thrust::distance(this->begin(), first);

	for( auto el = first; el!=last; el++)
		this->erase(el);

	return this->begin() + pos;
}



template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reference
Decays<N, detail::BackendPolicy<BACKEND> >::front()
{
	return this->begin()[0];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reference
Decays<N, detail::BackendPolicy<BACKEND> >::front() const
{
 	return this->cbegin()[0];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reference
Decays<N, detail::BackendPolicy<BACKEND> >::back()
{
	this->begin()[this->size()-1 ];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reference
Decays<N, detail::BackendPolicy<BACKEND> >::back() const
{
	this->cbegin()[this->size()-1 ];
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_pointer
Decays<N, detail::BackendPolicy<BACKEND> >::pdata( size_t particle, size_t component )
{
	this->fDecays[particle]->data(component);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_pointer
Decays<N, detail::BackendPolicy<BACKEND> >::pdata( size_t particle, size_t component ) const
{
	this->fDecays[particle]->data(component);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_pointer
Decays<N, detail::BackendPolicy<BACKEND> >::wdata( )
{
	this->fWeights->data();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_pointer
Decays<N, detail::BackendPolicy<BACKEND> >::wdata( ) const
{
	this->fWeights->data();
}


template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator
Decays<N, detail::BackendPolicy<BACKEND> >::begin()
{

	std::array<typename Decays<N,
	      detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
	for(size_t i=0; i<N; i++)
		particles_iterator_array[i] =this->fDecays[i].begin();
    auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

    auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.begin() ),
    		particles_iterator_tuple);

    return thrust::make_zip_iterator(  _iterator_tuple );

}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::iterator
Decays<N, detail::BackendPolicy<BACKEND> >::end()
{
	std::array<typename Decays<N,
	      detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
	for(size_t i=0; i<N; i++)
		particles_iterator_array[i] =this->fDecays[i].end();
    auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

    auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.end() ),
    		particles_iterator_tuple);

    return thrust::make_zip_iterator(  _iterator_tuple );



}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::rbegin()
{
	std::array<typename Decays<N,
		      detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
		for(size_t i=0; i<N; i++)
			particles_iterator_array[i] =this->fDecays[i].rbegin();
	    auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

	    auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.rbegin() ),
	    		particles_iterator_tuple);

	    return thrust::make_zip_iterator(  _iterator_tuple );

}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::rend()
{

	std::array<typename Decays<N,
			      detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
			for(size_t i=0; i<N; i++)
				particles_iterator_array[i] =this->fDecays[i].rend();
		    auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

		    auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.rend() ),
		    		particles_iterator_tuple);

		    return thrust::make_zip_iterator(  _iterator_tuple );

}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::begin() const
{
	std::array<typename Decays<N,
	detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
	for(size_t i=0; i<N; i++)
		particles_iterator_array[i] =this->fDecays[i].cbegin();
	auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

	auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.cbegin() ),
			particles_iterator_tuple);

	return thrust::make_zip_iterator(  _iterator_tuple );
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::end() const
{
	std::array<typename Decays<N,
	detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
	for(size_t i=0; i<N; i++)
		particles_iterator_array[i] =this->fDecays[i].cend();
	auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

	auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.cend() ),
			particles_iterator_tuple);

	return thrust::make_zip_iterator(  _iterator_tuple );

}


template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::rbegin() const
{

	std::array<typename Decays<N,
	detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
	for(size_t i=0; i<N; i++)
		particles_iterator_array[i] =this->fDecays[i].crbegin();
	auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

	auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.crbegin() ),
			particles_iterator_tuple);

	return thrust::make_zip_iterator(  _iterator_tuple );


}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::rend() const
{

	std::array<typename Decays<N,
		detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
		for(size_t i=0; i<N; i++)
			particles_iterator_array[i] =this->fDecays[i].crend();
		auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

		auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.crend() ),
				particles_iterator_tuple);

		return thrust::make_zip_iterator(  _iterator_tuple );


}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::cbegin() const
{

	std::array<typename Decays<N,
		detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
		for(size_t i=0; i<N; i++)
			particles_iterator_array[i] =this->fDecays[i].cbegin();
		auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

		auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.cbegin() ),
				particles_iterator_tuple);

		return thrust::make_zip_iterator(  _iterator_tuple );


}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::cend() const
{
	std::array<typename Decays<N,
			detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
			for(size_t i=0; i<N; i++)
				particles_iterator_array[i] =this->fDecays[i].cend();
			auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

			auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.cend() ),
					particles_iterator_tuple);

			return thrust::make_zip_iterator(  _iterator_tuple );

}


template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::crbegin() const
{

	std::array<typename Decays<N,
		detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
		for(size_t i=0; i<N; i++)
			particles_iterator_array[i] =this->fDecays[i].crbegin();
		auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

		auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.crbegin() ),
				particles_iterator_tuple);

		return thrust::make_zip_iterator(  _iterator_tuple );


}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::crend() const
{

	std::array<typename Decays<N,
			detail::BackendPolicy<BACKEND> >::particles_iterator, N> particles_iterator_array{};
			for(size_t i=0; i<N; i++)
				particles_iterator_array[i] =this->fDecays[i].crend();
			auto particles_iterator_tuple = detail::arrayToTuple( particles_iterator_array );

			auto _iterator_tuple = thrust::tuple_cat( thrust::make_tuple( fWeights.crend() ),
					particles_iterator_tuple);

			return thrust::make_zip_iterator(  _iterator_tuple );

}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::pbegin(size_t i)
{
	return fDecays->begin(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::pend(size_t i)
{
	return fDecays->end(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::prbegin(size_t i)
{
	return fDecays->rbegin(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::prend(size_t i)
{
	return fDecays->rend(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wbegin()
{
	return fWeights->begin();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wend()
{
	return fWeights->end();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wrbegin()
{
	return fWeights->rbegin();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wrend()
{
	return fWeights->rend();
}


template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::pbegin(size_t i) const
{
	return fDecays->begin(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::pend(size_t i) const
{
	return fWeights->end(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::prbegin(size_t i) const
{
	return fDecays->rbegin(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::particles_const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::prend(size_t i) const
{
	return fWeights->rend(i);
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wbegin() const
{
	return fDecays->begin();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_const_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wend() const
{
	return fWeights->end();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wrbegin() const
{
	return fDecays->rbegin();
}

template<size_t N, detail::Backend BACKEND>
typename Decays<N, detail::BackendPolicy<BACKEND> >::weights_const_reverse_iterator
Decays<N, detail::BackendPolicy<BACKEND> >::wrend() const
{
	return fWeights->rend();
}



template<size_t N1, hydra::detail::Backend BACKEND1,
         size_t N2, hydra::detail::Backend BACKEND2>
bool operator==(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs)
       {

	bool is_same_type = (N1 == N2)
			&& thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
			&& lhs.size() == rhs.size();
	bool result =1;

	auto comp = []__host__ __device__(thrust::tuple<
			typename Decays<N1, hydra::detail::BackendPolicy<BACKEND1>>::value_type,
			typename Decays<N2, hydra::detail::BackendPolicy<BACKEND2>>::value_type> const& values){
		return thrust::get<0>(values)== thrust::get<1>(values);

	};

	if(is_same_type){
			result = thrust::all_of(thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
					thrust::make_zip_iterator(lhs.end(), rhs.end()), comp);
	}
	return  result && is_same_type;

}

template<size_t N1, hydra::detail::Backend BACKEND1,
         size_t N2, hydra::detail::Backend BACKEND2>
bool operator!=(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs){

		bool is_same_type = (N1 == N2)
				&& thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
				&& lhs.size() == rhs.size();
		bool result =1;

		auto comp = []__host__ __device__(thrust::tuple<
				typename Decays<N1, hydra::detail::BackendPolicy<BACKEND1>>::value_type,
				typename Decays<N2, hydra::detail::BackendPolicy<BACKEND2>>::value_type> const& values){
			return (thrust::get<0>(values) == thrust::get<1>(values));

		};

		if(is_same_type){
				result = thrust::all_of(thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
						thrust::make_zip_iterator(lhs.end(), rhs.end()), comp);
		}
		return  (!result) && is_same_type;
}


}  // namespace hydra

#endif /* DECAYS_INL_ */
