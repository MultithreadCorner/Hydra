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
 * multiarray.inl
 *
 *  Created on: 23/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIARRAY_INL_
#define MULTIARRAY_INL_

namespace hydra {



template< size_t N, typename T, hydra::detail::Backend BACKEND>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>>::pop_back()
{
	for( size_t i=0; i<N; i++)
	this->fData[i].pop_back();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>>::push_back(const T (&args)[N])
{

	for( size_t i=0; i<N; i++)
	this->fData[i].push_back(args[i]);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>>::push_back(typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::value_type const& value)
{

	detail::do_push_back(this->fData, value);

}



template< size_t N, typename T, hydra::detail::Backend BACKEND>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>>::push_back(std::initializer_list<T>const& list_args)
{
	if(list_args.size()!=N){
		std::cout << "Warning: std::initializer_list<T>::size() != N " << " >> not push_back failed.";
	}
	for( size_t i=0; i<N; i++)
		this->fData[i].push_back(*(list_args.begin()+i));
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>>::size() const
{
	return this->fData[0].size();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>>::capacity() const
{
	return this->fData[0].capacity();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
bool multiarray<N,T,detail::BackendPolicy<BACKEND>>::empty() const
{
	return this->fData[0].empty();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
void multiarray<N,T,detail::BackendPolicy<BACKEND>>::resize(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].resize( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
void multiarray<N,T,detail::BackendPolicy<BACKEND>>::clear()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].clear();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
void multiarray<N,T,detail::BackendPolicy<BACKEND>>::shrink_to_fit()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].shrink_to_fit();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
void multiarray<N,T,detail::BackendPolicy<BACKEND>>::reserve(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].reserve( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator pos)
{
	size_t dist = thrust::distance(this->begin(), pos);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]= this->fData[i].erase( this->fData[i].begin()+dist );

	return detail::get_zip_iterator(iter_array);

}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator first,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator last)
{
	size_t dist_first = thrust::distance(this->begin(), first);
	size_t dist_last  = thrust::distance(this->begin(), last);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]=this->fData[i].erase(this->fData[i].begin() + dist_first,  this->fData[i].begin() + dist_last );

	return detail::get_zip_iterator(iter_array);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::insert(typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator pos,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::value_type const& value)
{
	size_t dist = thrust::distance(this->begin(), pos);
	multiarray<N,T,detail::BackendPolicy<BACKEND>>::array_type value_array{};
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator_array iter_array{};
	detail::tupleToArray(value, value_array);

	for( size_t i=0; i<N; i++)
		iter_array[i]=this->fData[i].insert(this->fData[i].begin() + dist, value_array[i] );

	return detail::get_zip_iterator(iter_array);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
template<typename InputIterator>
void multiarray<N,T,detail::BackendPolicy<BACKEND>>::insert(typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator pos,
		InputIterator first, InputIterator last)
{
	size_t dist = thrust::distance(this->begin(), pos);

	detail::do_insert(dist, this->fData, first.get_iterator_tuple(), last.get_iterator_tuple());

}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>>::front()
{
	return this->begin()[0];//detail::get_front_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>>::front() const
{
	return this->cbegin()[0];//detail::get_cfront_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>>::back()
{
	return this->begin()[this->size()-1 ];//detail::get_back_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>>::back() const
{
	return this->cbegin()[this->size()-1 ];//detail::get_cback_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>>::ptrs_tuple()
{
	return detail::get_data_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>>::ptrs_tuple() const
{
	return detail::get_cdata_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>>::ptrs_array()
{
	pointer_array arr{};
	for(size_t i=0; i< N; i++)
		arr[i]= this->fData[i].data();

	return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>>::ptrs_array() const
{
	pointer_array arr{};
		for(size_t i=0; i< N; i++)
			arr[i]= this->fData[i].data();

		return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::begin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].begin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::end()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].end();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::begin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::end() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::cbegin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::cend() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rbegin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rend()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::crbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::crend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

//==============================

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::begin(size_t i)
{
	return this->fData[i].begin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::end(size_t i)
{
	return this->fData[i].end();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rbegin(size_t i)
{
	return this->fData[i].rbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rend(size_t i)
{
	return this->fData[i].rend();
}

//==============================
template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::begin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::end(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::cbegin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::cend(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::rend(size_t i) const
{
	return this->fData[i].crend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::crbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>>::crend(size_t i) const
{
	return this->fData[i].crend();
}


template<size_t N1, typename T1, hydra::detail::Backend BACKEND1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2>
bool operator==(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2> >& rhs)
       {

	bool is_same_type = (N1 == N2)
			&& thrust::detail::is_same<T1,T2>::value
			&& thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
			&& lhs.size() == rhs.size();
	bool result =1;

	auto comp = []__host__ __device__(thrust::tuple<T1,T2> const& values){
		return thrust::get<0>(values)== thrust::get<1>(values);

	};

	if(is_same_type){

		for(size_t i=0; i<N1; i++ )
			result &= thrust::all_of(thrust::make_zip_iterator(lhs.begin(i), rhs.begin(i)),
					thrust::make_zip_iterator(lhs.end(i), rhs.end(i)), comp);
	}
	return  result && is_same_type;

}

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2>
bool operator!=(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2> >& rhs){

		bool is_same_type = (N1 == N2)
				&& thrust::detail::is_same<T1,T2>::value
				&& thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
				&& lhs.size() == rhs.size();
		bool result =1;

		auto comp = []__host__ __device__(thrust::tuple<T1,T2> const& values){
			return (thrust::get<0>(values) == thrust::get<1>(values));

		};

		if(is_same_type){

			for(size_t i=0; i<N1; i++ )
				result &= thrust::all_of(thrust::make_zip_iterator(lhs.begin(i), rhs.begin(i)),
						thrust::make_zip_iterator(lhs.end(i), rhs.end(i)), comp);
		}
		return  (!result) && is_same_type;
}



}  // namespace hydra

#endif /* MULTIARRAY_INL_ */
