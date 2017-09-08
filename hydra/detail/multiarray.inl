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

template< size_t N, typename T, hydra::detail::Backend BACKEND >
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::pop_back()
{
	for( size_t i=0; i<N; i++)
	this->fData[i].pop_back();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::push_back(const T (&args)[N])
{

	for( size_t i=0; i<N; i++)
	this->fData[i].push_back(args[i]);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::push_back(typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::value_type const& value)
{	do_push_back( value); }



template< size_t N, typename T, hydra::detail::Backend BACKEND >
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::push_back(std::initializer_list<T>const& list_args)
{
	if(list_args.size()!=N){
		std::cout << "Warning: std::initializer_list<T>::size() != N " << " >> not push_back failed.";
	}
	for( size_t i=0; i<N; i++)
		this->fData[i].push_back(*(list_args.begin()+i));
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::size() const
{
	return this->fData[0].size();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::capacity() const
{
	return this->fData[0].capacity();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
bool multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::empty() const
{
	return this->fData[0].empty();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::resize(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].resize( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::clear()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].clear();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::shrink_to_fit()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].shrink_to_fit();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reserve(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].reserve( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator pos)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]= this->fData[i].erase( this->fData[i].begin()+dist );

	return detail::get_zip_iterator(iter_array);

}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator first,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator last)
{
	size_t dist_first = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), first);
	size_t dist_last  = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), last);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]=this->fData[i].erase(this->fData[i].begin() + dist_first,

				this->fData[i].begin() + dist_last );

	return detail::get_zip_iterator(iter_array);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::insert(
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator pos,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::value_type const& value)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);
	multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator_tuple output{};

	do_insert(dist, output, value);
	return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(output );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
template<typename InputIterator>
void multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::insert(typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator pos,
		InputIterator first, InputIterator last)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);

	do_insert(dist, first.get_iterator_tuple(), last.get_iterator_tuple());

}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::front()
{
	return this->begin()[0];//detail::get_front_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::front() const
{
	return this->cbegin()[0];//detail::get_cfront_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::back()
{
	return this->begin()[this->size()-1 ];//detail::get_back_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::back() const
{
	return this->cbegin()[this->size()-1 ];//detail::get_cback_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::ptrs_tuple()
{
	return get_ptrs_tuple( );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::ptrs_tuple() const
{
	return get_cptrs_tuple( );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::ptrs_array()
{
	pointer_array arr{};
	for(size_t i=0; i< N; i++)
		arr[i]= this->fData[i].data();

	return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::ptrs_array() const
{
	pointer_array arr{};
		for(size_t i=0; i< N; i++)
			arr[i]= this->fData[i].data();

		return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::begin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].begin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::end()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].end();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::begin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::end() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::cbegin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::cend() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rbegin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rend()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::crbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::crend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

//==============================

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::begin(size_t i)
{
	return this->fData[i].begin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::end(size_t i)
{
	return this->fData[i].end();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rbegin(size_t i)
{
	return this->fData[i].rbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rend(size_t i)
{
	return this->fData[i].rend();
}

//==============================
template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::begin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::end(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::cbegin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::cend(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::rend(size_t i) const
{
	return this->fData[i].crend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::crbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::crend(size_t i) const
{
	return this->fData[i].crend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND >
const typename multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::vector_type&
multiarray<N,T,detail::BackendPolicy<BACKEND>,  void>::column(size_t i) const
{
	return this->fData[i];
}

/*
 *
 *
 *
 *
 *
 *
 */

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::pop_back()
{
	for( size_t i=0; i<N; i++)
	this->fData[i].pop_back();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::push_back(const T (&args)[N])
{

	for( size_t i=0; i<N; i++)
	this->fData[i].push_back(args[i]);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::push_back(typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::value_type const& value)
{	do_push_back( value); }



template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
inline void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::push_back(std::initializer_list<T>const& list_args)
{
	if(list_args.size()!=N){
		std::cout << "Warning: std::initializer_list<T>::size() != N " << " >> not push_back failed.";
	}
	for( size_t i=0; i<N; i++)
		this->fData[i].push_back(*(list_args.begin()+i));
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::size() const
{
	return this->fData[0].size();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
size_t multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::capacity() const
{
	return this->fData[0].capacity();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
bool multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::empty() const
{
	return this->fData[0].empty();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::resize(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].resize( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::clear()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].clear();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::shrink_to_fit()
{
	for( size_t i=0; i<N; i++)
		this->fData[i].shrink_to_fit();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reserve(size_t size)
{
	for( size_t i=0; i<N; i++)
		this->fData[i].reserve( size );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator pos)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]= this->fData[i].erase( this->fData[i].begin()+dist );

	return detail::get_zip_iterator(iter_array);

}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::erase(typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator first,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator last)
{
	size_t dist_first = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), first);
	size_t dist_last  = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), last);
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator_array iter_array{};

	for( size_t i=0; i<N; i++)
		iter_array[i]=this->fData[i].erase(this->fData[i].begin() + dist_first,

				this->fData[i].begin() + dist_last );

	return detail::get_zip_iterator(iter_array);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::insert(
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator pos,
		typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::value_type const& value)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);
	multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator_tuple output{};

	do_insert(dist, output, value);
	return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(output );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
template<typename InputIterator>
void multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::insert(typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator pos,
		InputIterator first, InputIterator last)
{
	size_t dist = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);

	do_insert(dist, first.get_iterator_tuple(), last.get_iterator_tuple());

}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::front()
{
	return this->begin()[0];//detail::get_front_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::front() const
{
	return this->cbegin()[0];//detail::get_cfront_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reference
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::back()
{
	return this->begin()[this->size()-1 ];//detail::get_back_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reference
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::back() const
{
	return this->cbegin()[this->size()-1 ];//detail::get_cback_tuple(this->fData);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::ptrs_tuple()
{
	return get_ptrs_tuple( );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_pointer_tuple
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::ptrs_tuple() const
{
	return get_cptrs_tuple( );
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::ptrs_array()
{
	pointer_array arr{};
	for(size_t i=0; i< N; i++)
		arr[i]= this->fData[i].data();

	return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_pointer_array
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::ptrs_array() const
{
	pointer_array arr{};
		for(size_t i=0; i< N; i++)
			arr[i]= this->fData[i].data();

		return arr;
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::begin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].begin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::end()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].end();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::trans_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::tbegin()
{
	return trans_iterator(this->begin(), detail::Caster<value_type,TargetType>());

}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::trans_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::tend()
{

	return  trans_iterator(this->end(), detail::Caster<value_type,TargetType>());
}
template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::begin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::end() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::cbegin() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::cend() const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].cend();

	return detail::get_zip_iterator(temp);
}


template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rbegin()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rend()
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].rend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator_array temp{};

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::crbegin()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crbegin();

	return detail::get_zip_iterator(temp);
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::crend()  const
{
	typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_reverse_iterator_array temp();

	for(size_t i=0; i< N; i++)
	temp[i]= this->fData[i].crend();

	return detail::get_zip_iterator(temp);
}

//==============================

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::begin(size_t i)
{
	return this->fData[i].begin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::end(size_t i)
{
	return this->fData[i].end();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rbegin(size_t i)
{
	return this->fData[i].rbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rend(size_t i)
{
	return this->fData[i].rend();
}

//==============================
template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::begin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::end(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::cbegin(size_t i) const
{
	return this->fData[i].cbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_viterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::cend(size_t i) const
{
	return this->fData[i].cend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::rend(size_t i) const
{
	return this->fData[i].crend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::crbegin(size_t i) const
{
	return this->fData[i].crbegin();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::const_vreverse_iterator
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::crend(size_t i) const
{
	return this->fData[i].crend();
}

template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
const typename multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::vector_type&
multiarray<N,T,detail::BackendPolicy<BACKEND>, TargetType>::column(size_t i) const
{
	return this->fData[i];
}

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1, typename TargetType1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2, typename TargetType2>
bool operator==(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1>, TargetType1>& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2>,  TargetType2>& rhs)
       {

	bool is_same_type = (N1 == N2)
			&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<T1,T2>::value
			&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
			&& lhs.size() == rhs.size();
	bool result =1;

	auto comp = []__host__ __device__(HYDRA_EXTERNAL_NS::thrust::tuple<T1,T2> const& values){
		return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	if(is_same_type){

		for(size_t i=0; i<N1; i++ )
			result &= HYDRA_EXTERNAL_NS::thrust::all_of(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(i), rhs.begin(i)),
					HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(i), rhs.end(i)), comp);
	}
	return  result && is_same_type;

}

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1, typename TargetType1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2, typename TargetType2>
bool operator!=(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1>,  TargetType1>& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2> ,  TargetType2>& rhs){

		bool is_same_type = (N1 == N2)
				&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<T1,T2>::value
				&& HYDRA_EXTERNAL_NS::thrust::detail::is_same<hydra::detail::BackendPolicy<BACKEND1>, hydra::detail::BackendPolicy<BACKEND2> >::value
				&& lhs.size() == rhs.size();
		bool result =1;

		auto comp = []__host__ __device__(HYDRA_EXTERNAL_NS::thrust::tuple<T1,T2> const& values){
			return (HYDRA_EXTERNAL_NS::thrust::get<0>(values) == HYDRA_EXTERNAL_NS::thrust::get<1>(values));

		};

		if(is_same_type){

			for(size_t i=0; i<N1; i++ )
				result &= HYDRA_EXTERNAL_NS::thrust::all_of(HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(i), rhs.begin(i)),
						HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(i), rhs.end(i)), comp);
		}
		return  (!result) && is_same_type;
}



}  // namespace hydra

#endif /* MULTIARRAY_INL_ */
