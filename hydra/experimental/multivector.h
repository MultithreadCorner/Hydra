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
 * multivector.h
 *
 *  Created on: 18/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_H_
#define MULTIVECTOR_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/detail/multivector.inc>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>



namespace hydra {

namespace experimental {

namespace detail {

_GenerateVoidCallArgs(shrink_to_fit)
_GenerateVoidCallArgs(clear)
_GenerateVoidCallArgs(pop_back )
_GenerateVoidCallArgs(reserve)
_GenerateVoidCallArgs(resize)
_GenerateVoidCallTuple(push_back)
_GenerateNonVoidCallArgs(size)
_GenerateNonVoidCallArgs(empty)
_GenerateNonVoidCallArgs(front)
_GenerateNonVoidCallArgs(back)
_GenerateNonVoidCallArgs(begin)
_GenerateNonVoidCallArgs(end)
_GenerateNonVoidCallArgs(cbegin)
_GenerateNonVoidCallArgs(cend)
_GenerateNonVoidCallArgs(rbegin)
_GenerateNonVoidCallArgs(rend)
_GenerateNonVoidCallArgs(crbegin)
_GenerateNonVoidCallArgs(crend)
_GenerateNonVoidCallArgs(data)
_GenerateNonVoidCallArgs(capacity)
_GenerateNonVoidCallArgs(erase)
}

/**
 * The correct thing to do here is to define policies to Vector, Tuple and Zipping
 * providing the ::methods make_tuple, ::make_zip_iterator, ::get<I>, copy,
 * etc...
 * This is only a preliminary version and will work only inside thrust
 */

template< template<typename...> class V, template<typename...> class Alloc, typename ...T>
class multivector{

public:

	//allocator

    //tuples of types
	typedef thrust::tuple<T...> 			                                  value_tuple_type;
	typedef thrust::tuple<V<T, Alloc<T>>...> 		                          storage_tuple_type;
	typedef thrust::tuple<typename V<T, Alloc<T>>::pointer...> 	              pointer_tuple_type;
	typedef thrust::tuple<typename V<T, Alloc<T>>::const_pointer...> 	      const_pointer_tuple_type;
	typedef thrust::tuple<typename V<T, Alloc<T>>::reference...> 	          reference_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::const_reference...>        const_reference_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::size_type...> 	          size_type_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::iterator...> 	          iterator_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::const_iterator...>         const_iterator_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::reverse_iterator...>       reverse_iterator_tuple;
	typedef thrust::tuple<typename V<T, Alloc<T>>::const_reverse_iterator...> const_reverse_iterator_tuple;

	//zipped iterators
	typedef thrust::zip_iterator<iterator_tuple>                 iterator;
	typedef thrust::zip_iterator<const_iterator_tuple>           const_iterator;

	//zipped reverse_iterators
	typedef thrust::zip_iterator<reverse_iterator_tuple>         reverse_iterator;
	typedef thrust::zip_iterator<const_reverse_iterator_tuple>   const_reverse_iterator;

	/**
	 * default constructor
	 */
	explicit multivector():
				fStorage(thrust::make_tuple( V<T, Alloc<T>>()... ) )
	{}

	/**
	 * constructor size_t n
	 */
	explicit multivector(size_t n):
				fStorage(thrust::make_tuple( V<T, Alloc<T>>(n)... ) )
	{}

	/**
	 * copy constructor
	 */
	template< template<typename...> class V2, template<typename...> class Alloc2>
	multivector( multivector< V2, Alloc2, T... > const&  other)
	{
		resize(other.size());
		thrust::copy(other.begin(), other.end(), begin() );
	}

	/**
	 * assignment operator=
	 */
	template< template<typename...> class V2, template<typename...> class Alloc2>
	multivector< V, Alloc, T... >& operator=( multivector< V2, Alloc2, T... > const&  v)
	{
		this->resize(v.size());
		thrust::copy(v.begin(), v.end(), this->begin() );
		return *this;
	}


	void push_back(T const&... args)
	{
		push_back_call_tuple(fStorage, thrust::make_tuple(args...) );
	}

	void push_back(thrust::tuple<T...> const& args)
	{
		push_back_call_tuple( fStorage, args);
	}

	pointer_tuple_type data()
	{
		return detail::data_call_args( fStorage );
	}

	const_pointer_tuple_type data() const
	{
		return detail::data_call_args( fStorage );
	}

	const size_t size()
	{
		auto sizes = detail::size_call_args( fStorage );
		return thrust::get<0>(sizes);
	}

	size_t capacity() const
	{
		auto sizes = detail::capacity_call_args( fStorage );
		return thrust::get<0>(sizes);
	}

	bool empty() const
	{
		auto empties = detail::empty_call_args( fStorage );
		return thrust::get<0>(empties);
	}



	void resize(size_t size)
	{
		detail::resize_call_args( fStorage, size );
	}

	void clear()
	{
		detail::clear_call_args(fStorage);
	}

	void shrink_to_fit()
	{
		detail::shrink_to_fit_call_args(fStorage);
	}

	void reserve(size_t size)
	{
		detail::reserve_call_args(fStorage, size );
	}

	__host__
	reference_tuple front()
	{
		return  detail::front_call_args(fStorage);
	}

	__host__
	const_reference_tuple front() const
	{
		return  detail::front_call_args(fStorage);
	}

	__host__
	reference_tuple back()
	{
		return  detail::back_call_args(fStorage);
	}

	__host__
	const_reference_tuple back() const
	{
		return  detail::back_call_args(fStorage);
	}

    __host__
	iterator begin()
	{
		return	thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	}

	__host__
	iterator end()
	{
		return	thrust::make_zip_iterator( detail::end_call_args(fStorage) );
	}

	__host__
	const_iterator cbegin() const
	{
		return	thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	}

	__host__
	const_iterator cend() const
	{
		return	thrust::make_zip_iterator( detail::cend_call_args(fStorage) );
	}

	__host__
	reverse_iterator rbegin()
	{
		return	thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	}

	__host__
	reverse_iterator rend()
	{
		return	thrust::make_zip_iterator( detail::rend_call_args(fStorage) );
	}

	__host__
	const_reverse_iterator crbegin() const
	{
		return	thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	}

	__host__
	const_reverse_iterator crend() const
	{
		return	thrust::make_zip_iterator( detail::crend_call_args(fStorage) );
	}

	__host__
	reference_tuple operator[](size_t n)
	{
		return *(begin() + (n < size() ? n : size()-1) );
	}

	__host__
	const_reference_tuple operator[](size_t n) const
	{
		return *(begin() + (n < size() ? n : size()-1) );
	}



private:

	storage_tuple_type fStorage;

};

}  // namespace experimental

}  // namespace hydra


#endif /* MULTIVECTOR_H_ */
