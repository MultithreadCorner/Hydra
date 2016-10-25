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

_GenerateNonVoidCallArgsC(size)
_GenerateNonVoidCallArgsC(empty)
_GenerateNonVoidCallArgs(front)
_GenerateNonVoidCallArgsC(front)
_GenerateNonVoidCallArgs(back)
_GenerateNonVoidCallArgs(begin)
_GenerateNonVoidCallArgs(end)
_GenerateNonVoidCallArgsC(cbegin)
_GenerateNonVoidCallArgsC(cend)
_GenerateNonVoidCallArgs(rbegin)
_GenerateNonVoidCallArgs(rend)
_GenerateNonVoidCallArgsC(crbegin)
_GenerateNonVoidCallArgsC(crend)
_GenerateNonVoidCallArgs(data)
_GenerateNonVoidCallArgsC(data)
_GenerateNonVoidCallArgsC(capacity)
_GenerateNonVoidCallArgs(erase)

}

/**
 * The correct thing to do here is to define policies to Vector, Tuple and Zipping
 * providing the ::methods make_tuple, ::make_zip_iterator, ::get<I>, copy,
 * etc...
 * This is only a preliminary version and will work only inside thrust
 */

template< template<typename...> class Vector, template<typename...> class Allocator, typename ...T>
class multivector{

public:

	//allocator

    //tuples of types
	typedef thrust::tuple<T...> 			                                          value_tuple_type;
	typedef thrust::tuple<Vector<T, Allocator<T>>...> 		                          storage_tuple_type;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::pointer...> 	          pointer_tuple_type;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::const_pointer...> 	      const_pointer_tuple_type;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::reference...> 	          reference_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::const_reference...>       const_reference_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::size_type...> 	          size_type_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::iterator...> 	          iterator_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::const_iterator...>         const_iterator_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::reverse_iterator...>       reverse_iterator_tuple;
	typedef thrust::tuple<typename Vector<T, Allocator<T>>::const_reverse_iterator...> const_reverse_iterator_tuple;

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
				fStorage(thrust::make_tuple( Vector<T, Allocator<T>>()... ) ),
				fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
				fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
				fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
				fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
				fSize( thrust::get<0>(fStorage ).size())
	{}

	/**
	 * constructor size_t n
	 */
	explicit multivector(size_t n):
				fStorage(thrust::make_tuple( Vector<T, Allocator<T>>(n)... ) ),
				fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
				fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
				fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
				fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
	            fSize( thrust::get<0>(fStorage ).size())
	{}

	/**
	 * copy constructor
	 */

	template< template<typename...> class Vector2, template<typename...> class Allocator2>
	multivector( multivector< Vector2, Allocator2, T... > const&  other)
	{
		this->resize(other.size());

		thrust::copy(other.begin(), other.end(), this->begin() );

	}

	/**
	 * assignment operator=
	 */

	template< template<typename...> class Vector2, template<typename...> class Allocator2>
	multivector< Vector, Allocator, T... >& operator=( multivector< Vector2, Allocator2, T... > const&  v)
	{
		this->resize(v.size());

		thrust::copy(v.begin(), v.end(), this->begin() );

		return *this;
	}

	__host__ inline
	void pop_back()
	{
		detail::pop_back_call_args(fStorage);
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin=thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin=thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__ inline
	void push_back(T const&... args)
	{
		detail::push_back_call_tuple(fStorage, thrust::make_tuple(args...) );
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__ inline
	void push_back(thrust::tuple<T...> const& args)
	{
		detail::push_back_call_tuple( fStorage, args);
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__
	pointer_tuple_type data()
	{
		return detail::data_call_args( fStorage );

	}

	__host__
	const_pointer_tuple_type data() const
	{
		return detail::data_call_args( fStorage );
	}

	__host__
	size_t size() const
	{
		//auto sizes = detail::size_call_args( fStorage );
		//return thrust::get<0>(fStorage ).size();
		return fSize;
	}

	__host__
	size_t capacity() const
	{
		//auto sizes = detail::capacity_call_args( fStorage );
		//return thrust::get<0>(sizes);
		return thrust::get<0>(fStorage ).capacity();
	}

	__host__
	bool empty() const
	{
		//auto empties = detail::empty_call_args( fStorage );
		//return thrust::get<0>(empties);
		return thrust::get<0>(fStorage ).empty();
	}

	__host__
	void resize(size_t size)
	{
		detail::resize_call_args( fStorage, size );
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__
	void clear()
	{
		detail::clear_call_args(fStorage);
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__
	void shrink_to_fit()
	{
		detail::shrink_to_fit_call_args(fStorage);
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}
	__host__
	void reserve(size_t size)
	{
		detail::reserve_call_args(fStorage, size );
		this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
		this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
		this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
		this->fSize = thrust::get<0>(fStorage ).size();
	}

	__host__ inline
	reference_tuple front()
	{
		return  detail::front_call_args(fStorage);
	}

	__host__ inline
	const_reference_tuple front() const
	{
		return  detail::front_call_args(fStorage);
	}

	__host__ inline
	reference_tuple back()
	{
		return  detail::back_call_args(fStorage);
	}

	__host__ inline
	const_reference_tuple back() const
	{
		return  detail::back_call_args(fStorage);
	}

    __host__ inline
	iterator begin()
	{
		return	fBegin;
	}

	__host__ inline
	iterator end()
	{
		return	fBegin+fSize;
	}

	__host__ inline
	const_iterator cbegin() const
	{
		return	fConstBegin;
	}

	__host__ inline
	const_iterator cend() const
	{
		return	fConstBegin+fSize;
	}

	__host__ inline
	reverse_iterator rbegin()
	{
		return	fReverseBegin;
	}

	__host__ inline
	reverse_iterator rend()
	{
		return	fReverseBegin+fSize;
	}

	__host__ inline
	const_reverse_iterator crbegin() const
	{
		return	fConstReverseBegin;
	}

	__host__ inline
	const_reverse_iterator crend() const
	{
		return	fConstReverseBegin+fSize;
	}

	__host__ inline
	reference_tuple operator[](size_t n)
	{
		//iterator t = begin();
		return fBegin[n] ;
	}

	__host__ inline
	const_reference_tuple operator[](size_t n) const
	{
		//const_iterator t = cbegin();
		return fConstBegin[n];
	}



private:

	storage_tuple_type fStorage;
	iterator fBegin;
	reverse_iterator fReverseBegin;
	const_iterator fConstBegin;
	const_reverse_iterator fConstReverseBegin;

	size_t   fSize;

};

}  // namespace experimental

}  // namespace hydra


#endif /* MULTIVECTOR_H_ */
