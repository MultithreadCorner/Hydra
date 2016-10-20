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

_GenerateVoidMember(shrink_to_fit)
_GenerateVoidMember(clear)
_GenerateVoidMember(reserve)
_GenerateVoidMember(resize)

_GenerateVoidMemberArgs(push_back)

_GenerateVoidMemberTuple(push_back)

_GenerateNonVoidMember(size)
_GenerateNonVoidMember(begin)
_GenerateNonVoidMember(end)
_GenerateNonVoidMember(cbegin)
_GenerateNonVoidMember(cend)
_GenerateNonVoidMember(rbegin)
_GenerateNonVoidMember(rend)
_GenerateNonVoidMember(crbegin)
_GenerateNonVoidMember(crend)
_GenerateNonVoidMember(data)
_GenerateNonVoidMember(capacity)

}

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

	explicit multivector():
		fStorage(thrust::make_tuple( V<T, Alloc<T>>()... ) )
	{}


	explicit multivector(size_t n):
		fStorage(thrust::make_tuple( V<T, Alloc<T>>(n)... ) )
	{}

	template< template<typename...> class V2, template<typename...> class Alloc2>
	multivector( multivector< V2, Alloc2, T... > const&  v)
	{
		this->resize(v.size());
		thrust::copy(v.begin(), v.end(), begin() );
	}

	template< template<typename...> class V2, template<typename...> class Alloc2>
	multivector< V, Alloc, T... >& operator=( multivector< V2, Alloc2, T... > const&  v)
	{
			this->resize(v.size());
			thrust::copy(v.begin(), v.end(), this->begin() );
			return *this;
	}



	void push_back(T const&... args)
	{
		push_back_invoke_with_args( fStorage, args...);
	}

	void push_back(thrust::tuple<T...> const& args)
	{
		push_back_invoke_with_tuple( fStorage, args);
	}

	pointer_tuple_type data()
	{
		return detail::data_invoke(fStorage );
	}

	const_pointer_tuple_type data() const
	{
		return detail::data_invoke(fStorage );
	}

	const size_t size()
	{
		auto sizes = detail::size_invoke(fStorage );
		return thrust::get<0>(sizes);
	}

	size_t capacity() const
	{
		auto sizes = detail::capacity_invoke(fStorage );
		return thrust::get<0>(sizes);
	}

	void resize(size_t size)
	{
		detail::resize_in_tuple(fStorage, size );
	}

	void clear()
	{
		detail::clear_in_tuple(fStorage);
	}

	void shrink_to_fit()
	{
		detail::shrink_to_fit_in_tuple(fStorage);
	}

	void reserve(size_t size)
	{
		detail::reserve_in_tuple(fStorage, size );
	}

   void swap( multivector< V, Alloc, T... >&  v	)
   {

    }

    __host__
	iterator begin()
	{
		return	thrust::make_zip_iterator( detail::begin_invoke(fStorage) );
	}

	__host__
	iterator end()
	{
		return	thrust::make_zip_iterator( detail::end_invoke(fStorage) );
	}

	__host__
	const_iterator cbegin() const
	{
		return	thrust::make_zip_iterator( detail::cbegin_invoke(fStorage) );
	}

	__host__
	const_iterator cend() const
	{
		return	thrust::make_zip_iterator( detail::cend_invoke(fStorage) );
	}

	__host__
	reverse_iterator rbegin()
	{
		return	thrust::make_zip_iterator( detail::rbegin_invoke(fStorage) );
	}

	__host__
	reverse_iterator rend()
	{
		return	thrust::make_zip_iterator( detail::rend_invoke(fStorage) );
	}

	__host__
	const_reverse_iterator crbegin() const
	{
		return	thrust::make_zip_iterator( detail::crbegin_invoke(fStorage) );
	}

	__host__
	const_reverse_iterator crend() const
	{
		return	thrust::make_zip_iterator( detail::crend_invoke(fStorage) );
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
