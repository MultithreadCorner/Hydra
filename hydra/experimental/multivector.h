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

/**
 * The correct thing to do here is to define policies to Vector, Tuple and Zipping
 * providing the ::methods make_tuple, ::make_zip_iterator, ::get<I>, copy,
 * etc...
 * This is only a preliminary version and will work only inside thrust
 */

template< template<typename...> class Vector, template<typename...> class Allocator, typename ...T>
class multivector{

public:


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
	explicit multivector();

	/**
	 * constructor size_t n
	 */
	explicit multivector(size_t n);

	/**
	 * constructor size_t n, ...values
	 */

	explicit multivector(size_t n, T... value);

	/**
	 * constructor size_t n, ...values
	 */
	explicit multivector(size_t n, value_tuple_type  value);

	/**
	 * copy constructor
	 */
	template< template<typename...> class Vector2,
	template<typename...> class Allocator2>
	multivector( multivector< Vector2, Allocator2, T... >const&  other);


	/**
	 * assignment operator=
	 */

	template< template<typename...> class Vector2, template<typename...> class Allocator2>
	multivector< Vector, Allocator, T... >&
	operator=( multivector< Vector2, Allocator2, T... > const&  v);

	inline void pop_back();

	inline void push_back(T const&... args);

	inline void push_back(thrust::tuple<T...> const& args);

	pointer_tuple_type data();

	const_pointer_tuple_type data() const;

	size_t size() const;

	size_t capacity() const;

	bool empty() const;

	void resize(size_t size);

	void clear();

	void shrink_to_fit();

	void reserve(size_t size);

	reference_tuple front();

	const_reference_tuple front() const;

	reference_tuple back();

	const_reference_tuple back() const;

    iterator begin();

	iterator end();

	const_iterator cbegin() const;

    const_iterator cend() const;

	reverse_iterator rbegin();

	reverse_iterator rend();

	const_reverse_iterator crbegin() const;

	const_reverse_iterator crend() const;

	//----------------------
	template<unsigned int I>
	auto vbegin()
	-> typename thrust::tuple_element<I, iterator_tuple>::type;

	template<unsigned int I>
	auto vend()
	-> typename thrust::tuple_element<I, iterator_tuple>::type;

	template<unsigned int I>
	auto vcbegin() const
	-> typename thrust::tuple_element<I, const_iterator_tuple>::type;

	template<unsigned int I>
	auto vcend() const
	-> typename thrust::tuple_element<I, const_iterator_tuple>::type;

	template<unsigned int I>
	auto vrbegin()
	-> typename thrust::tuple_element<I, reverse_iterator_tuple>::type;

	template<unsigned int I>
	auto vrend()
	-> typename thrust::tuple_element<I, reverse_iterator_tuple>::type;

	template<unsigned int I>
	auto vcrbegin() const
	-> typename thrust::tuple_element<I, const_reverse_iterator_tuple>::type;

	template<unsigned int I>
	auto vcrend() const
	-> typename thrust::tuple_element<I, const_reverse_iterator_tuple>::type;

	//-------------------------------------
	inline	reference_tuple operator[](size_t n)
	{	return fBegin[n] ;	}

	 inline const_reference_tuple operator[](size_t n) const
	{	return fConstBegin[n]; }


private:

	storage_tuple_type fStorage;
	iterator fBegin;
	reverse_iterator fReverseBegin;
	const_iterator fConstBegin;
	const_reverse_iterator fConstReverseBegin;

	iterator_tuple fTBegin;
	const_iterator_tuple fTConstBegin;
	reverse_iterator_tuple fTReverseBegin;
	const_reverse_iterator_tuple fTConstReverseBegin;

	size_t   fSize;

};


}  // namespace experimental

}  // namespace hydra

#include <hydra/experimental/detail/multivector.inl>

#endif /* MULTIVECTOR_H_ */
