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
 * multiarray.h
 *
 *  Created on: 22/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIARRAY_H_
#define MULTIARRAY_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <array>

//#include<hydra/detail/multiarray.inc>

namespace hydra {

template<size_t N, typename T, typename BACKEND>
class multiarray;

/**
 * @brief This class implements storage in SoA layouts for
 * table where all elements have the same type.
 */
template< size_t N, typename T, hydra::detail::Backend BACKEND>
class multiarray<N, T, hydra::detail::BackendPolicy<BACKEND> >
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<T> vector_t;
	typedef std::array<vector_t, N> data_t;

public:

	typedef detail::tuple_type<N, T> row_t;
	typedef std::array<T, N> array_type;
	//reference
	typedef typename vector_t::reference vreference;
	typedef typename vector_t::const_reference const_vreference;
	typedef typename detail::tuple_type<N,vreference>::type reference_tuple;
	typedef typename detail::tuple_type<N,const_vreference>::type const_reference_tuple;

	//pointer
	typedef typename vector_t::pointer vpointer;
	typedef typename vector_t::const_pointer const_vpointer;

	//vector iterators
	typedef vector_t vector_type;
	typedef typename vector_t::iterator 				viterator;
	typedef typename vector_t::const_iterator 			const_viterator;
	typedef typename vector_t::reverse_iterator 		vreverse_iterator;
	typedef typename vector_t::const_reverse_iterator 	const_vreverse_iterator;

	//iterators tuple
	typedef typename detail::tuple_type<N, viterator>::type 				iterator_tuple;
	typedef typename detail::tuple_type<N, const_viterator>::type 			const_iterator_tuple;
	typedef typename detail::tuple_type<N, vreverse_iterator>::type 		reverse_iterator_tuple;
	typedef typename detail::tuple_type<N, const_vreverse_iterator>::type	const_reverse_iterator_tuple;

	//iterators array
	typedef std::array<viterator, N> 				iterator_array;
	typedef std::array<const_viterator, N> 			const_iterator_array;
	typedef std::array<vreverse_iterator, N> 		reverse_iterator_array;
	typedef std::array<const_vreverse_iterator, N> 	const_reverse_iterator_array;

	//zip iterator
	typedef thrust::zip_iterator<iterator_tuple>		 iterator;
	typedef thrust::zip_iterator<const_iterator_tuple>	 const_iterator;
	typedef thrust::zip_iterator<reverse_iterator_tuple>		 reverse_iterator;
	typedef thrust::zip_iterator<const_reverse_iterator_tuple>	 const_reverse_iterator;

	//stl-like typedefs
	 typedef size_t size_type;

	 typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
	 typedef typename thrust::iterator_traits<iterator>::value_type value_type;
	 //typedef typename thrust::iterator_traits<iterator>::pointer pointer;
	 typedef typename detail::tuple_type<N,vpointer>::type pointer_tuple;
	 typedef typename detail::tuple_type<N,const_vpointer>::type const_pointer_tuple;
	 typedef std::array<vpointer,N>       pointer_array;
	 typedef std::array<const_vpointer,N> const_pointer_array;

	 typedef typename thrust::iterator_traits<iterator>::reference reference;
	 typedef typename thrust::iterator_traits<const_iterator>::reference const_reference;
	 typedef typename thrust::iterator_traits<reverse_iterator>::reference reverse_reference;
	 typedef typename thrust::iterator_traits<const_reverse_iterator>::reference const_reverse_reference;
	 typedef typename thrust::iterator_traits<iterator>::iterator_category iterator_category;

	multiarray():
		fData(data_t())
	{

	};

	multiarray(size_t n)
	{


		fData = data_t();
		for( size_t i=0; i<N; i++)
			fData[i].resize(n);
	};

	multiarray(multiarray<N,T,detail::BackendPolicy<BACKEND>> const& other )
	{
		fData = data_t();
		for( size_t i=0; i<N; i++)
			fData[i] = std::move(vector_t(other.begin(i), other.end(i)));
	}

	multiarray(multiarray<N,T,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.MoveData())
	{}

	template< hydra::detail::Backend BACKEND2>
	multiarray(multiarray<N,T,detail::BackendPolicy<BACKEND2>> const& other )
	{
		fData = data_t();

		for( size_t i=0; i<N; i++)
			fData[i] = std::move( vector_t( other.begin(i), other.end(i) ) );
	}

	template< typename Iterator>
	multiarray(Iterator begin, Iterator end )
	{
		fData = data_t();
		do_copy(begin, end );

	}

	multiarray<N,T,detail::BackendPolicy<BACKEND>>&
	operator=(multiarray<N,T,detail::BackendPolicy<BACKEND>> const& other )
	{
			if(this==&other) return *this;

			for( size_t i=0; i<N; i++)
				this->fData[i] = std::move(vector_t(other.begin(), other.end()));

			return *this;
	}

	multiarray<N,T,detail::BackendPolicy<BACKEND>>&
	operator=(multiarray<N,T,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this==&other) return *this;
		this->fData =other.MoveData();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	multiarray<N,T,detail::BackendPolicy<BACKEND> >&
	operator=(multiarray<N,T,detail::BackendPolicy<BACKEND2> > const& other )
	{

		for( size_t i=0; i<N; i++)
			this->fData[i] = std::move( vector_t( other.begin(i), other.end(i) ) );
		return *this;
	}


	inline void pop_back();

	inline void push_back(const T (&args)[N]);

	inline void	push_back(std::initializer_list<T>const& list_args);

	inline void	push_back(value_type const& value);

	size_t size() const;

	size_t capacity() const;

	bool empty() const;

	void resize(size_t size);

	void clear();

	void shrink_to_fit();

	void reserve(size_t size);

	iterator erase(iterator pos);

	iterator erase(iterator first, iterator last);

	iterator insert(iterator position, const value_type &x);

	void insert(iterator position, size_type n, const value_type &x);


	template<typename InputIterator>
	void insert(iterator position, InputIterator first, InputIterator last);

	reference front();

	const_reference front() const;

	reference back();

	const_reference back() const;

	pointer_tuple ptrs_tuple();
	const_pointer_tuple ptrs_tuple() const;

	pointer_array ptrs_array();
	const_pointer_array ptrs_array() const;

    //vpointer data( size_t i);
    //const_vpointer data( size_t i) const;

	//non-constant access
	iterator begin();
	iterator end();

	//non-constant access
	reverse_iterator rbegin();
	reverse_iterator rend();

	//constant access
	const_iterator begin() const;
	const_iterator end() const;
	const_reverse_iterator rbegin() const;
	const_reverse_iterator rend() const;
	const_iterator cbegin() const;
	const_iterator cend() const;
	const_reverse_iterator crbegin() const;
	const_reverse_iterator crend() const;

	//non-constant access
	viterator begin(size_t i);
	viterator end(size_t i);
	vreverse_iterator rbegin(size_t i);
	vreverse_iterator rend(size_t i);

	//constant access
	const_viterator begin(size_t i) const;
	const_viterator end(size_t i) const;
	const_viterator cbegin(size_t i)  const;
	const_viterator cend(size_t i) const ;
	const_vreverse_iterator rbegin(size_t i) const ;
	const_vreverse_iterator rend(size_t i) const ;
	const_vreverse_iterator crbegin(size_t i) const ;
	const_vreverse_iterator crend(size_t i) const ;

	const vector_type& column(size_t i) const;

	//
	inline	reference_tuple operator[](size_t n)
	{	return begin()[n] ;	}

	inline const_reference_tuple operator[](size_t n) const
	{	return cbegin()[n]; }

private:

	//----------------------------------------------
	//ptrs
	template<size_t ...Index>
	inline pointer_tuple get_ptrs_tuple_helper( detail::index_sequence<Index...>)
	{	return hydra::make_tuple( fData[Index].data()... ); }

	inline pointer_tuple get_ptrs_tuple()
	{ return get_ptrs_tuple_helper( detail::make_index_sequence<N> { } ); }

	//cptrs
	template<size_t ...Index>
	inline const_pointer_tuple get_cptrs_tuple_helper( detail::index_sequence<Index...>)
	{	return hydra::make_tuple(fData[Index].data()... ); }

	inline const_pointer_tuple get_cptrs_tuple() const
	{ return get_cptrs_tuple_helper( detail::make_index_sequence<N> { } ); }

	//copy
	template<size_t I, typename Iterator>
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_copy(Iterator begin, Iterator end )
	{ }

	template<size_t I=0, typename Iterator>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_copy(Iterator begin, Iterator end)
	{

		fData[I] = std::move( vector_t( get<I>(begin.get_iterator_tuple()) ,
				get<I>(end.get_iterator_tuple()) ) );
		do_copy<I + 1>( begin, end);
	}


	//insert
	template<size_t I>
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_insert(size_t dist, iterator_tuple& output, value_type const& value)
	{ }

	template<size_t I=0>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_insert(size_t dist, iterator_tuple& output, value_type const& value)
	{
		get<I>(output) = fData[I].insert(fData[I].begin() + dist, get<I>(value) );
	    do_insert<I + 1>(dist, output,value );
	}

	template<size_t I, template<typename ...> class Tuple, typename ...Iterators>
	inline typename thrust::detail::enable_if<(I == sizeof...(Iterators)), void >::type
	do_insert(size_t dist, Tuple<Iterators...> const& first_tuple, Tuple<Iterators...> const& last_tuple)
	{}

	template<size_t I = 0, template<typename ...> class Tuple, typename ...Iterators>
	inline typename thrust::detail::enable_if<(I < sizeof...(Iterators)), void >::type
	do_insert(size_t dist, Tuple<Iterators...> const& first, Tuple<Iterators...> const& last)
	{
	    fData[I].insert(fData[I].begin() + dist, get<I>(first), get<I>(last) );
	    do_insert<I + 1, Tuple, Iterators... >(dist, first, last );
	}

    //push_back
	template<size_t I>
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_push_back(value_type const& value)
	{}

	template<size_t I = 0>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_push_back(value_type const& value)
	{
	    fData[I].push_back(get<I>(value));
	    do_push_back<I + 1>(value );
	}



	data_t MoveData()
	{
		return std::move(fData);
	}

	data_t fData;


};

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2>
bool operator==(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2> >& rhs);

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2>
bool operator!=(const multiarray<N1, T1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const multiarray<N2, T2, hydra::detail::BackendPolicy<BACKEND2> >& rhs);



}  // namespace hydra

#include<hydra/detail/multiarray.inl>

#endif /* MULTIARRAY_H_ */
