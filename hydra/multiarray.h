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
 * multiarray.h
 *
 *  Created on: 25/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIARRAY_H_
#define MULTIARRAY_H_

#include <array>
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/Caster.h>
#include <hydra/Tuple.h>
#include <hydra/Placeholders.h>
#include <hydra/Iterator.h>
#include <hydra/Range.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/logical.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>


namespace hydra {

template<typename T, size_t N,  typename BACKEND>
class multiarray;

template< typename T, size_t N,  hydra::detail::Backend BACKEND>
class multiarray<T, N, hydra::detail::BackendPolicy<BACKEND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND>    system_t;

	typedef std::array<typename system_t::template container<T>, N> storage_t;

public:


	//------------------
	//column typedefs
    //------------------
	typedef typename system_t::template container<T> column_type;
	//reference
	typedef typename column_type::reference reference_v;
	typedef typename column_type::const_reference const_reference_v;
	//pointer
	typedef typename column_type::pointer pointer_v;
	typedef typename column_type::const_pointer const_pointer_v;
	//vector iterators
	//typedef vector_t vector_type;
	typedef typename column_type::iterator 				iterator_v;
	typedef typename column_type::const_iterator 			const_iterator_v;
	typedef typename column_type::reverse_iterator 		reverse_iterator_v;
	typedef typename column_type::const_reverse_iterator 	const_reverse_iterator_v;
	//-----------------------
	//tuple of iterators
	//-----------------------
	typedef typename detail::tuple_type<N, iterator_v>::type 				iterator_t;
	typedef typename detail::tuple_type<N, const_iterator_v>::type 			const_iterator_t;
	typedef typename detail::tuple_type<N, reverse_iterator_v>::type 		reverse_iterator_t;
	typedef typename detail::tuple_type<N, const_reverse_iterator_v>::type	const_reverse_iterator_t;
	//-----------------------
	//array of iterators
	//-----------------------
	typedef std::array<iterator_v, N> 				iterator_a;
	typedef std::array<const_iterator_v, N> 			const_iterator_a;
	typedef std::array<reverse_iterator_v, N> 		reverse_iterator_a;
	typedef std::array<const_reverse_iterator_v, N> 	const_reverse_iterator_a;

	//-----------------------
	//STL-like interface
	//-----------------------
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<iterator_t>		          iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_iterator_t>	      const_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<reverse_iterator_t>		  reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_reverse_iterator_t> const_reverse_iterator;

	typedef size_t size_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::difference_type difference_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_iterator>::reference const_reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::iterator_category iterator_category;

	typedef typename detail::tuple_type<N, T>::type tuple_type;
	//cast iterator
	template<typename Functor>
	using caster_iterator = HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
			iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	template<typename Functor>
	using caster_reverse_iterator = HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
			reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	//constructors


	multiarray() = default;

	multiarray(size_t n) { __resize(n); };

	multiarray(size_t n, value_type const& value) {
		__resize(n);
		HYDRA_EXTERNAL_NS::thrust::fill(begin(), end(), value );
	}

	template<typename Int, typename = typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_integral<Int>::value>::type >
	multiarray(hydra::pair<Int, typename detail::tuple_type<N, T>::type > const& pair){
		__resize(pair.first);
		HYDRA_EXTERNAL_NS::thrust::fill(begin(), end(), pair.second );
	}


	multiarray(multiarray<T,N,detail::BackendPolicy<BACKEND>> const& other )
	{
		__resize( other.size() );
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
	}

	multiarray(multiarray<T,N,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.__move())
	{}

	template< hydra::detail::Backend BACKEND2>
	multiarray(multiarray<T,N,detail::BackendPolicy<BACKEND2>> const& other )
	{
		__resize(other.size());
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
	}

	template< typename Iterator>
	multiarray(Iterator first, Iterator last )
	{
		__resize( HYDRA_EXTERNAL_NS::thrust::distance(first, last) );
		HYDRA_EXTERNAL_NS::thrust::copy(first, last, begin());
	}

	// assignment

	multiarray<T,N,detail::BackendPolicy<BACKEND>>&
	operator=(multiarray<T,N,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(this==&other) return *this;

		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());

		return *this;
	}

	multiarray<T,N,detail::BackendPolicy<BACKEND>>&
	operator=(multiarray<T,N,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this == &other) return *this;
		this->fData = other.__move();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	multiarray<T,N,detail::BackendPolicy<BACKEND> >&
	operator=(multiarray<T,N,detail::BackendPolicy<BACKEND2> > const& other )
	{
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
		return *this;
	}


	inline void pop_back()
	{
		__pop_back();
	}

	inline void	push_back(value_type const& value)
	{
		__push_back( value );
	}

	template<typename Functor, typename Obj>
	inline void	push_back(Functor  const& functor, Obj const& obj)
	{
		__push_back( functor(obj) );
	}

	inline size_type size() const
	{
		return HYDRA_EXTERNAL_NS::thrust::distance( begin(), end() );
	}

	inline size_type capacity() const
	{
		return std::get<0>(fData).capacity();
	}

	inline bool empty() const
	{
		return std::get<0>(fData).empty();
	}

	inline void resize(size_type size)
	{
		this->__resize(size);
	}


	inline void clear()
	{
		this->__clear();
	}

	inline void shrink_to_fit()
	{
		this->__shrink_to_fit();
	}

	inline void reserve(size_type size)
	{
		this->__reserve(size);
	}

	inline iterator erase(iterator pos)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		return this->__erase(position);
	}

	inline iterator erase(iterator first, iterator last)
	{
		size_type first_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), first);
		size_type last_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), last);

		return this->__erase( first_position, last_position);
	}

	inline iterator insert(iterator pos, const value_type &x)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		return this->__insert(position, x);
	}

	inline void insert(iterator pos, size_type n, const value_type &x)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		this->__insert(position, n, x);
	}


	template< typename InputIterator>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	 detail::is_instantiation_of< HYDRA_EXTERNAL_NS::thrust::zip_iterator, InputIterator>::value, void>::type
	insert(iterator pos, InputIterator first, InputIterator last)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);

		this->__insert(position, first.get_iterator_tuple(), last.get_iterator_tuple());
	}


	inline reference front()
	{
		return this->__front();
	}

	inline const_reference front() const
	{
	   return this->__front();
	}

	inline reference back()
	{
		return this->__back();
	}

	inline const_reference back() const
	{
		return this->__back();
	}

	//non-constant access
	inline iterator begin()
	{
		return this->__begin();
	}

	inline iterator end()
	{
		return this->__end();
	}

	//constant automatic conversion access
	template<typename Functor>
	inline caster_iterator<Functor> begin( Functor const& caster )
	{
		return this->__caster_begin(caster);
	}


	template<typename Functor>
	inline caster_iterator<Functor> end( Functor const& caster )
	{
		return this->__caster_end(caster);
	}

	//constant automatic conversion access
	template<typename Functor>
	inline 	caster_reverse_iterator<Functor> rbegin( Functor const& caster )
	{
		return this->__caster_rbegin(caster);
	}

	template<typename Functor>
	inline caster_reverse_iterator<Functor> rend( Functor const& caster )
	{
		return this->__caster_rend(caster);
	}

	//non-constant access
	inline reverse_iterator rbegin()
	{
		return this->__rbegin();
	}

	inline reverse_iterator rend()
	{
		return this->__rend();
	}

	//constant access
	inline const_iterator begin() const
	{
		return this->__begin();
	}

	inline const_iterator end() const
	{
		return this->__end();
	}

	inline const_reverse_iterator rbegin() const
	{
		return this->__rbegin();
	}

	inline const_reverse_iterator rend() const
	{
		return this->__rend();
	}

	inline const_iterator cbegin() const
	{
		return this->__cbegin() ;
	}

	inline const_iterator cend() const
	{
		return this->	__cend() ;
	}

	inline const_reverse_iterator crbegin() const
	{
		return this->__crbegin() ;
	}

	inline const_reverse_iterator crend() const
	{
		return 	this->__crend() ;
	}

	//-------------------------
	//without placeholders
	//-------------------------
	//non-constant access
	inline iterator_v begin(size_t i)
	{
		return fData[i].begin();
	}

	inline iterator_v end(size_t i)
	{
		return fData[i].end();
	}

	inline reverse_iterator_v rbegin(size_t i)
	{
		return fData[i].rbegin();
	}

	inline reverse_iterator_v rend(size_t i)
	{
		return fData[i].rend();
	}

	//constant access
	inline const_iterator_v begin(size_t i) const
	{
		return fData[i].cbegin();
	}

	inline const_iterator_v end(size_t i) const
	{
		return fData[i].cend();
	}

	inline const_iterator_v cbegin(size_t i)  const
	{
		return fData[i].cbegin();
	}

	inline const_iterator_v cend(size_t i) const
	{
		return fData[i].cend();
	}

	inline const_reverse_iterator_v rbegin(size_t i) const
	{
		return fData[i].crbegin();
	}

	inline const_reverse_iterator_v rend(size_t i) const
	{
		return fData[i].crend();
	}

	inline const_reverse_iterator_v crbegin(size_t i) const
	{
		return fData[i].crbegin();
	}

	inline const_reverse_iterator_v crend(size_t i) const
	{
		return fData[i].crend();
	}

	inline const column_type& column(size_t i ) const
	{
		return fData[i];
	}


	//-------------------------
	//with placeholders
	//-------------------------
	//non-constant access
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, iterator_v >::type>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return this->__begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, iterator_v >::type>
	end(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return this->__end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline iterator_v begin(placeholders::placeholder<I>)
	{
		return fData[I].begin();
	}

	template<unsigned int I>
	 inline iterator_v end(placeholders::placeholder<I>)
	{
		return fData[I].end();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, reverse_iterator_v >::type>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return this->__rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, reverse_iterator_v >::type>
	rend(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return this->__rend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline reverse_iterator_v rbegin(placeholders::placeholder<I> )
	{
		return std::get<I>(fData).rbegin();
	}

	template<unsigned int I>
	 inline reverse_iterator_v rend(placeholders::placeholder<I> )
	{
		return std::get<I>(fData).rend();
	}

	//constant access
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	end(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline const_iterator_v begin(placeholders::placeholder<I> ) const
	{
		return std::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 inline const_iterator_v end(placeholders::placeholder<I> ) const
	{
		return std::get<I>(fData).cend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	cbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__cbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	cend(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__cend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline const_iterator_v cbegin(placeholders::placeholder<I> )  const
	{
		return std::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 inline const_iterator_v cend(placeholders::placeholder<I>  ) const
	{
		return std::get<I>(fData).cend();
	}


	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	rend(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__rend(c1, c2, cn...);
	}


	template<unsigned int I>
	 inline const_reverse_iterator_v rbegin(placeholders::placeholder<I> ) const
	{
		return std::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline const_reverse_iterator_v rend(placeholders::placeholder<I>  ) const
	{
		return std::get<I>(fData).crend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	crbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__crbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline  HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	crend(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return this->__crend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline const_reverse_iterator_v crbegin(placeholders::placeholder<I> ) const
	{
		return std::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline const_reverse_iterator_v crend(placeholders::placeholder<I>  ) const
	{
		return std::get<I>(fData).crend();
	}

	template<unsigned int I>
	inline const column_type& column(placeholders::placeholder<I>  ) const
	{
		return std::get<I>(fData);
	}
	//
	template<typename Functor>
	 inline caster_iterator<Functor> operator[](Functor const& caster)
	{	return begin(caster) ;	}

	//
	template<unsigned int I>
	 inline iterator_v
	operator[](placeholders::placeholder<I> index)
	{	return begin(index) ;	}

	template<unsigned int I>
	 inline const_iterator_v
	operator[](placeholders::placeholder<I> index) const
	{	return cbegin(index); }


	//
	 inline reference operator[](size_t n)
	{	return begin()[n] ;	}

	 inline const_reference operator[](size_t n) const
	{	return cbegin()[n]; }

private:

	//__________________________________________
	// caster accessors
	template<typename Functor>
	 inline caster_iterator<Functor> __caster_begin( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->begin(), caster);
	}

	template<typename Functor>
	 inline caster_iterator<Functor> __caster_end( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->end(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rbegin( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rbegin(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rend( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rend(), caster);
	}
	//__________________________________________
	// pop_back
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__pop_back(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__pop_back()
	{
		std::get<I>(fData).pop_back();
		__pop_back<I + 1>();
	}

	//__________________________________________
	// resize
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__resize(size_type){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__resize(size_type n)
	{
		std::get<I>(fData).resize(n);
		__resize<I + 1>(n);
	}

	//__________________________________________
	// push_back
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back( value_type const& ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back( value_type const& value )
	{
		std::get<I>(fData).push_back( HYDRA_EXTERNAL_NS::thrust::get<I>(value) );
		__push_back<I + 1>( value );
	}

	//__________________________________________
	// clear
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__clear(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__clear( )
	{
		std::get<I>(fData).clear();
		__clear<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__shrink_to_fit(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__shrink_to_fit( )
	{
		std::get<I>(fData).shrink_to_fit();
		__shrink_to_fit<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__reserve(size_type ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__reserve(size_type size )
	{
		std::get<I>(fData).reserve(size);
		__reserve<I + 1>(size);
	}

	//__________________________________________
	// erase
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ){ }

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void>::type
	__erase_helper(size_type position )
	{
		std::get<I>(fData).erase(
				std::get<I>(fData).begin()+position);
		__erase_helper<I+1>(position);
	}

	 inline iterator __erase(size_type position )
	{
		__erase_helper(position);

		return begin() + position;
	}

	//__________________________________________
	// erase

	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ,  size_type ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void>::type
	__erase_helper( size_type first_position,  size_type last_position)
	{
		std::get<I>(fData).erase(
				std::get<I>(fData).begin() + first_position,
				std::get<I>(fData).begin() + last_position );

		__erase_helper<I+1>(first_position, last_position);
	}

	 inline iterator __erase( size_type first_position,  size_type last_position )
	{
		__erase_helper( first_position, last_position );

		return begin() + first_position;
	}

	//__________________________________________
	// insert
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type ,  const value_type&){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position,  const value_type &x)
	{
		std::get<I>(fData).insert(
				std::get<I>(fData).begin()+position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );

		__insert_helper<I+1>(position,x);
	}

	 inline iterator __insert(size_type position,  const value_type &x )
	{
		__insert_helper(position, x );

		return begin()+position;
	}

	//__________________________________________
	// insert
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type , size_type , const value_type&){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position, size_type n, const value_type &x)
	{
		std::get<I>(fData).insert(
				std::get<I>(fData).begin() + position, n,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );
	}

	 inline iterator __insert(size_t position, size_type n,  const value_type &x )
	{
		__insert_helper(position ,  n, x );
		return begin()+position+n;
	}


	//__________________________________________
	// insert
	template<size_t I,typename InputIterator >
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert(size_type, InputIterator const&, InputIterator const& ){}

	template<size_t I=0,typename InputIterator >
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert(size_type position, InputIterator const& first, InputIterator const& last  )
	{
		std::get<I>(fData).insert(std::get<I>(fData).begin() + position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(first),
				HYDRA_EXTERNAL_NS::thrust::get<I>(last) );

		__insert<I + 1>( position,  first, last );
	}

	//__________________________________________
	//front
	 inline reference __front()
	{
		return *(begin());
	}

	 inline const_reference __front() const
	{
		return *(cbegin());
	}

	//__________________________________________
	//back
	 inline reference __back()
	{
		return *(end()-1);
	}

	 inline const_reference __back() const
	{
		return *(cend()-1);
	}

    //
	// /_____________ Begin ______________\
	// \             -------              /
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, iterator_v >::type>
	 __begin(placeholders::placeholder<I1> , placeholders::placeholder<I2> , placeholders::placeholder<IN>...)
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).begin(),
						 std::get<I2>(fData).begin(),
						 std::get<IN>(fData).begin()...));
	 }


	//begin
	template<size_t ...I>
	 inline iterator __begin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).begin()...) );
	}

	 inline iterator __begin(){
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	__begin(placeholders::placeholder<I1> ,	placeholders::placeholder<I2> ,	placeholders::placeholder<IN>...) const
			{
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I1>(fData).begin(),
						std::get<I2>(fData).begin(),
						std::get<IN>(fData).begin()...));
			}

	template<size_t ...I>
	 inline const_iterator __begin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).begin()... ) );
	}



	 inline const_iterator __begin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	 __cbegin(placeholders::placeholder<I1>, placeholders::placeholder<I2>, placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).cbegin(),
						 std::get<I2>(fData).cbegin(),
						 std::get<IN>(fData).cbegin()...));
			 }


	template<size_t ...I>
	 inline const_iterator __cbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).cbegin() ... )	);
	}

	 inline const_iterator __cbegin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ End ______________
	//end
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, iterator_v >::type>
	 	 __end( placeholders::placeholder<I1> , placeholders::placeholder<I2> ,	 placeholders::placeholder<IN>...)
	 	 {
	 		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
	 				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
	 						 std::get<I1>(fData).end(),
	 						 std::get<I2>(fData).end(),
	 						 std::get<IN>(fData).end()...));
	 	 }

	 template<size_t ...I>
	 inline iterator __end_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).end()...) );
	}

	 inline iterator __end(){
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	 __end( placeholders::placeholder<I1> , placeholders::placeholder<I2> , placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).end(),
						 std::get<I2>(fData).end(),
						 std::get<IN>(fData).end()...));
			 }

	template<size_t ...I>
	 inline const_iterator __end_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).end()... ) );
	}

	 inline const_iterator __end() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_iterator_v >::type>
	 __cend( placeholders::placeholder<I1> , placeholders::placeholder<I2> , placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).cend(),
						 std::get<I2>(fData).cend(),
						 std::get<IN>(fData).cend()...));
			 }

	template<size_t ...I>
	 inline const_iterator __cend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).cend() ... )	);
	}

	 inline const_iterator __cend() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse Begin ______________
	//rbegin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, reverse_iterator_v >::type>
	 __rbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2> , placeholders::placeholder<IN>...)
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).rbegin(),
						 std::get<I2>(fData).rbegin(),
						 std::get<IN>(fData).rbegin()...));
			 }

	template<size_t ...I>
	 inline reverse_iterator __rbegin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).rbegin()...) );
	}

	 inline reverse_iterator __rbegin(){
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	 __rbegin(placeholders::placeholder<I1>, placeholders::placeholder<I2> , placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).rbegin(),
						 std::get<I2>(fData).rbegin(),
						 std::get<IN>(fData).rbegin()...));
			 }


	template<size_t ...I>
	 inline const_reverse_iterator __rbegin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).rbegin()... ) );
	}

	 inline const_reverse_iterator __rbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	 __crbegin(placeholders::placeholder<I1> ,
			 placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).crbegin(),
						 std::get<I2>(fData).crbegin(),
						 std::get<IN>(fData).crbegin()...));
			 }

	template<size_t ...I>
	 inline const_reverse_iterator __crbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).crbegin() ... )	);
	}

	 inline const_reverse_iterator __crbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse End ______________
	//rend
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, reverse_iterator_v >::type>
	 __rend(placeholders::placeholder<I1> ,
			 placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...)
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).rend(),
						 std::get<I2>(fData).rend(),
						 std::get<IN>(fData).rend()...));
			 }

	template<size_t ...I>
	 inline reverse_iterator __rend_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).rend()...) );
	}

	 inline reverse_iterator __rend(){
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	 __rend(placeholders::placeholder<I1> ,
			 placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).rend(),
						 std::get<I2>(fData).rend(),
						 std::get<IN>(fData).rend()...));
			 }

	template<size_t ...I>
	 inline const_reverse_iterator __rend_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).rend()... ) );
	}

	 inline const_reverse_iterator __rend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline HYDRA_EXTERNAL_NS::thrust::zip_iterator<typename detail::tuple_type< sizeof...(IN)+2, const_reverse_iterator_v >::type>
	 __crend(placeholders::placeholder<I1> ,
			 placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...) const
			 {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 std::get<I1>(fData).crend(),
						 std::get<I2>(fData).crend(),
						 std::get<IN>(fData).crend()...));
			 }

	template<size_t ...I>
	 inline const_reverse_iterator __crend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						std::get<I>(fData).crend() ... )	);
	}

	 inline const_reverse_iterator __crend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}



	storage_t&&  __move()
	{
		return std::move(fData);
	}


	storage_t fData;

};



/**
 * Return the column ```_I``` of the hydra::multiarray.
 * @param other
 * @return
 */
template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
get(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.column(placeholders::placeholder<I>{}))
{
	return other.column(placeholders::placeholder<I>{});
}

/**
 * Return the column ```_I``` of the hydra::multiarray.
 * @param other
 * @return
 */
template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
get(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.column(placeholders::placeholder<I>{}))
{
	return other.column(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
begin(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
end(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
begin(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
end(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}



template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
rbegin(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
rend(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
rbegin(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename T, size_t N>
inline auto
rend(placeholders::placeholder<I>, multiarray<T,N, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}

template<typename T,size_t N,  hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator==(const multiarray<T, N, hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multiarray<T, N, hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__(
			HYDRA_EXTERNAL_NS::thrust::tuple<
			typename detail::tuple_type<N, T>::type,
			typename detail::tuple_type<N, T>::type
	> const& values)
	{
			return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	return HYDRA_EXTERNAL_NS::thrust::all_of(
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end()  , rhs.end()  ), comparison);
}


template<typename T,size_t N,  hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator!=(const multiarray<T,N,  hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multiarray<T,N,  hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__(
			HYDRA_EXTERNAL_NS::thrust::tuple<
			typename detail::tuple_type<N, T>::type,
			typename detail::tuple_type<N, T>::type> const& values){
		return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	return !(HYDRA_EXTERNAL_NS::thrust::all_of(
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(), rhs.end())
	, comparison));
}

template<hydra::detail::Backend BACKEND, typename T, size_t N, unsigned int...I>
auto columns( multiarray<T,N, detail::BackendPolicy<BACKEND>>const& other, placeholders::placeholder<I>...cls)
-> Range<decltype(std::declval<	multiarray<T,N, detail::BackendPolicy<BACKEND>>const&>().begin(placeholders::placeholder<I>{}...))
>
{

	typedef decltype( other.begin(cls...)) iterator_type;
	return Range<iterator_type>( other.begin(cls...), other.end(cls...));
}

template<  hydra::detail::Backend BACKEND, typename T, size_t N, unsigned int...I>
auto columns( multiarray<T,N, detail::BackendPolicy<BACKEND>>& other, placeholders::placeholder<I>...cls)
-> Range<decltype(std::declval<multiarray<T,N, detail::BackendPolicy<BACKEND>>&&>().begin(placeholders::placeholder<I>{}...))>
{

	typedef decltype( other.begin(cls...)) iterator_type;
	return Range<iterator_type>( other.begin(cls...), other.end(cls...));
}



}  // namespace hydra



#endif /* MULTIARRAY2_H_ */
