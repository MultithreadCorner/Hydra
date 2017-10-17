/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 *  Created on: 10/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR2_H_
#define MULTIVECTOR2_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/Caster.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/logical.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <array>


namespace hydra {

template<typename T, typename BACKEND>
class multivector2;

/**
 * @brief This class implements storage in SoA layouts for
 * table where all elements have the same type.
 */
template<typename ...T, hydra::detail::Backend BACKEND>
class multivector2< HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	constexpr static size_t N = sizeof...(T);

	//Useful aliases
	template<typename Type>
	using vector = typename system_t::template container<Type>;

	template<typename Type>
	using pointer_v    = typename system_t::template container<Type>::pointer;

	template<typename Type>
	using iterator_v   = typename system_t::template container<Type>::iterator;

	template<typename Type>
	using const_iterator_v   = typename system_t::template container<Type>::const_iterator;

	template<typename Type>
	using reverse_iterator_v   = typename system_t::template container<Type>::reverse_iterator;

	template<typename Type>
	using const_reverse_iterator_v   = typename system_t::template container<Type>::const_reverse_iterator;

	template<typename Type>
	using reference_v = typename system_t::template container<Type>::reference;

	template<typename Type>
	using const_reference_v = typename system_t::template container<Type>::const_reference;

	template<typename Type>
	using value_type_v = typename system_t::template container<Type>::value_type;



public:

	typedef HYDRA_EXTERNAL_NS::thrust::tuple< vector<T>...	  	 > 	storage_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< pointer_v<T>... 	 > 	pointer_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< value_type_v<T>... > 	value_type_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< iterator_v<T>...   >  iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_iterator_v<T>...  > 	const_iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< reverse_iterator_v<T>...> 	reverse_iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_reverse_iterator_v<T>... > 	const_reverse_iterator_t;

	typedef HYDRA_EXTERNAL_NS::thrust::tuple< reference_v<T>...		 > 	reference_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_reference_v<T>...> 	const_reference_t;

	//zip iterator
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<iterator_t>		     		 iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_iterator_t>	 		 const_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<reverse_iterator_t>	 		 reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_reverse_iterator_t>	 const_reverse_iterator;

	 //stl-like typedefs
	 typedef size_t size_type;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_iterator>::reference const_reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;




	multivector2():
		fData(storage_t())
	{ };

	multivector2(size_t n){
		fData = data_t();
		__resize(n);
	};

	multivector2(size_t n, value_type const& value){
		fData = data_t();
		__resize(n);
		HYDRA_EXTERNAL_NS::thrust::fill(begin(), end(), value );
	};

	multivector2(multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other )
	{
		fData = data_t();
		__resize(other.size());
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(i), other.end(i), begin());
	}

	multivector2(multivector2<T...,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.__move())
	{}

	template< hydra::detail::Backend BACKEND2>
	multivector2(multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND2>> const& other )
	{
		fData = data_t();

		__resize(other.size());

		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
	}

	template< typename Iterator>
	multivector2(Iterator first, Iterator last )
	{
		fData = data_t();

		__resize( HYDRA_EXTERNAL_NS::thrust::distance(first, last) );

		HYDRA_EXTERNAL_NS::thrust::copy(first, last, begin());

	}

	multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(this==&other) return *this;

		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());

		return *this;
	}

	multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this == &other) return *this;
		this->fData = other.__move();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector2<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND2> > const& other )
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

	size_type size() const
	{
		return HYDRA_EXTERNAL_NS::thrust::distance( begin(), endl() );
	}

	size_type capacity() const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<0>(fData).capacity();
	}

	bool empty() const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<0>(fData).empty();
	}

	void resize(size_type size)
	{
		__resize(size);
	}


	void clear()
	{
		__clear();
	}

	void shrink_to_fit()
	{
		__shrink_to_fit();
	}

	void reserve(size_type size)
	{
		__reserve(size);
	}

	iterator erase(iterator pos)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		return __erase(position);
	}

	iterator erase(iterator first, iterator last)
	{
		size_type first_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), first);
		size_type last_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), last);

		return __erase( first_position, last_position);
	}

	iterator insert(iterator pos, const value_type &x)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		return __insert(position, x);
	}

	void insert(iterator pos, size_type n, const value_type &x)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		__insert(position, n, x);
	}


	template< typename InputIterator>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::tuple,
		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference< InputIterator >::type >::type >::value ||
	detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references,
		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference< InputIterator >::type>::type >::value, void>::type
	insert(iterator pos, InputIterator first, InputIterator last)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
		__insert(position, first, last);
	}


	reference front()
	{
		return __front();
	}

	const_reference front() const
	{
	   return __front();
	}

	reference back()
	{
		return __back();
	}

	const_reference back() const
	{
		return __back();
	}

	//non-constant access
	iterator begin()
	{
		return __begin();
	}

	iterator end()
	{
		return __end();
	}

	//constant automatic conversion access
	template<typename Functor>
	auto begin( Functor const& caster ) const
	-> decltype( __caster_begin(caster) )
	{
		return __caster_begin(caster);
	}

	template<typename Functor>
	auto  end( Functor const& caster ) const
	-> decltype( __caster_end(caster) )
	{
		return __caster_end(caster);
	}

	//constant automatic conversion access
	template<typename Functor>
	auto rbegin( Functor const& caster ) const
	-> decltype( __caster_rbegin(caster) )
	{
		return __caster_rbegin(caster);
	}

	template<typename Functor>
	auto  rend( Functor const& caster ) const
	-> decltype( __caster_rend(caster) )
	{
		return __caster_rend(caster);
	}

	//non-constant access
	reverse_iterator rbegin()
	{
		return __rbegin();
	}

	reverse_iterator rend()
	{
		return __rend();
	}

	//constant access
	const_iterator begin() const
	{
		return __begin();
	}

	const_iterator end() const
	{
		return __end();
	}

	const_reverse_iterator rbegin() const
	{
		return __rbegin();
	}

	const_reverse_iterator rend() const
	{
		return __rend();
	}

	const_iterator cbegin() const
	{
		return __cbegin() ;
	}

	const_iterator cend() const
	{
		return 	__cend() ;
	}

	const_reverse_iterator crbegin() const
	{
		return __crbegin() ;
	}

	const_reverse_iterator crend() const
	{
		return 	__crend() ;
	}

	//non-constant access
	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, iterator_t>::type
	begin(placeholders::placeholder<I> const& index)
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, iterator_t>::type
	end(placeholders::placeholder<I> const& index)
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, reverse_iterator_t>::type
	rbegin(placeholders::placeholder<I> const& index)
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, reverse_iterator_t>::type
	rend(placeholders::placeholder<I> const& index)
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend();
	}

	//constant access
	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	begin(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	end(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	cbegin(placeholders::placeholder<I> const& index)  const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	cend(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	rbegin(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	rend(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	crbegin(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin();
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	crend(placeholders::placeholder<I> const& index) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend();
	}

	template<size_t I>
	const typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, storage_t>::type&
	column(size_t i) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData);
	}

	//
	inline	reference_tuple operator[](size_t n)
	{	return begin()[n] ;	}

	inline const_reference_tuple operator[](size_t n) const
	{	return cbegin()[n]; }

private:
	//__________________________________________
	// caster accessors
	template<typename Functor>
	HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
					iterator, typename std::result_of<Functor(value_type)>::type >
	__caster_begin( Functor const& caster ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(value_type)>::type >(begin(), caster);
	}

	template<typename Functor>
	HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
					iterator, typename std::result_of<Functor(value_type)>::type >
	__caster_end( Functor const& caster ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(value_type)>::type >(end(), caster);
	}

	template<typename Functor>
	HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
	iterator, typename std::result_of<Functor(value_type)>::type >
	__caster_rbegin( Functor const& caster ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(value_type)>::type >(rbegin(), caster);
	}

	template<typename Functor>
	HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
	iterator, typename std::result_of<Functor(value_type)>::type >
	__caster_rend( Functor const& caster ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(value_type)>::type >(rend(), caster);
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
		 HYDRA_EXTERNAL_NS::thrust::get<I>(fData).pop_back();
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
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).resize(n);
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
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).push_back( HYDRA_EXTERNAL_NS::thrust::get<I>(value) );
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
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).clear();
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
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).shrink_to_fit();
		__shrink_to_fit<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__reserve(size_type size){}

	template<size_t I=0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__reserve(size_type size )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).reserve(size);
		__reserve<I + 1>(size);
	}

	//__________________________________________
	// erase
	template<size_t ...I>
	inline  iterator __erase_helper( size_type position, detail::index_sequence<I...>)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).erase(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()+position);

		return begin()+position;
	}

	inline iterator __erase(size_type position )
	{
		return __erase_helper(position, detail::make_index_sequence<N> { });
	}

	//__________________________________________
	// erase
	template<size_t ...I>
	inline iterator __erase_helper( size_type first_position,  size_type last_position, detail::index_sequence<I...>)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).erase(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + first_position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + last_position);

		return begin() + firs_position;
	}

	inline iterator __erase( size_type first_position,  size_type last_position )
	{
		return __erase_helper( first_position,last_position, detail::make_index_sequence<N> { });
	}

	//__________________________________________
	// insert
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type position,  const value_type &x, detail::index_sequence<I...>)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()+position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );

		return begin()+position;
	}

	inline iterator __insert(size_type position,  const value_type &x )
	{
		return __insert_helper(position, x , detail::make_index_sequence<N> { });
	}

	//__________________________________________
	// insert
	template<size_t ...I>
	inline iterator __insert_helper( size_type position, size_type n, const value_type &x, detail::index_sequence<I...>)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + position, n,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );

		return begin()+position+n;
	}

	inline iterator __insert(size_t position, size_type n,  const value_type &x )
	{
		return __insert_helper(position, x  ,  n, detail::make_index_sequence<N> { });
	}


	//__________________________________________
	// insert
	template<size_t I,typename InputIterator >
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert(size_type position, InputIterator first, InputIterator last ){}

	template<size_t I=0,typename InputIterator >
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert(size_type position, InputIterator first, InputIterator last  )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + position,
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


	// _____________ Begin ______________
	//begin
	template<size_t ...I>
	inline iterator __begin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
					HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()...) );
	}

	inline iterator __begin(){
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<size_t ...I>
	inline const_iterator __begin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()... ) );
	}

	inline const_iterator __begin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<size_t ...I>
	inline const_iterator __cbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin() ... )	);
	}

	inline const_iterator __cbegin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ End ______________
	//end
	template<size_t ...I>
	inline iterator __end_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end()...) );
	}

	inline iterator __end(){
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<size_t ...I>
	inline const_iterator __end_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end()... ) );
	}

	inline const_iterator __end() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<size_t ...I>
	inline const_iterator __cend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend() ... )	);
	}

	inline const_iterator __cend() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	/**
	 *
	 *
	 * @return
	 */

	// _____________ Reverse Begin ______________
	//rbegin
	template<size_t ...I>
	inline reverse_iterator __rbegin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
					HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin()...) );
	}

	inline reverse_iterator __rbegin(){
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	template<size_t ...I>
	inline const_reverse_iterator __rbegin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin()... ) );
	}

	inline const_reverse_iterator __rbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	template<size_t ...I>
	inline const_reverse_iterator __crbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin() ... )	);
	}

	inline const_reverse_iterator __crbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse End ______________
	//rend
	template<size_t ...I>
	inline reverse_iterator __rend_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend()...) );
	}

	inline reverse_iterator __rend(){
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	template<size_t ...I>
	inline const_reverse_iterator __rend_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend()... ) );
	}

	inline const_reverse_iterator __rend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	template<size_t ...I>
	inline const_reverse_iterator __crend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend() ... )	);
	}

	inline const_reverse_iterator __crend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}



	storage_t  __move()
	{
		return std::move(fData);
	}

	storage_t fData;


};


#if 0
/**
 * @brief This class implements storage in SoA layouts for
 * table where all elements have the same type.
 */
template< size_t N, typename T, hydra::detail::Backend BACKEND, typename TargetType>
class multivector<N, T, hydra::detail::BackendPolicy<BACKEND>,  TargetType >
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
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<iterator_tuple>		 iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_iterator_tuple>	 const_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<reverse_iterator_tuple>		 reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_reverse_iterator_tuple>	 const_reverse_iterator;


	//stl-like typedefs
	 typedef size_t size_type;

	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::difference_type difference_type;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	 //typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::pointer pointer;
	 typedef typename detail::tuple_type<N,vpointer>::type pointer_tuple;
	 typedef typename detail::tuple_type<N,const_vpointer>::type const_pointer_tuple;
	 typedef std::array<vpointer,N>       pointer_array;
	 typedef std::array<const_vpointer,N> const_pointer_array;

	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_iterator>::reference const_reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<reverse_iterator>::reference reverse_reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_reverse_iterator>::reference const_reverse_reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::iterator_category iterator_category;

	 //cast iterator
	 typedef  HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::Caster< value_type, TargetType>,  iterator>  trans_iterator;
	 typedef  HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::Caster< value_type, TargetType>,  reverse_iterator>  reverse_trans_iterator;

	multivector():
		fData(data_t())
	{

	};

	multivector(size_t n)
	{


		fData = data_t();
		for( size_t i=0; i<N; i++)
			fData[i].resize(n);
	};

	multivector(multivector<N,T,detail::BackendPolicy<BACKEND>> const& other )
	{
		fData = data_t();
		for( size_t i=0; i<N; i++)
			fData[i] = std::move(vector_t(other.begin(i), other.end(i)));
	}

	multivector(multivector<N,T,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.MoveData())
	{}

	template< hydra::detail::Backend BACKEND2>
	multivector(multivector<N,T,detail::BackendPolicy<BACKEND2>> const& other )
	{
		fData = data_t();

		for( size_t i=0; i<N; i++)
			fData[i] = std::move( vector_t( other.begin(i), other.end(i) ) );
	}

	template< typename Iterator>
	multivector(Iterator begin, Iterator end )
	{
		fData = data_t();
		do_copy(begin, end );

	}

	multivector<N,T,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<N,T,detail::BackendPolicy<BACKEND>> const& other )
	{
			if(this==&other) return *this;

			for( size_t i=0; i<N; i++)
				this->fData[i] = std::move(vector_t(other.begin(), other.end()));

			return *this;
	}

	multivector<N,T,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<N,T,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this==&other) return *this;
		this->fData =other.MoveData();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	multivector<N,T,detail::BackendPolicy<BACKEND> >&
	operator=(multivector<N,T,detail::BackendPolicy<BACKEND2> > const& other )
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
	reverse_iterator rbegin();
	reverse_iterator rend();

	//transformed iterator
	trans_iterator tbegin();
	trans_iterator tend();
	reverse_trans_iterator trbegin();
	reverse_trans_iterator trend();


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
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	do_copy(Iterator begin, Iterator end )
	{ }

	template<size_t I=0, typename Iterator>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	do_copy(Iterator begin, Iterator end)
	{

		fData[I] = std::move( vector_t( get<I>(begin.get_iterator_tuple()) ,
				get<I>(end.get_iterator_tuple()) ) );
		do_copy<I + 1>( begin, end);
	}


	//insert
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	do_insert(size_t dist, iterator_tuple& output, value_type const& value)
	{ }

	template<size_t I=0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	do_insert(size_t dist, iterator_tuple& output, value_type const& value)
	{
		get<I>(output) = fData[I].insert(fData[I].begin() + dist, get<I>(value) );
	    do_insert<I + 1>(dist, output,value );
	}

	template<size_t I, template<typename ...> class Tuple, typename ...Iterators>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == sizeof...(Iterators)), void >::type
	do_insert(size_t dist, Tuple<Iterators...> const& first_tuple, Tuple<Iterators...> const& last_tuple)
	{}

	template<size_t I = 0, template<typename ...> class Tuple, typename ...Iterators>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Iterators)), void >::type
	do_insert(size_t dist, Tuple<Iterators...> const& first, Tuple<Iterators...> const& last)
	{
	    fData[I].insert(fData[I].begin() + dist, get<I>(first), get<I>(last) );
	    do_insert<I + 1, Tuple, Iterators... >(dist, first, last );
	}

    //push_back
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	do_push_back(value_type const& value)
	{}

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
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



template<size_t N1, typename T1, hydra::detail::Backend BACKEND1, typename CASTER1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2, typename CASTER2>
bool operator==(const multivector<N1, T1, hydra::detail::BackendPolicy<BACKEND1>, CASTER1 >& lhs,
                const multivector<N2, T2, hydra::detail::BackendPolicy<BACKEND2>, CASTER2 >& rhs);

template<size_t N1, typename T1, hydra::detail::Backend BACKEND1, typename CASTER1,
         size_t N2, typename T2, hydra::detail::Backend BACKEND2, typename CASTER2>
bool operator!=(const multivector<N1, T1, hydra::detail::BackendPolicy<BACKEND1> , CASTER1 >& lhs,
                const multivector<N2, T2, hydra::detail::BackendPolicy<BACKEND2>, CASTER2 >& rhs);

#endif

}  // namespace hydra

//#include<hydra/detail/multivector2.inl>




#endif /* MULTIVECTOR_H_ */
