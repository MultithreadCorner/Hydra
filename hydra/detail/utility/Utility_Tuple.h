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
 * Utility_Tuple.h
 *
 *  Created on: 20/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef UTILITY_TUPLE_H_
#define UTILITY_TUPLE_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/Parameter.h>

//thrust
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>

#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>

#include <type_traits>
#include <array>

namespace hydra {

	namespace detail {

	//---------------------------------------
	// get the type of a tuple with a given the type and the number of elements
    /*
	template <size_t N, typename T>
	struct tuple_type {

		typedef  decltype(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::tuple<T>(),
				typename tuple_type<N-1, T>::type())) type;
	};

	template <typename T>
	struct tuple_type<0, T> {
		typedef decltype(HYDRA_EXTERNAL_NS::thrust::tuple<>()) type;
	};
    */

	//-----------------
	template<typename Tuple1,typename Tuple2>
	struct tuple_cat_type;


	template<template<typename ...>class Tuple, typename ...T1,typename ...T2>
	struct tuple_cat_type<Tuple<T1...>, Tuple<T2...> >
	{
		typedef Tuple<T1..., T2...> type;
	};

	//-----------------
	template <size_t N,template<typename> class COM>
	struct CompareTuples{

		template<typename Tuple1, typename Tuple2>
		__hydra_host__ __hydra_device__
		GBool_t operator( )(Tuple1 t1, Tuple2 t2 )
		{
			return COM< typename HYDRA_EXTERNAL_NS::thrust::tuple_element<N, Tuple1>::type>()(HYDRA_EXTERNAL_NS::thrust::get<N>(t1) , HYDRA_EXTERNAL_NS::thrust::get<N>(t2) );
		}

	};

	template <size_t N, typename T>
	struct tuple_type {
		typedef typename repeat<T, N, HYDRA_EXTERNAL_NS::thrust::tuple>::type type;
	};

	template <size_t N, typename T>
	struct references_tuple_type {
		typedef typename repeat<T, N, HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references>::type type;
	};

	//--------------------------------------
	template<typename T, typename C, size_t N, size_t I>
	__hydra_host__  __hydra_device__ inline
	typename std::enable_if< I==N,void>::type
	max_helper( T const&, C&, size_t&){}

	template<typename T, typename C, size_t N, size_t I=1>
	__hydra_host__  __hydra_device__ inline
	typename std::enable_if< I < N, void>::type
	max_helper( T const& tuple, C& max_value, size_t& max_index){

	 max_index = max_value > HYDRA_EXTERNAL_NS::thrust::get<I>(tuple) ? max_index : I;
	 max_value = max_value > HYDRA_EXTERNAL_NS::thrust::get<I>(tuple) ? max_value : HYDRA_EXTERNAL_NS::thrust::get<I>(tuple);

	 max_helper<T, C, N, I+1>(tuple, max_value, max_index);
	}

	template<typename ...T>
	__hydra_host__  __hydra_device__ inline
	size_t max(HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& tuple){

	typedef typename std::common_type<T...>::type C;

	 size_t i=0;
	 C max_value = HYDRA_EXTERNAL_NS::thrust::get<0>(tuple);

	 max_helper<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,C,sizeof...(T)>(tuple, max_value, i);

	 return i;

	}

	//--------------------------------------
	//get zip iterator
	template<typename Iterator, size_t ...Index>
	__hydra_host__ inline
	auto get_zip_iterator_helper(std::array<Iterator, sizeof ...(Index)>const & array_of_iterators,
			index_sequence<Index...>)
	-> decltype( HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::make_tuple( array_of_iterators[Index]...)) )
	{
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::make_tuple( array_of_iterators[Index]...) );
	}


	template<typename IteratorHead, typename IteratorTail, size_t ...Index>
	__hydra_host__ inline auto
	get_zip_iterator_helper(IteratorHead head,
			std::array<IteratorTail, sizeof ...(Index)>const & array_of_iterators,
			index_sequence<Index...>)
	-> decltype( HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::make_tuple(head, array_of_iterators[Index]...)) )
	{
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(head , array_of_iterators[Index]...));
	}

	template<typename Iterator, size_t N>
	__hydra_host__ inline
	auto get_zip_iterator(std::array<Iterator, N>const & array_of_iterators)
	-> decltype( get_zip_iterator_helper( array_of_iterators , make_index_sequence<N> { }) )
	{
		return get_zip_iterator_helper( array_of_iterators , make_index_sequence<N> { } );

	}

	template<typename IteratorHead, typename IteratorTail, size_t N>
	__hydra_host__ inline
	auto get_zip_iterator(IteratorHead head, std::array<IteratorTail, N>const & array_of_iterators)
	-> decltype( get_zip_iterator_helper(head, array_of_iterators , make_index_sequence<N> { }) )
	{
		return get_zip_iterator_helper(head, array_of_iterators , make_index_sequence<N> { } );

	}
	//----------------------------------------
	// make a tuple of references to a existing tuple
	template<typename ...T, size_t... I>
	auto make_rtuple_helper(HYDRA_EXTERNAL_NS::thrust::tuple<T...>& t , index_sequence<I...>)
	-> HYDRA_EXTERNAL_NS::thrust::tuple<T&...>
	{ return HYDRA_EXTERNAL_NS::thrust::tie(HYDRA_EXTERNAL_NS::thrust::get<I>(t)...) ;}

	template<typename ...T>
	auto  make_rtuple( HYDRA_EXTERNAL_NS::thrust::tuple<T...>& t )
	-> HYDRA_EXTERNAL_NS::thrust::tuple<T&...>
	{
		return make_rtuple_helper( t, make_index_sequence<sizeof...(T)> {});
	}

	//---------------------------------------------
	// get a reference to a tuple object by index
	template<typename R, typename T, size_t I>
	__hydra_host__  __hydra_device__ inline
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == HYDRA_EXTERNAL_NS::thrust::tuple_size<T>::value), void>::type
	_get_element(const size_t , T& , R*&  )
	{ }

	template<typename R, typename T, size_t I=0>
	__hydra_host__  __hydra_device__ inline
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<( I < HYDRA_EXTERNAL_NS::thrust::tuple_size<T>::value), void>::type
	_get_element(const size_t index, T& t, R*& ptr )
	{

	    index==I ? ptr=&HYDRA_EXTERNAL_NS::thrust::get<I>(t):0;

	    _get_element<R,T,I+1>(index, t, ptr);
	}

	template<typename R, typename ...T>
	__hydra_host__  __hydra_device__ inline
	R& get_element(const size_t index, HYDRA_EXTERNAL_NS::thrust::tuple<T...>& t)
	{
	    R* ptr;
	    _get_element( index, t, ptr );

	    return *ptr;


	}

	//----------------------------------------
	template<typename ...T1, typename ...T2, size_t... I1, size_t... I2 >
	__hydra_host__  __hydra_device__ inline
	void split_tuple_helper(HYDRA_EXTERNAL_NS::thrust::tuple<T1...> &t1, HYDRA_EXTERNAL_NS::thrust::tuple<T2...> &t2,
			HYDRA_EXTERNAL_NS::thrust::tuple<T1..., T2...> const& t , index_sequence<I1...>, index_sequence<I2...>)
	{
		t1 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I1>(t)... );
		t2 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I2+ sizeof...(T1)>(t)... );

	}

	template< typename ...T1,  typename ...T2>
	__hydra_host__  __hydra_device__ inline
	void split_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<T1...> &t1, HYDRA_EXTERNAL_NS::thrust::tuple<T2...> &t2,
			HYDRA_EXTERNAL_NS::thrust::tuple<T1..., T2...> const& t)
	{
	    return split_tuple_helper(t1, t2, t ,
	    		make_index_sequence<sizeof...(T1)>{},
	    		make_index_sequence<sizeof...(T2)>{} );
	}

	//----------------------------------------
	template<typename ...T, size_t... I1, size_t... I2 >
	__hydra_host__  __hydra_device__ inline
	auto split_tuple_helper(HYDRA_EXTERNAL_NS::thrust::tuple<T...> &t, index_sequence<I1...>, index_sequence<I2...>)
	-> decltype( HYDRA_EXTERNAL_NS::thrust::make_pair(HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I1>(t)... ), HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I2+ + sizeof...(I1)>(t)... ) ) )
	{
		auto t1 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I1>(t)... );
		auto t2 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I2+ sizeof...(I1)>(t)... );

		return HYDRA_EXTERNAL_NS::thrust::make_pair(t1, t2);
	}

	template< size_t N, typename ...T>
	__hydra_host__  __hydra_device__ inline
	auto split_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<T...>& t)
	-> decltype( split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} ) )
	{
	    return split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} );
	}

	template<typename ...T, size_t... I1, size_t... I2 >
	__hydra_host__  __hydra_device__ inline
	auto split_tuple_helper(HYDRA_EXTERNAL_NS::thrust::tuple<T...>  const& t, index_sequence<I1...>, index_sequence<I2...>)
	-> decltype( HYDRA_EXTERNAL_NS::thrust::make_pair(HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I1>(t)... ), HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I2+ + sizeof...(I1)>(t)... ) ) )
	{
		auto t1 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I1>(t)... );
		auto t2 = HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<I2+ sizeof...(I1)>(t)... );

		return HYDRA_EXTERNAL_NS::thrust::make_pair(t1, t2);
	}

	template< size_t N, typename ...T>
	__hydra_host__  __hydra_device__ inline
	auto split_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& t)
	-> decltype( split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} ) )
	{
		return split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} );
	}



	//----------------------------------------
	template<typename Head,  typename ...Tail,  size_t... Is >
	__hydra_host__ __hydra_device__
	auto dropFirstHelper(HYDRA_EXTERNAL_NS::thrust::tuple<Head, Tail...> const& t  , index_sequence<Is...> )
	-> HYDRA_EXTERNAL_NS::thrust::tuple<Tail...>
	{
		return HYDRA_EXTERNAL_NS::thrust::tie( HYDRA_EXTERNAL_NS::thrust::get<Is+1>(t)... );
	}

	template< typename Head,  typename ...Tail>
	__hydra_host__ __hydra_device__
	auto dropFirst( HYDRA_EXTERNAL_NS::thrust::tuple<Head, Tail...> const& t)
	-> decltype(dropFirstHelper( t, make_index_sequence<sizeof...(Tail) >{} ))
	{
		return dropFirstHelper(t , make_index_sequence<sizeof ...(Tail)>{} );
	}

	//----------------------------------------
	template<typename T, typename Head,  typename ...Tail,  size_t... Is >
	auto changeFirstHelper(T& new_first, HYDRA_EXTERNAL_NS::thrust::tuple<Head, Tail...>  const& t  , index_sequence<Is...> )
	-> HYDRA_EXTERNAL_NS::thrust::tuple<T,Tail...>
	{
		return HYDRA_EXTERNAL_NS::thrust::make_tuple(new_first, HYDRA_EXTERNAL_NS::thrust::get<Is+1>(t)... );
	}

	template<typename T, typename Head,  typename ...Tail>
	auto changeFirst(T& new_first, HYDRA_EXTERNAL_NS::thrust::tuple<Head, Tail...>  const& t)
	-> decltype(changeFirstHelper(new_first, t , make_index_sequence<sizeof ...(Tail)>{} ))
	{
		return changeFirstHelper(new_first, t , make_index_sequence<sizeof ...(Tail)>{} );
	}


	//----------------------------------------
	template<typename T, size_t I>
	T passthrough(T&& value){
		return std::move( std::forward<T>(value));
	}


	//make a homogeneous tuple with same value in all elements
	template <typename T,  size_t... Is >
	auto make_tuple_helper(T&& value, index_sequence<Is...>)
	-> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(passthrough<T,Is>(std::forward<T>(value))...))
	{
		return HYDRA_EXTERNAL_NS::thrust::make_tuple(passthrough<T,Is>(std::forward<T>(value))...);
	}


	template <  size_t N, typename T,typename TupleType=typename tuple_type<N,T>::type >
	TupleType make_tuple(T&& value)
	{

		return make_tuple_helper( std::forward<T>(value), make_index_sequence<N>{});
	}


	//---------------------------------------
	// convert a std::array to tuple.
	template <typename T,  size_t... Is>
	auto arrayToTupleHelper(std::array<T, sizeof...(Is)>const & Array, index_sequence<Is...> )
	-> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(Array[Is]...))
	{
		return HYDRA_EXTERNAL_NS::thrust::make_tuple(Array[Is]...);
	}

	template <typename T,  size_t N>
	auto arrayToTuple(std::array<T, N>const& Array)
	-> decltype(arrayToTupleHelper(Array, make_index_sequence<N>{}))
	{
		return arrayToTupleHelper(Array, make_index_sequence<N>{} );
	}

	//---------------------------------------
	// convert a generic array to tuple
	template <typename T, size_t... Indices>
	__hydra_host__  __hydra_device__
	inline auto arrayToTupleHelper( T* Array,
			index_sequence<Indices...>)
	-> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(Array[Indices]...))
	{
		return HYDRA_EXTERNAL_NS::thrust::make_tuple(Array[Indices]...);
	}

	template <typename T, size_t N>
	__hydra_host__  __hydra_device__
	inline auto arrayToTuple(T* Array)
	-> decltype( arrayToTupleHelper(Array, make_index_sequence<N>{ }))
	{
		return arrayToTupleHelper(Array, make_index_sequence<N>{ });
	}

	//---------------------------------------
	// set a generic array with tuple values
	template<size_t I = 0,  typename ArrayType, typename FistType, typename ...OtherTypes>
	__hydra_host__  __hydra_device__ inline
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	are_all_same<FistType,OtherTypes...>::value, void>::type
	assignArrayToTuple(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> &,  ArrayType (&)[sizeof...(OtherTypes)+1])
	{}

	template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	__hydra_host__  __hydra_device__
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	are_all_same<FistType,OtherTypes...>::value, void >::type
	assignArrayToTuple(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> & t, ArrayType (&Array)[sizeof...(OtherTypes)+1] )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(t) = (typename ArrayType::args_type) Array[I];
		assignArrayToTuple<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	}

	//---------------------------------------
	// set a std::array with tuple values
	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	              are_all_same<FistType,OtherTypes...>::value, void>::type
	 tupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const&, std::array<FistType, sizeof...(OtherTypes) + 1>&)
	 {}

	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 tupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const& t, std::array<FistType, sizeof...(OtherTypes) + 1>& Array)
	 {

		 Array[I] = HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 tupleToArray<I + 1,FistType, OtherTypes... >( t, Array);
	 }


	 //---------------------------------------
	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const&,
			 ArrayType (&)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType&, OtherTypes&...> const&,
			 ArrayType (&)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType&, OtherTypes&...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references<FistType, OtherTypes...> const&,
			 ArrayType (&)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references<FistType, OtherTypes...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 //---------------------------------------
	 // set a generic array with tuple values
	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	              are_all_same<FistType,OtherTypes...>::value, void>::type
	 tupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const &,  typename std::remove_reference<FistType>::type*)
	 {}

	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	           are_all_same<FistType,OtherTypes...>::value, void >::type
	 tupleToArray(HYDRA_EXTERNAL_NS::thrust::tuple<FistType, OtherTypes...> const & t,  typename std::remove_reference<FistType>::type* Array)
	 {

		 Array[I] = HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 tupleToArray<I + 1,FistType, OtherTypes... >( t, Array);
	 }



	 //----------------------------------------------------------
	 template<typename A, typename Tp, size_t I>
	 struct is_homogeneous_base:
	 std::conditional<HYDRA_EXTERNAL_NS::thrust::detail::is_same<
	 typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	     typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<A>::type
	 >::type,
	 typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	   typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
	      typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I-1, Tp>::type
	   >::type
	 >::type>::value, is_homogeneous_base<A, Tp, I-1>,
	 HYDRA_EXTERNAL_NS::thrust::detail::false_type>::type{ };

	 template<typename A, typename Tp>
	 struct is_homogeneous_base<A, Tp, 0>: HYDRA_EXTERNAL_NS::thrust::detail::true_type{ };

	 template<typename A, typename Tuple>
	 struct is_homogeneous:  is_homogeneous_base<A, Tuple, HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value > {};

	 // set array of pointers to point to the tuple elements
	 template<size_t I= 0, typename Array_Type, typename T>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == HYDRA_EXTERNAL_NS::thrust::tuple_size<T>::value) && is_homogeneous<Array_Type, T>::value, void>::type
	 set_ptrs_to_tuple(T& , Array_Type** )
	 {}

	 template<size_t I = 0, typename Array_Type, typename T>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < HYDRA_EXTERNAL_NS::thrust::tuple_size<T>::value) && is_homogeneous<Array_Type, T>::value, void>::type
	 set_ptrs_to_tuple(T& t, Array_Type** Array)
	 {
		 Array[I] =  &HYDRA_EXTERNAL_NS::thrust::get<I>(t);
		 set_ptrs_to_tuple<I + 1,Array_Type,T>(t, Array);
	 }

	 //---------------------------------------
	 //get a element of a given tuple by index. value is stored in x


     template<typename T1, typename  T2>
     void ptr_setter( T1*& ptr, typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<HYDRA_EXTERNAL_NS::thrust::detail::is_same<T1,T2>::value, T2 >::type* el ){ ptr=el; }

     template<typename T1, typename  T2>
     void ptr_setter( T1*&, typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<!HYDRA_EXTERNAL_NS::thrust::detail::is_same<T1,T2>::value, T2 >::type* ){  }


     template<unsigned int I, typename Ptr, typename ...Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_ptr_to_tuple_element(const int, std::tuple<Tp...>&, Ptr*&)
	 {	 }

	 template<unsigned int I = 0, typename Ptr, typename ...Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 set_ptr_to_tuple_element(const  int index, std::tuple<Tp...> & t, Ptr*& ptr)
	 {
		 if(index == I ) ptr_setter<Ptr, typename std::tuple_element<I, std::tuple<Tp...>>::type>( ptr, &std::get<I>(t));

		 set_ptr_to_tuple_element<I + 1, Ptr, Tp...>(index, t, ptr);

	 }

     //---------------------------------------
	 //get a element of a given tuple by index. value is stored in x
	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 get_tuple_element(const int, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, T& )
	 {}

	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 get_tuple_element(const  int index, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, T& x)
	 {
		 if(index == I)x=T(  HYDRA_EXTERNAL_NS::thrust::get<I>(t));
		 get_tuple_element<I + 1, T, Tp...>(index, t,x );
	 }


	 //---------------------------------------
	 //set a element of a given tuple by index to value  x
	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_tuple_element(const int, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, T& )
	 {}

	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 set_tuple_element(const  int index, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t, const T& x)
	 {
		 if(index == I) HYDRA_EXTERNAL_NS::thrust::get<I>(t)=x;
		 get_tuple_element<I + 1, T, Tp...>(index, t,x );
	 }

	 //---------------------------------------
	 // evaluate a void functor taking as argument
	 // a given tuple element
	 template<unsigned int I = 0, typename FuncT, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 eval_on_tuple_element(int, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, FuncT const&)
	 {}

	 template<unsigned int I = 0, typename FuncT, typename ...Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 eval_on_tuple_element(int index, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, FuncT const& f)
	 {
		 if (index == 0) std::forward<FuncT const>(f)(HYDRA_EXTERNAL_NS::thrust::get<I>(t));
		 eval_on_tuple_element<I + 1, FuncT, Tp...>(index-1, t, std::forward<FuncT const>(f));
	 }

	 //--------------------------------------
	 // given a tuple of functors, evaluate a
	 // element taking as argument ArgType const&
	 // arg and return on r
	 //--------------------------------------
	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 eval_tuple_element(Return_Type&, int, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, ArgType const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 eval_tuple_element(Return_Type& r, int index, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, ArgType const& arg)
	 {
		 if (index == 0)
		 {
			 r = (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(std::forward<ArgType const>(arg));
			 return;
		 }
		 eval_tuple_element<I + 1, Return_Type,ArgType, Tp...>( r , index-1, t,arg );
	 }

	 //----------------------------------------------------------
	 // given a tuple of functors, evaluate and accumulate
	 // element taking as argument ArgType const&
	 // arg and return on r
	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&&, ArgType&&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t, ArgType&& arg)
	 {
		 r = r+ (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>( std::forward<HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>>(t))(std::forward<ArgType>(arg));
		 sum_tuple<I + 1, Return_Type, ArgType, Tp...>( r , std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t), std::forward<ArgType>(arg) );
	 }


	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, ArgType const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>const& t, ArgType const& arg)
	 {
		 r = r+ (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg);
		 sum_tuple<I + 1, Return_Type, ArgType, Tp...>( r , t,arg );
	 }


	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple2(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&&, ArgType1&&, ArgType2&& )
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple2(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t, ArgType1&& arg1, ArgType2&& arg2)
	 {
		 r = r + (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg1, arg2);
		 sum_tuple2<I + 1, Return_Type, ArgType1, ArgType2, Tp...>( r , std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t)
				 , std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2));
	 }

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple2(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>const&, ArgType1 const&, ArgType2 const& )
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple2(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, ArgType1 const& arg1, ArgType2 const& arg2)
	 {
		 r = r + (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg1, arg2);
		 sum_tuple2<I + 1, Return_Type, ArgType1, ArgType2, Tp...>( r , t,arg1, arg2);
	 }


	 //----------------------------------------------------------
	 // given a tuple of functors, evaluate and multiply
	 // element taking as argument ArgType &
	 // arg and return on r
	 template<size_t I = 0, typename Return_Type,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 multiply_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 multiply_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>const& t)
	 {
		 r = r*( (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t));
		 multiply_tuple<I + 1, Return_Type, Tp...>( r , t );
	 }

	 template<size_t I = 0, typename Return_Type,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 multiply_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 multiply_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t)
	 {
		 r = r*( (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t)));
		 multiply_tuple<I + 1, Return_Type, Tp...>( r , std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t) );
	 }

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, ArgType const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 product_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, ArgType const& arg)
	 {
		 r = r*( (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg));
		 product_tuple<I + 1, Return_Type, ArgType, Tp...>( r , t,arg );
	 }


	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> &&, ArgType&&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 product_tuple(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t, ArgType&& arg)
	 {
		 r = r*( (Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t))(std::forward<ArgType>(arg)));
		 product_tuple<I + 1, Return_Type, ArgType, Tp...>( r , std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t), std::forward<ArgType>(arg) );
	 }

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple2(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, ArgType1  const&, ArgType2 const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < (sizeof...(Tp))),void >::type
	 product_tuple2(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, ArgType1 const& arg1, ArgType2 const& arg2)
	 {
		 r = r*((Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg1, arg2));
		 product_tuple2<I + 1, Return_Type, ArgType1, ArgType2,  Tp...>( r , t,arg1, arg2 );
	 }


	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple2(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&&, ArgType1&&, ArgType2&&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < (sizeof...(Tp))),void >::type
	 product_tuple2(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t, ArgType1&& arg1, ArgType2&& arg2)
	 {
		 r = r*((Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t))(std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2)));
		 product_tuple2<I + 1, Return_Type, ArgType1, ArgType2,  Tp...>( r ,
				 std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t), std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2)) ;
	 }

	 template<size_t I = 0, size_t N,typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == N, void>::type
	 product_tuple3(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const&, ArgType1 const&, ArgType2 const&)
	 {}

	 template<size_t I = 0,size_t N, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N),void >::type
	 product_tuple3(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t, ArgType1 const& arg1, ArgType2 const& arg2)
	 {
		 r = r*((Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(t)(arg1, arg2));
		 product_tuple3<I + 1,N, Return_Type, ArgType1, ArgType2,  Tp...>( r , t,arg1, arg2 );
	 }


	 template<size_t I = 0, size_t N,typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == N, void>::type
	 product_tuple3(Return_Type&, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&&, ArgType1&&, ArgType2&&)
	 {}

	 template<size_t I = 0,size_t N, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N),void >::type
	 product_tuple3(Return_Type& r, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&& t, ArgType1&& arg1, ArgType2&& arg2)
	 {
		 r = r*((Return_Type) HYDRA_EXTERNAL_NS::thrust::get<I>(std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t))(std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2)));
		 product_tuple3<I + 1,N, Return_Type, ArgType1, ArgType2,  Tp...>( r ,
				 std::forward< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> >(t), std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2));
	 }

/*
	 //evaluate a tuple of functors and return a tuple of results
	 template< typename Tup, typename ArgType, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_helper(size_t n, ArgType* x, Tup& tup, index_sequence<index...>)
	 -> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(n,x)...))
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(n,x)...);
	 }

	 template< typename Tup, typename ArgType>
	 __hydra_host__  __hydra_device__
	 inline auto invoke(size_t n, ArgType* x, Tup& tup)
	 -> decltype(invoke_helper2(n, x, tup, make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value> { }))
	 {
		 constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
		 return invoke_helper2(n, x, tup, make_index_sequence<Size> { });
	 }
*/
	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_normalized_helper( ArgType const& x, Tup const& tup, index_sequence<index...>)
	 -> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x)...))
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup).GetNorm()*
				 HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x)...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType>
	 __hydra_host__  __hydra_device__
	 inline auto invoke_normalized(ArgType const& x, Tup const& tup)
	 -> decltype(invoke_helper(x, tup, make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value> { }))
	 {
		 constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
		 return invoke_normalized_helper( x, tup, make_index_sequence<Size> { });
	 }

	 //evaluate a tuple of functors and return a tuple of results
	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_normalized_helper( ArgType&& x, Tup&& tup, index_sequence<index...>)
	 -> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup))(std::forward<ArgType>(x))...))
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup)).GetNorm()
				 *HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup))(std::forward<ArgType>(x))...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType,
	 	 size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<Tup>::type>::value>
	 __hydra_host__  __hydra_device__
	 inline auto invoke_normalized(ArgType&& x, Tup const& tup)
	 -> decltype(invoke_helper(std::forward<ArgType>(x), std::forward<Tup>(tup), make_index_sequence<N> { }))
	 {
		 //constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
		 return invoke_normalized_helper( std::forward<ArgType>(x), std::forward<Tup>(tup), make_index_sequence<N> { });
	 }



	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_helper( ArgType&& x,  Tup&& tup, index_sequence<index...>)
	 -> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup))(std::forward<ArgType>(x))...))
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup))(std::forward<ArgType>(x))...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType,
	 size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<Tup>::type>::value>
	 __hydra_host__  __hydra_device__
	 inline auto invoke(ArgType&& x,  Tup&& tup)
	 -> decltype(invoke_helper(std::forward<ArgType>(x), std::forward<Tup>(tup), make_index_sequence<N> { }))
	 {
		 return invoke_helper(std::forward<ArgType>(x), std::forward<Tup>(tup), make_index_sequence<N> { });
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_helper( ArgType const& x, Tup const& tup, index_sequence<index...>)
	 -> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x)...))
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x)...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType,
	  size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<Tup>::type>::value>
	 __hydra_host__  __hydra_device__
	 inline auto invoke(ArgType const& x, Tup const& tup)
	 -> decltype(invoke_helper(x, tup, make_index_sequence<N> { }))
	 {
		 //constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
		 return invoke_helper( x, tup, make_index_sequence<N> { });
	 }



	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template<typename Tup, typename ArgType1, typename ArgType2, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_helper( ArgType1&& x, ArgType2&& y, Tup&& tup, index_sequence<index...>)
	 -> decltype( HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>( std::forward<Tup>(tup))(std::forward<ArgType1>(x), std::forward<ArgType2>(y))...) )
	 {
		 return  HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(std::forward<Tup>(tup))(std::forward<ArgType1>(x), std::forward<ArgType2>(y))...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType1, typename ArgType2,
	  size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<Tup>::type>::value>
	 __hydra_host__  __hydra_device__
	 inline auto invoke(ArgType1&& x, ArgType2&& y,  Tup&& tup)
	 -> decltype(invoke_helper( std::forward<ArgType1>(x), std::forward<ArgType2>(y),  std::forward<Tup>(tup), make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value> { }) )
	 {
		 return invoke_helper( std::forward<ArgType1>(x), std::forward<ArgType2>(y), std::forward<Tup>(tup), make_index_sequence<N>{});
	 }



	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template<typename Tup, typename ArgType1, typename ArgType2, size_t ... index>
	 __hydra_host__ __hydra_device__
	 inline auto invoke_helper( ArgType1 const& x, ArgType2 const& y, Tup const& tup, index_sequence<index...>)
	 -> decltype( HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x,y)...) )
	 {
		 return  HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(tup)(x,y)...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType1, typename ArgType2>
	 __hydra_host__  __hydra_device__
	 inline auto invoke(ArgType1 const& x, ArgType2 const& y,  Tup const& tup)
	 -> decltype(invoke_helper( x, y, tup, make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value> { }) )
	 {
		 constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
		 return invoke_helper( x, y, tup, make_index_sequence<Size> { });
	 }


	 // set functors in tuple
	 template<size_t I = 0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_functors_in_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&, const std::vector<double>&)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 set_functors_in_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t,  const std::vector<double>& parameters)
	 {
		 HYDRA_EXTERNAL_NS::thrust::get<I>(t).SetParameters(parameters);
		 set_functors_in_tuple<I + 1,Tp...>( t, parameters );
	 }


	 template<size_t I = 0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 print_parameters_in_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 print_parameters_in_tuple(HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t)
	 {
		 HYDRA_EXTERNAL_NS::thrust::get<I>(t).PrintRegisteredParameters();
		 print_parameters_in_tuple<I + 1,Tp...>(t);
	 }

	 template<size_t I=0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 add_parameters_in_tuple(std::vector<hydra::Parameter*>& , HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>&)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 add_parameters_in_tuple(std::vector<hydra::Parameter*>& user_parameters, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t)
	 {
		 HYDRA_EXTERNAL_NS::thrust::get<I>(t).AddUserParameters(user_parameters);
		 add_parameters_in_tuple<I + 1,Tp...>(user_parameters,t);
	 }



	 //extract a element of a ntuple
	 template<typename Type, typename Tuple>
	 __hydra_host__  __hydra_device__
	 inline Type extract(const int idx, const Tuple& t)
	 {
		 Type x;
		 get_tuple_element(idx, t, x);

		 return x;
	 }


	 //accumulate by sum
	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type >
	 __hydra_host__  __hydra_device__
	 inline Return_Type accumulate(ArgType& x, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple(r, t, x);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type >
	 __hydra_host__  __hydra_device__
	 inline Return_Type accumulate(ArgType const& x, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple(r, t, x);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType1, typename ArgType2,
	 typename ... Tp, typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value>::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type accumulate2(ArgType1& x, ArgType2& y, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple2(r, t, x, y);
		 return r;
	 }


	 template<typename Return_Type, typename ArgType1, typename ArgType2,
	 typename ... Tp, typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value>::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type accumulate2(ArgType1 const& x, ArgType2 const& y, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple2(r, t, x, y);
		 return r;
	 }


	 //accumulate by product
	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type product(ArgType const& x, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>  const& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 product_tuple(r, t, x);
		 return r;
	 }

	 //accumulate by product
	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type product(ArgType& x, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> & t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 product_tuple(r, t, x);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType1,typename ArgType2, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type product2(ArgType1 const& x, ArgType2 const& y, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...> const& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 constexpr size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>>::value;
		 product_tuple3<0,N,Return_Type,ArgType1, ArgType2,Tp...>(r, t, x, y);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType1,typename ArgType2, typename ... Tp,
	 typename=typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,Return_Type>::value >::type>
	 __hydra_host__  __hydra_device__
	 inline Return_Type product2(ArgType1& x, ArgType2& y, HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 constexpr size_t N=HYDRA_EXTERNAL_NS::thrust::tuple_size< HYDRA_EXTERNAL_NS::thrust::tuple<Tp...>>::value;
		 product_tuple3<0,N,Return_Type,ArgType1, ArgType2,Tp...>(r, t, x, y);
		 return r;
	 }

	 template<size_t I, typename ...T>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == sizeof...(T)),void >::type
	 add_tuple_values(GReal_t& , HYDRA_EXTERNAL_NS::thrust::tuple<T...> const&)
	 { }

	 template<size_t I=0, typename ...T>
	 __hydra_host__  __hydra_device__
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(T)),void >::type
     add_tuple_values(GReal_t& result, HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& tpl )
	 {
		 result += HYDRA_EXTERNAL_NS::thrust::get<I>(tpl);
		 add_tuple_values<I+1, T...>(result,tpl);
	 }

	 template<typename Tuple, size_t ...I>
	 __hydra_host__  __hydra_device__
	 inline	typename tuple_type<sizeof...(I), GReal_t>::type
	 multiply_array_tuple_helper(GReal_t (&fCoeficients)[sizeof...(I)],Tuple const& tpl,
			 index_sequence<I...>)
	 {
		 return HYDRA_EXTERNAL_NS::thrust::make_tuple(fCoeficients[I]*HYDRA_EXTERNAL_NS::thrust::get<I>(tpl)... );
	 }

	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline	auto multiply_array_tuple(GReal_t (&fCoeficients)[sizeof...(T)],
			HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& tpl)
	-> decltype( multiply_array_tuple_helper(fCoeficients,tpl,make_index_sequence<sizeof...(T)>{}) )
	{
		return multiply_array_tuple_helper(fCoeficients,tpl,make_index_sequence<sizeof...(T)>{});
	}


	}//namespace detail


}//namespace hydra

#endif /* UTILITY_TUPLE_H_ */
