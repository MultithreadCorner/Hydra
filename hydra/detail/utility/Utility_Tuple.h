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
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>

#include <thrust/iterator/detail/tuple_of_iterator_references.h>

#include <type_traits>
#include <array>

namespace hydra {

	namespace detail {

	//---------------------------------------
	// get the type of a tuple with a given the type and the number of elements
    /*
	template <size_t N, typename T>
	struct tuple_type {

		typedef  decltype(thrust::tuple_cat(thrust::tuple<T>(),
				typename tuple_type<N-1, T>::type())) type;
	};

	template <typename T>
	struct tuple_type<0, T> {
		typedef decltype(thrust::tuple<>()) type;
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
		GBool_t operator( )(Tuple1 t1, Tuple2 t2 )
		{
			return COM< typename thrust::tuple_element<N, Tuple1>::type>()(thrust::get<N>(t1) , thrust::get<N>(t2) );
		}
	};

	template <size_t N, typename T>
	struct tuple_type {
		typedef typename repeat<T, N, thrust::tuple>::type type;
	};

	template <size_t N, typename T>
	struct references_tuple_type {
		typedef typename repeat<T, N, thrust::detail::tuple_of_iterator_references>::type type;
	};


	//--------------------------------------
	//get zip iterator
	template<typename Iterator, size_t ...Index>
	__host__ inline
	auto get_zip_iterator_helper(std::array<Iterator, sizeof ...(Index)>const & array_of_iterators,
			index_sequence<Index...>)
	-> decltype( thrust::make_zip_iterator( thrust::make_tuple( array_of_iterators[Index]...)) )
	{
		return thrust::make_zip_iterator( thrust::make_tuple( array_of_iterators[Index]...) );
	}


	template<typename IteratorHead, typename IteratorTail, size_t ...Index>
	__host__ inline auto
	get_zip_iterator_helper(IteratorHead head,
			std::array<IteratorTail, sizeof ...(Index)>const & array_of_iterators,
			index_sequence<Index...>)
	-> decltype( thrust::make_zip_iterator( thrust::make_tuple(head, array_of_iterators[Index]...)) )
	{
		return thrust::make_zip_iterator(thrust::make_tuple(head , array_of_iterators[Index]...));
	}

	template<typename Iterator, size_t N>
	__host__ inline
	auto get_zip_iterator(std::array<Iterator, N>const & array_of_iterators)
	-> decltype( get_zip_iterator_helper( array_of_iterators , make_index_sequence<N> { }) )
	{
		return get_zip_iterator_helper( array_of_iterators , make_index_sequence<N> { } );

	}

	template<typename IteratorHead, typename IteratorTail, size_t N>
	__host__ inline
	auto get_zip_iterator(IteratorHead head, std::array<IteratorTail, N>const & array_of_iterators)
	-> decltype( get_zip_iterator_helper(head, array_of_iterators , make_index_sequence<N> { }) )
	{
		return get_zip_iterator_helper(head, array_of_iterators , make_index_sequence<N> { } );

	}
	//----------------------------------------
	// make a tuple of references to a existing tuple
	template<typename ...T, size_t... I>
	auto make_rtuple_helper(thrust::tuple<T...>& t , index_sequence<I...>)
	-> thrust::tuple<T&...>
	{ return thrust::tie(thrust::get<I>(t)...) ;}

	template<typename ...T>
	auto  make_rtuple( thrust::tuple<T...>& t )
	-> thrust::tuple<T&...>
	{
		return make_rtuple_helper( t, make_index_sequence<sizeof...(T)> {});
	}

	//---------------------------------------------
	// get a reference to a tuple object by index
	template<typename R, typename T, size_t I>
	__host__  __device__ inline
	typename thrust::detail::enable_if<(I == thrust::tuple_size<T>::value), void>::type
	_get_element(const size_t index, T& t, R*& ptr )
	{ }

	template<typename R, typename T, size_t I=0>
	__host__  __device__ inline
	typename thrust::detail::enable_if<( I < thrust::tuple_size<T>::value), void>::type
	_get_element(const size_t index, T& t, R*& ptr )
	{

	    index==I ? ptr=&thrust::get<I>(t):0;

	    _get_element<R,T,I+1>(index, t, ptr);
	}

	template<typename R, typename ...T>
	__host__  __device__ inline
	R& get_element(const size_t index, thrust::tuple<T...>& t)
	{
	    R* ptr;
	    _get_element( index, t, ptr );

	    return *ptr;


	}

	//----------------------------------------
	template<typename ...T1, typename ...T2, size_t... I1, size_t... I2 >
	__host__  __device__ inline
	void split_tuple_helper(thrust::tuple<T1...> &t1, thrust::tuple<T2...> &t2,
			thrust::tuple<T1..., T2...> const& t , index_sequence<I1...>, index_sequence<I2...>)
	{
		t1 = thrust::tie( thrust::get<I1>(t)... );
		t2 = thrust::tie( thrust::get<I2+ sizeof...(T1)>(t)... );

	}

	template< typename ...T1,  typename ...T2>
	__host__  __device__ inline
	void split_tuple(thrust::tuple<T1...> &t1, thrust::tuple<T2...> &t2,
			thrust::tuple<T1..., T2...> const& t)
	{
	    return split_tuple_helper(t1, t2, t ,
	    		make_index_sequence<sizeof...(T1)>{},
	    		make_index_sequence<sizeof...(T2)>{} );
	}

	//----------------------------------------
	template<typename ...T, size_t... I1, size_t... I2 >
	__host__  __device__ inline
	auto split_tuple_helper(thrust::tuple<T...> &t, index_sequence<I1...>, index_sequence<I2...>)
	-> decltype( thrust::make_pair(thrust::tie( thrust::get<I1>(t)... ), thrust::tie( thrust::get<I2+ + sizeof...(I1)>(t)... ) ) )
	{
		auto t1 = thrust::tie( thrust::get<I1>(t)... );
		auto t2 = thrust::tie( thrust::get<I2+ sizeof...(I1)>(t)... );

		return thrust::make_pair(t1, t2);
	}

	template< size_t N, typename ...T>
	__host__  __device__ inline
	auto split_tuple(thrust::tuple<T...>& t)
	-> decltype( split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} ) )
	{
	    return split_tuple_helper( t, make_index_sequence<N>{}, make_index_sequence<sizeof...(T)-N>{} );
	}



	//----------------------------------------
	template<typename Head,  typename ...Tail,  size_t... Is >
	__host__ __device__
	auto dropFirstHelper(thrust::tuple<Head, Tail...> const& t  , index_sequence<Is...> )
	-> thrust::tuple<Tail...>
	{
		return thrust::tie( thrust::get<Is+1>(t)... );
	}

	template< typename Head,  typename ...Tail>
	__host__ __device__
	auto dropFirst( thrust::tuple<Head, Tail...> const& t)
	-> decltype(dropFirstHelper( t, make_index_sequence<sizeof...(Tail) >{} ))
	{
		return dropFirstHelper(t , make_index_sequence<sizeof ...(Tail)>{} );
	}

	//----------------------------------------
	template<typename T, typename Head,  typename ...Tail,  size_t... Is >
	auto changeFirstHelper(T& new_first, thrust::tuple<Head, Tail...>  const& t  , index_sequence<Is...> )
	-> thrust::tuple<T,Tail...>
	{
		return thrust::make_tuple(new_first, thrust::get<Is+1>(t)... );
	}

	template<typename T, typename Head,  typename ...Tail>
	auto changeFirst(T& new_first, thrust::tuple<Head, Tail...>  const& t)
	-> decltype(changeFirstHelper(new_first, t , make_index_sequence<sizeof ...(Tail)>{} ))
	{
		return changeFirstHelper(new_first, t , make_index_sequence<sizeof ...(Tail)>{} );
	}


	//----------------------------------------
	//make a homogeneous tuple with same value in all elements
	template <typename T,  size_t... Is >
	auto make_tuple_helper(std::array<T, sizeof ...(Is)>& Array, index_sequence<Is...>)
	-> decltype(thrust::make_tuple(Array[Is]...))
	{
		return thrust::make_tuple(Array[Is]...);
	}


	template <typename T,  size_t N>
	auto make_tuple(T value)
	-> decltype(make_tuple_helper(std::array<T,N>(),  make_index_sequence<N>{}))
	{
		std::array<T,N> Array;

		for(auto v:Array) v=value;

		return make_tuple_helper( Array, make_index_sequence<N>{});
	}


	//---------------------------------------
	// convert a std::array to tuple.
	template <typename T,  size_t... Is>
	auto arrayToTupleHelper(std::array<T, sizeof...(Is)>const & Array, index_sequence<Is...> )
	-> decltype(thrust::make_tuple(Array[Is]...))
	{
		return thrust::make_tuple(Array[Is]...);
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
	__host__  __device__
	inline auto arrayToTupleHelper( T* Array,
			index_sequence<Indices...>)
	-> decltype(thrust::make_tuple(Array[Indices]...))
	{
		return thrust::make_tuple(Array[Indices]...);
	}

	template <typename T, size_t N>
	__host__  __device__
	inline auto arrayToTuple(T* Array)
	-> decltype( arrayToTupleHelper(Array, make_index_sequence<N>{ }))
	{
		return arrayToTupleHelper(Array, make_index_sequence<N>{ });
	}

	//---------------------------------------
	// set a generic array with tuple values
	template<size_t I = 0,  typename ArrayType, typename FistType, typename ...OtherTypes>
	__host__  __device__ inline
	typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	are_all_same<FistType,OtherTypes...>::value, void>::type
	assignArrayToTuple(thrust::tuple<FistType, OtherTypes...> &,  ArrayType (&Array)[sizeof...(OtherTypes)+1])
	{}

	template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	__host__  __device__
	inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	are_all_same<FistType,OtherTypes...>::value, void >::type
	assignArrayToTuple(thrust::tuple<FistType, OtherTypes...> & t, ArrayType (&Array)[sizeof...(OtherTypes)+1] )
	{
		thrust::get<I>(t) = (typename ArrayType::args_type) Array[I];
		assignArrayToTuple<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	}

	//---------------------------------------
	// set a std::array with tuple values
	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 inline typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	              are_all_same<FistType,OtherTypes...>::value, void>::type
	 tupleToArray(thrust::tuple<FistType, OtherTypes...> const&, std::array<FistType, sizeof...(OtherTypes) + 1>& Array)
	 {}

	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 tupleToArray(thrust::tuple<FistType, OtherTypes...> const& t, std::array<FistType, sizeof...(OtherTypes) + 1>& Array)
	 {

		 Array[I] = thrust::get<I>(t);
		 tupleToArray<I + 1,FistType, OtherTypes... >( t, Array);
	 }


	 //---------------------------------------
	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(thrust::tuple<FistType, OtherTypes...> const&,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(thrust::tuple<FistType, OtherTypes...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(thrust::tuple<FistType&, OtherTypes&...> const&,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(thrust::tuple<FistType&, OtherTypes&...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 // set a std::array with tuple values
	 template<size_t I = 0, typename ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void>::type
	 assignTupleToArray(thrust::detail::tuple_of_iterator_references<FistType, OtherTypes...> const&,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {}

	 template<size_t I = 0, typename  ArrayType, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	 is_hydra_convertible_to_tuple<ArrayType>::value &&
	 are_all_same<FistType,OtherTypes...>::value, void >::type
	 assignTupleToArray(thrust::detail::tuple_of_iterator_references<FistType, OtherTypes...> const& t,
			 ArrayType (&Array)[sizeof...(OtherTypes)+1])
	 {

		 Array[I] = thrust::get<I>(t);
		 assignTupleToArray<I + 1,ArrayType,FistType, OtherTypes... >( t, Array);
	 }

	 //---------------------------------------
	 // set a generic array with tuple values
	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == (sizeof...(OtherTypes) + 1) &&
	              are_all_same<FistType,OtherTypes...>::value, void>::type
	 tupleToArray(thrust::tuple<FistType, OtherTypes...> &,  typename std::remove_reference<FistType>::type* Array)
	 {}

	 template<size_t I = 0, typename FistType, typename ...OtherTypes>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(OtherTypes)+1) &&
	           are_all_same<FistType,OtherTypes...>::value, void >::type
	 tupleToArray(thrust::tuple<FistType, OtherTypes...> & t,  typename std::remove_reference<FistType>::type* Array)
	 {

		 Array[I] = thrust::get<I>(t);
		 tupleToArray<I + 1,FistType, OtherTypes... >( t, Array);
	 }



	 //----------------------------------------------------------
	 template<typename A, typename Tp, size_t I>
	 struct is_homogeneous_base:
	 std::conditional<thrust::detail::is_same<typename std::remove_reference<A>::type,
	 typename  std::remove_reference<typename thrust::tuple_element<I-1, Tp>::type>::type>::value, is_homogeneous_base<A, Tp, I-1>,
	 thrust::detail::false_type>::type{ };

	 template<typename A, typename Tp>
	 struct is_homogeneous_base<A, Tp, 0>: thrust::detail::true_type{ };

	 template<typename A, typename Tuple>
	 struct is_homogeneous:  is_homogeneous_base<A, Tuple, thrust::tuple_size<Tuple>::value > {};

	 // set array of pointers to point to the tuple elements
	 template<size_t I= 0, typename Array_Type, typename T>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I == thrust::tuple_size<T>::value) && is_homogeneous<Array_Type, T>::value, void>::type
	 set_ptrs_to_tuple(T& t, Array_Type** )
	 {}

	 template<size_t I = 0, typename Array_Type, typename T>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < thrust::tuple_size<T>::value) && is_homogeneous<Array_Type, T>::value, void>::type
	 set_ptrs_to_tuple(T& t, Array_Type** Array)
	 {
		 Array[I] =  &thrust::get<I>(t);
		 set_ptrs_to_tuple<I + 1,Array_Type,T>(t, Array);
	 }

	 //---------------------------------------
	 //get a element of a given tuple by index. value is stored in x


     template<typename T1, typename  T2>
     void ptr_setter( T1*& ptr, typename thrust::detail::enable_if<thrust::detail::is_same<T1,T2>::value, T2 >::type* el ){ ptr=el; }

     template<typename T1, typename  T2>
     void ptr_setter( T1*& ptr, typename thrust::detail::enable_if<!thrust::detail::is_same<T1,T2>::value, T2 >::type* el ){  }


     template<unsigned int I, typename Ptr, typename ...Tp>
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_ptr_to_tuple_element(const int, std::tuple<Tp...>&, Ptr*&)
	 {	 }

	 template<unsigned int I = 0, typename Ptr, typename ...Tp>
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 set_ptr_to_tuple_element(const  int index, std::tuple<Tp...> & t, Ptr*& ptr)
	 {
		 if(index == I ) ptr_setter<Ptr, typename std::tuple_element<I, std::tuple<Tp...>>::type>( ptr, &std::get<I>(t));

		 set_ptr_to_tuple_element<I + 1, Ptr, Tp...>(index, t, ptr);

	 }

     //---------------------------------------
	 //get a element of a given tuple by index. value is stored in x
	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 get_tuple_element(const int, thrust::tuple<Tp...> const&, T& x)
	 {}

	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 get_tuple_element(const  int index, thrust::tuple<Tp...> const& t, T& x)
	 {
		 if(index == I)x=T(  thrust::get<I>(t));
		 get_tuple_element<I + 1, T, Tp...>(index, t,x );
	 }


	 //---------------------------------------
	 //set a element of a given tuple by index to value  x
	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_tuple_element(const int, thrust::tuple<Tp...> const&, T& x)
	 {}

	 template<unsigned int I = 0, typename T, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 set_tuple_element(const  int index, thrust::tuple<Tp...>& t, const T& x)
	 {
		 if(index == I) thrust::get<I>(t)=x;
		 get_tuple_element<I + 1, T, Tp...>(index, t,x );
	 }

	 //---------------------------------------
	 // evaluate a void functor taking as argument
	 // a given tuple element
	 template<unsigned int I = 0, typename FuncT, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 eval_on_tuple_element(int, thrust::tuple<Tp...> const&, FuncT const&)
	 {}

	 template<unsigned int I = 0, typename FuncT, typename ...Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 eval_on_tuple_element(int index, thrust::tuple<Tp...> const& t, FuncT const& f)
	 {
		 if (index == 0) std::forward<FuncT const>(f)(thrust::get<I>(t));
		 eval_on_tuple_element<I + 1, FuncT, Tp...>(index-1, t, std::forward<FuncT const>(f));
	 }

	 //--------------------------------------
	 // given a tuple of functors, evaluate a
	 // element taking as argument ArgType const&
	 // arg and return on r
	 //--------------------------------------
	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 eval_tuple_element(Return_Type&, int, thrust::tuple<Tp...> const&, ArgType const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 eval_tuple_element(Return_Type& r, int index, thrust::tuple<Tp...> const& t, ArgType const& arg)
	 {
		 if (index == 0)
		 {
			 r = (Return_Type) thrust::get<I>(t)(std::forward<ArgType const>(arg));
			 return;
		 }
		 eval_tuple_element<I + 1, Return_Type,ArgType, Tp...>( r , index-1, t,arg );
	 }

	 //----------------------------------------------------------
	 // given a tuple of functors, evaluate and accumulate
	 // element taking as argument ArgType const&
	 // arg and return on r

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple(Return_Type&, thrust::tuple<Tp...> const&, ArgType const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple(Return_Type& r, thrust::tuple<Tp...>& t, ArgType& arg)
	 {
		 r = r+ (Return_Type) thrust::get<I>(t)(std::forward<ArgType&>(arg));
		 sum_tuple<I + 1, Return_Type, ArgType, Tp...>( r , t,arg );
	 }

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 sum_tuple2(Return_Type&, thrust::tuple<Tp...>&, ArgType1&, ArgType2& )
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)), void >::type
	 sum_tuple2(Return_Type& r, thrust::tuple<Tp...>& t, ArgType1& arg1, ArgType2& arg2)
	 {
		 r = r + (Return_Type) thrust::get<I>(t)(std::forward<ArgType1&>(arg1), std::forward<ArgType2&>(arg2));
		 sum_tuple2<I + 1, Return_Type, ArgType1, ArgType2, Tp...>( r , t,arg1, arg2);
	 }


	 //----------------------------------------------------------
	 // given a tuple of functors, evaluate and multiply
	 // element taking as argument ArgType &
	 // arg and return on r

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple(Return_Type&, thrust::tuple<Tp...>&, ArgType&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 product_tuple(Return_Type& r, thrust::tuple<Tp...>& t, ArgType& arg)
	 {
		 r = r*( (Return_Type) thrust::get<I>(t)(std::forward<ArgType&>(arg)));
		 product_tuple<I + 1, Return_Type, ArgType, Tp...>( r , t,arg );
	 }

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 product_tuple2(Return_Type&, thrust::tuple<Tp...> const&, ArgType1 const&, ArgType2 const&)
	 {}

	 template<size_t I = 0, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < (sizeof...(Tp))),void >::type
	 product_tuple2(Return_Type& r, thrust::tuple<Tp...> const& t, ArgType1 const& arg1, ArgType2 const& arg2)
	 {
		 r = r*((Return_Type) thrust::get<I>(t)(std::forward<ArgType1&>(arg1), std::forward<ArgType2&>(arg2)));
		 product_tuple2<I + 1, Return_Type, ArgType1, ArgType2,  Tp...>( r , t,arg1, arg2 );
	 }

	 template<size_t I = 0, size_t N,typename Return_Type, typename ArgType1, typename ArgType2, typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<I == N, void>::type
	 product_tuple3(Return_Type&, thrust::tuple<Tp...>&, ArgType1&, ArgType2&)
	 {}

	 template<size_t I = 0,size_t N, typename Return_Type, typename ArgType1, typename ArgType2,  typename ... Tp>
	 __host__  __device__
	 inline typename thrust::detail::enable_if<(I < N),void >::type
	 product_tuple3(Return_Type& r, thrust::tuple<Tp...>& t, ArgType1& arg1, ArgType2& arg2)
	 {
		 r = r*((Return_Type) thrust::get<I>(t)(std::forward<ArgType1&>(arg1), std::forward<ArgType2&>(arg2)));
		 product_tuple3<I + 1,N, Return_Type, ArgType1, ArgType2,  Tp...>( r , t,arg1, arg2 );
	 }

/*
	 //evaluate a tuple of functors and return a tuple of results
	 template< typename Tup, typename ArgType, size_t ... index>
	 __host__ __device__
	 inline auto invoke_helper(size_t n, ArgType* x, Tup& tup, index_sequence<index...>)
	 -> decltype(thrust::make_tuple(thrust::get<index>(tup)(n,x)...))
	 {
		 return thrust::make_tuple(thrust::get<index>(tup)(n,x)...);
	 }

	 template< typename Tup, typename ArgType>
	 __host__  __device__
	 inline auto invoke(size_t n, ArgType* x, Tup& tup)
	 -> decltype(invoke_helper2(n, x, tup, make_index_sequence< thrust::tuple_size<Tup>::value> { }))
	 {
		 constexpr size_t Size = thrust::tuple_size<Tup>::value;
		 return invoke_helper2(n, x, tup, make_index_sequence<Size> { });
	 }
*/
	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType, size_t ... index>
	 __host__ __device__
	 inline auto invoke_helper( ArgType& x, Tup& tup, index_sequence<index...>)
	 -> decltype(thrust::make_tuple(thrust::get<index>(tup)(x)...))
	 {
		 return thrust::make_tuple(thrust::get<index>(tup)(x)...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType>
	 __host__  __device__
	 inline auto invoke(ArgType& x, Tup& tup)
	 -> decltype(invoke_helper(x, tup, make_index_sequence< thrust::tuple_size<Tup>::value> { }))
	 {
		 constexpr size_t Size = thrust::tuple_size<Tup>::value;
		 return invoke_helper( x, tup, make_index_sequence<Size> { });
	 }

	 //evaluate a tuple of functors and return a tuple of results
	 __hydra_exec_check_disable__
	 template<typename Tup, typename ArgType1, typename ArgType2, size_t ... index>
	 __host__ __device__
	 inline auto invoke_helper( ArgType1& x, ArgType2& y, Tup& tup, index_sequence<index...>)
	 -> decltype( thrust::make_tuple(thrust::get<index>(tup)(x,y)...) )
	 {
		 return  thrust::make_tuple(thrust::get<index>(tup)(x,y)...);
	 }

	 __hydra_exec_check_disable__
	 template< typename Tup, typename ArgType1, typename ArgType2>
	 __host__  __device__
	 inline auto invoke(ArgType1& x, ArgType2& y,  Tup& tup)
	 -> decltype(invoke_helper( x, y, tup, make_index_sequence< thrust::tuple_size<Tup>::value> { }) )
	 {
		 constexpr size_t Size = thrust::tuple_size<Tup>::value;
		 return invoke_helper( x, y, tup, make_index_sequence<Size> { });
	 }


	 // set functors in tuple
	 template<size_t I = 0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 set_functors_in_tuple(thrust::tuple<Tp...>&, const std::vector<double>& parameters)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 set_functors_in_tuple(thrust::tuple<Tp...>& t,  const std::vector<double>& parameters)
	 {
		 thrust::get<I>(t).SetParameters(parameters);
		 set_functors_in_tuple<I + 1,Tp...>( t, parameters );
	 }


	 template<size_t I = 0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 print_parameters_in_tuple(thrust::tuple<Tp...>&)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 print_parameters_in_tuple(thrust::tuple<Tp...>& t)
	 {
		 thrust::get<I>(t).PrintRegisteredParameters();
		 print_parameters_in_tuple<I + 1,Tp...>(t);
	 }

	 template<size_t I=0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type
	 add_parameters_in_tuple(std::vector<hydra::Parameter*>& user_parameters, thrust::tuple<Tp...>&)
	 {}

	 template<size_t I = 0, typename ... Tp>
	 __host__
	 inline typename thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type
	 add_parameters_in_tuple(std::vector<hydra::Parameter*>& user_parameters, thrust::tuple<Tp...>& t)
	 {
		 thrust::get<I>(t).AddUserParameters(user_parameters);
		 add_parameters_in_tuple<I + 1,Tp...>(user_parameters,t);
	 }



	 //extract a element of a ntuple
	 template<typename Type, typename Tuple>
	 __host__  __device__
	 inline Type extract(const int idx, const Tuple& t)
	 {
		 Type x;
		 get_tuple_element(idx, t, x);

		 return x;
	 }


	 //accumulate by sum
	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<thrust::complex<double>,Return_Type>::value >::type >
	 __host__  __device__
	 inline Return_Type accumulate(ArgType& x, thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple(r, t, x);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType1, typename ArgType2,
	 typename ... Tp, typename=typename thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<thrust::complex<double>,Return_Type>::value>::type>
	 __host__  __device__
	 inline Return_Type accumulate2(ArgType1& x, ArgType2& y, thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::zero();
		 sum_tuple2(r, t, x, y);
		 return r;
	 }

	 //accumulate by product
	 template<typename Return_Type, typename ArgType, typename ... Tp,
	 typename=typename thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<thrust::complex<double>,Return_Type>::value >::type>
	 __host__  __device__
	 inline Return_Type product(ArgType const& x, thrust::tuple<Tp...> const& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 product_tuple(r, t, x);
		 return r;
	 }

	 template<typename Return_Type, typename ArgType1,typename ArgType2, typename ... Tp,
	 typename=typename thrust::detail::enable_if<std::is_convertible<Return_Type, double>::value || std::is_constructible<thrust::complex<double>,Return_Type>::value >::type>
	 __host__  __device__
	 inline Return_Type product2(ArgType1& x, ArgType2& y, thrust::tuple<Tp...>& t)
	 {
		 Return_Type r=TypeTraits<Return_Type>::one();
		 constexpr size_t N=thrust::tuple_size< thrust::tuple<Tp...>>::value;
		 product_tuple3<0,N,Return_Type,ArgType1, ArgType2,Tp...>(r, t, x, y);
		 return r;
	 }

	}//namespace detail


}//namespace hydra

#endif /* UTILITY_TUPLE_H_ */
