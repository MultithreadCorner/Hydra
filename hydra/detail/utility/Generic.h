
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
 * Generic.h
 *
 *  Created on: 20/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENERIC_H_
#define GENERIC_H_

//std
#include <type_traits>
#include <array>


namespace hydra {

	namespace detail {

	//-------------------------------------
	// is this type an instance of other template?
	template < template <typename...> class Template, typename T >
	struct is_instantiation_of : std::false_type {};

	template < template <typename...> class Template, typename... Args >
	struct is_instantiation_of< Template, Template<Args...> > : std::true_type {};


	//------------------------------------
	// calculate COND1 && COND2 &&...&& CONDN. In summary a AND of all conditions
	template<typename... Conds>
	  struct conditions_and
	  : std::true_type
	  { };

	template<typename Cond, typename... Conds>
	  struct conditions_and<Cond, Conds...>
	  : std::conditional<Cond::value, conditions_and<Conds...>, std::false_type>::type
	  { };

	//-----------------------------------
	//bool pack trick:
	//example:
	//
	template<bool...> struct bool_pack{};

	//template<bool... bs>
	//using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...> >;

	template<bool... bs>
	struct all_true:std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...> >{} ;

	template<>
	struct all_true<>:std::true_type{} ;



template<class R, class...Ts>
	using are_all_same = all_true<std::is_same<Ts, R>::value...>;


template<class ...A> struct CanConvert{
	template<class ...B> struct To{
    typedef all_true< std::is_convertible<A,B>::value...> type;
	};
};
	//-----------------------
	//check if all types are the same and store result on are_all_same<>::value
	//template<typename R, typename... T>
	//using are_all_same = conditions_and<std::is_same<R, T>...>;

	//-----------------------
	//check if a list of types T is derived of B<T, T::return_type >
	//usefull for hydra functors with CRTP (T needs have return_type defined)
	template <template<typename F, typename R> class Base, typename T, typename... Ts>
	struct are_base_of :
	    std::conditional<std::is_base_of<Base<T, typename T::return_type>, T>::value,
	                     are_base_of<Base, Ts...>, std::false_type>::type{};

	template <template<typename F, typename R> class  Base, typename T>
	struct are_base_of<Base, T> : std::is_base_of<Base<T, typename T::return_type>, T> {};

	//--------------------------------------------
    // hydra implementation of integer_sequence and make_index_sequence
	// available in c++14 but not on c++11
	template<size_t ...Ints>
	struct index_sequence
	{
		using type = index_sequence;
		using value_type = size_t;
		static constexpr std::size_t size()
		{
			return sizeof...(Ints);
		}
	};


	template<class Sequence1, class Sequence2>
	struct _merge_and_renumber;

	template<size_t ... I1, size_t ... I2>
	struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>:
	index_sequence<	I1..., (sizeof...(I1)+I2)...> {	};

	template<size_t N>
	struct make_index_sequence:_merge_and_renumber<typename make_index_sequence<N / 2>::type,
			typename make_index_sequence<N - N / 2>::type> {};

	template<> struct make_index_sequence<0> : index_sequence<> {};
	template<> struct make_index_sequence<1> : index_sequence<0> {};


	//--------------------------------
	template<typename, typename>
	struct append_to_type_seq { };

	template<typename T, typename... Ts, template<typename...> class TT>
	struct append_to_type_seq<T, TT<Ts...>>	{ using type = TT<Ts..., T>; };

	template<typename T, unsigned int N, template<typename...> class TT>
	struct repeat {
	    using type = typename
	        append_to_type_seq<T, typename repeat<T, N-1, TT>::type >::type;
	};

	template<typename T, template<typename...> class TT>
	struct repeat<T, 0, TT>
	{ using type = TT<>; };

	//--------------------------------
	template<bool FLAG>
	struct ObjSelector;

	template<>
	struct ObjSelector<true>
	{
		template<typename T1, typename T2>
		static T1 select(T1 const& obj1, T2 const& /*obj2*/ ){ return obj1;}
	};

	template<>
	struct ObjSelector<false>
	{
		template<typename T1, typename T2>
		static T2 select(T1 const& /*obj1*/, T2 const& obj2 ){ return obj2;}
	};


	//pow c-time

	template<int B, unsigned int E>
	struct power {
	    enum{ value = B*power<B, E-1>::value };
	};

	template< int B >
	struct power<B, 0> {
	    enum{ value = 1 };
	};
/*
 *  conversion of one-dimensional index to multidimensional one
 * ____________________________________________________________
 */

	//----------------------------------------
	// multiply  std::array elements
	//----------------------------------------
	template<typename T, size_t N, size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply( std::array<T, N> const& , T&  )
	{ }

	template<typename T, size_t N, size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply( std::array<T, N> const&  obj, T& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply< T, N, I+1>( obj, result );
	}

	//----------------------------------------
	// multiply static array elements
	//----------------------------------------
	template<typename T, size_t N, size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply(const T (&)[N] , T& )
	{ }

	template<typename T, size_t N, size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply(const  T (&obj)[N], T& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply< T, N, I+1>( obj, result );
	}



	//-------------------------
	// std::array version
	//-------------------------
	//end of recursion
	template<typename T, size_t DIM, size_t I>
	typename std::enable_if< (I==DIM) && (std::is_integral<T>::value), void  >::type
	get_indexes(size_t , std::array<T, DIM> const& , std::array<T,DIM>& )
	{}

	//begin of the recursion
	template<typename T, size_t DIM, size_t I=0>
	typename std::enable_if< (I<DIM) && (std::is_integral<T>::value), void  >::type
	get_indexes(size_t index, std::array<T, DIM> const& depths, std::array<T,DIM>& indexes)
	{

		size_t factor    =  1;
	    multiply<size_t, DIM, I+1>(depths, factor );

		indexes[I]  =  index/factor;

	    size_t next_index =  index%factor;

		get_indexes<T, DIM, I+1>(next_index, depths, indexes );

	}

	//-------------------------
	// static array version
	//-------------------------
	//end of recursion
	template<typename T, size_t DIM, size_t I>
	typename std::enable_if< (I==DIM) && (std::is_integral<T>::value), void  >::type
	get_indexes(size_t , const T ( &)[DIM], T (&)[DIM])
	{}

	//begin of the recursion
	template<typename T, size_t DIM, size_t I=0>
	typename std::enable_if< (I<DIM) && (std::is_integral<T>::value), void  >::type
	get_indexes(size_t index,const  T ( &depths)[DIM], T (&indexes)[DIM] )
	{

		size_t factor    =  1;
	    multiply<size_t, DIM, I+1>(depths, factor );

		indexes[I]  =  index/factor;

	    size_t next_index =  index%factor;

		get_indexes<T, DIM, I+1>(next_index, depths, indexes );

	}

	}//namespace detail
}//namespace hydra


#endif /* GENERIC_H_ */
