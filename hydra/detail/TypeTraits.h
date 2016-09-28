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
 * TypeTraits.h
 *
 *  Created on: 09/06/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TYPETRAITS_H_
#define TYPETRAITS_H_


#include <hydra/detail/Config.h>
#include <hydra/Containers.h>

#include <type_traits>
#include <thrust/complex.h>
#include <thrust/detail/type_traits.h>
#include <thrust/execution_policy.h>

namespace std {

		template<class T, class U>
		struct common_type<thrust::complex<T>, thrust::complex<U> > {
			typedef thrust::complex<typename common_type<T, U>::type > type;
		};

		template<class T, class U>
		struct common_type<T, thrust::complex<U> > {
			typedef thrust::complex<typename common_type<T, U>::type> type;
		};

		template<class T, class U>
		struct common_type<thrust::complex<U>, T > {
			typedef thrust::complex<typename common_type<T, U>::type > type;
		};
}


namespace hydra {

	namespace detail {

		enum {kInvalidNumber = -111};

		template<typename T>
		struct TypeTraits
		{

			typedef T type;
			__host__  __device__ inline static type zero(){ return type(0.0) ;}
			__host__  __device__ inline static type one(){ return type(1.0) ;}
			__host__  __device__ inline static type invalid(){ return  type(kInvalidNumber) ;}

		};

		template<typename T>
		struct TypeTraits<thrust::complex<T>>
		{

			typedef thrust::complex<T> type;
			__host__  __device__ inline static type zero(){ return type(0.0,0.0) ;}
			__host__  __device__ inline static type one(){ return type(1.0, 0.0) ;}
			__host__  __device__ inline static type invalid(){ return  type(kInvalidNumber, 0.0) ;}

		};

		//--------------------------------
		template< int BACKEND>
		struct BackendTraits;

		template<>
		struct BackendTraits<device>: thrust::execution_policy<thrust::detail::device_t>
		{
			template<typename T>
			using   container = mc_device_vector<T>;
			//typedef thrust::execution_policy<thrust::detail::device_t> policy;

		};

		template<>
		struct BackendTraits<host>: thrust::execution_policy<thrust::detail::host_t>
		{
			template<typename T>
			using   container = mc_host_vector<T>;
			//typedef thrust::execution_policy<thrust::detail::host_t> policy;
		};



		//----------------------
		template<bool Condition, template<typename ...> class T1, template<typename ...> class T2>
		struct if_then_else_tt;

		template<template<typename ...> class T1, template<typename ...> class T2>
		struct if_then_else_tt<true, T1, T2>
		{
			template<typename ...T>
			using type=T1<T...>;
		};

		template<template<typename ...> class T1, template<typename ...> class T2>
		struct if_then_else_tt<false, T1, T2>
		{
			template<typename ...T>
			using type=T2<T...>;
		};


		//----------------------

		template<bool C, typename T1, typename T2>
		class if_then_else;

		template<typename T1, typename T2>
		class if_then_else<true, T1, T2>
		{
		public:
			typedef T1 type;
		};

		template<typename T1, typename T2>
		class if_then_else<false, T1, T2>
		{
		public:
			typedef T2 type;
		};

		//----------------------
		template< class... T >
		using common_type_t = typename std::common_type<T...>::type;



	 template <typename T>
		struct function_traits
			: public function_traits<decltype(&T::operator())>
		{};

		template <typename ClassType, typename ReturnType, typename... Args>
			struct function_traits<ReturnType(ClassType::*)(Args...) const>
			{
				enum { argument_count = sizeof...(Args) };

				typedef ReturnType return_type;
				typedef thrust::tuple<Args...> args_type;

				template <size_t i>
				struct arg
				{
					typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
				 };
			};


		template <typename ClassType, typename ReturnType, typename... Args>
			struct function_traits<ReturnType(ClassType::*)(Args&...)>
			{
				enum { argument_count = sizeof...(Args) };

				typedef ReturnType return_type;
				typedef thrust::tuple<Args...> args_type;

				template <size_t i>
				struct arg
				{
					typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
				 };
			};

		// check if given type is specialization of a certain template REF

		template<typename T, template<typename...> class REF>
		struct is_specialization : std::false_type {};

		template<template<typename...> class REF, typename... Args>
		struct is_specialization<REF<Args...>, REF>: std::true_type {};


}

}

#endif /* TYPETRAITS_H_ */
