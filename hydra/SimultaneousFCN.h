/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * SimultaneousFCN.h
 *
 *  Created on: 04/02/2020
 *      Author: Davide Brundu
 */

#ifndef SIMULTANEOUSFCN_H_
#define SIMULTANEOUSFCN_H_

#include <hydra/FCN.h>
#include <hydra/detail/utility/Generic.h>


namespace hydra {

	template<typename ...ESTIMATORS>
	class SimultaneousFCN;


	/**
	 * \ingroup fit
	 * \brief SimultaneousFCN object
	 * \tparam ESTIMATORS
	 */
	template<typename ...ESTIMATORS>
	class SimultaneousFCN< hydra_thrust::tuple<ESTIMATORS...> >;


	namespace detail {

		template<typename T, template <typename...> class Class>
		struct is_instance_of : std::false_type {};

		template<template <typename...> class Class,typename ...T >
		struct is_instance_of< Class<T...>, Class >:  std::true_type {};

		template<typename ...ESTIMATORS >
		struct valid_estimators: hydra::detail::all_true< (hydra::detail::is_instance_of<ESTIMATORS, LogLikelihoodFCN>::value
		                                                || hydra::detail::is_instance_of<ESTIMATORS, SimultaneousFCN>::value)... >
		{  };

	}  // namespace detail



	/**
	 * \ingroup fit
	 * \brief Conveniency function to build up simultaneous fcn
	 * @param estimators hydra FCN objects: hydra::LogLikelihoodFCN or hydra::SimultaneousFCN
	 * @return
	 */
	template<typename ...ESTIMATORS>
	typename std::enable_if<hydra::detail::valid_estimators<ESTIMATORS...>::value, SimultaneousFCN< hydra_thrust::tuple<ESTIMATORS...>> >::type
	make_simultaneous_fcn( ESTIMATORS& ... estimators);



}  // namespace hydra

#include<hydra/detail/SimultaneousFCN.inl>


#endif /* SIMULTANEOUSFCN_H_ */
