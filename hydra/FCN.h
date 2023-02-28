/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * FCN.h
 *
 *  Created on: 10/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FCN_H_
#define FCN_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>

#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/detail/functors/LogLikelihood.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <hydra/detail/Print.h>
#include <hydra/UserParameters.h>
#include <hydra/detail/EstimatorTraits.h>

#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>

#include <Minuit2/FCNBase.h>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <utility>
#include <limits>


namespace hydra {

namespace detail {

template<typename ArgType>
struct FCNWeightsReducerUnary: public hydra_thrust::unary_function<ArgType, double>
{


	__hydra_host__ __hydra_device__ 	inline
	double operator()(ArgType t){
		double r = 1.0;
		detail::multiply_tuple(r,t);
		return r;
	}

};

} //namespace detail

/**
 * \ingroup fit
 * FCN base class
 */
template<typename T, bool E= detail::is_hydra_estimator<T>::value>
class FCN;

/**
 * \ingroup fit
 */
template< template<typename ...> class Estimator, typename PDF, typename Iterator, typename ...Iterators>
class FCN<Estimator<PDF,Iterator,Iterators...>, true>;

template< template<typename ...> class Estimator, typename PDF, typename Iterator>
class FCN<Estimator<PDF,Iterator>, true >;

template<typename ...ESTIMATORS>
class FCN< hydra_thrust::tuple<FCN<ESTIMATORS>...>, false   >;

/**
 * \ingroup fit
 * \brief Convenience function to build up simultaneous fcn simultaneous
 */
template<typename ...ESTIMATORS>
FCN< hydra_thrust::tuple<FCN<ESTIMATORS>...>, false >
make_simultaneous_fcn( FCN<ESTIMATORS>const& ... fcns);

} //namespace hydra

#include <hydra/detail/FCN1.inl>
#include <hydra/detail/FCN2.inl>
#include <hydra/detail/FCN3.inl>


#endif /* FCN2_H_ */
