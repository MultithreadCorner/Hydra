/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * ProcessSPlot.h
 *
 *  Created on: 15/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PROCESSSPLOT_H_
#define PROCESSSPLOT_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

#if (HYDRA__HOST_SYSTEM==OMP || HYDRA_DEVICE_SYSTEM==OMP )
#define  HYDRA_EIGEN_MAX_STATIC_ALIGN_BYTES 0
#endif
#include <hydra/detail/external/hydra_Eigen/Dense>

namespace hydra {

namespace detail {



template<typename F1, typename F2, typename ...Fs >
struct CovMatrixUnary
{
	typedef hydra::tuple<F1, F2, Fs...> functors_tuple_type;
	constexpr static size_t nfunctors = sizeof...(Fs)+2;
	typedef typename detail::tuple_type<nfunctors*nfunctors, double>::type matrix_tuple;
	typedef hydra::Eigen::Matrix<double, nfunctors, nfunctors> covariant_matrix_type;

	CovMatrixUnary( Parameter(&coeficients)[nfunctors], functors_tuple_type const& functors ):
		fFunctors(functors)
	{
		for(size_t i=0;i<nfunctors; i++)
				fCoefficients[i] = coeficients[i].GetValue();
	}

	__hydra_host__ __hydra_device__
	CovMatrixUnary( CovMatrixUnary<F1, F2, Fs...> const& other ):
		fFunctors( other.fFunctors )
	{
		for(size_t i=0;i<nfunctors; i++)
			fCoefficients[i] = other.fCoefficients[i];
	}

	__hydra_host__ __hydra_device__
	CovMatrixUnary<F1, F2, Fs...> &
	operator=( CovMatrixUnary<F1, F2, Fs...> const& other )
	{
			if(this==&other) return *this;
			fFunctors = other.fFunctors ;
			for(size_t i=0;i<nfunctors; i++)
				fCoefficients[i] = other.fCoefficients[i];
			 return *this;
	}


	template<size_t N, size_t INDEX>
	struct index{

	 constexpr static const size_t I = INDEX/N;
	 constexpr static const size_t J = INDEX%N;
	};


	template<typename T, int N, int I>
	__hydra_host__ __hydra_device__
	inline typename hydra::thrust::detail::enable_if<(I == N*N),void >::type
	set_matrix(double denominator, T&&, hydra::Eigen::Matrix<double, N, N>&){ }

	template<typename T, int N, int I=0>
	__hydra_host__ __hydra_device__
	inline typename hydra::thrust::detail::enable_if<(I < N*N),void >::type
	set_matrix(double denominator, T&& ftuple, hydra::Eigen::Matrix<double, N, N>& fcovmatrix  )
	{
		fcovmatrix(index<N, I>::I, index<N, I>::J ) = \
				hydra::thrust::get<index<N, I>::I>(std::forward<T>(ftuple))*\
				hydra::thrust::get<index<N, I>::J>(std::forward<T>(ftuple))\
				/denominator;

		set_matrix<T, N, I+1>(denominator, ftuple, fcovmatrix);
	}

	template<typename Type>
	__hydra_host__ __hydra_device__
	inline covariant_matrix_type operator()(Type x)
	{
		auto fvalues  = detail::invoke_normalized(x, fFunctors);
		auto wfvalues = detail::multiply_array_tuple(fCoefficients, fvalues);

		GReal_t denominator   = 0;
		detail::add_tuple_values(denominator, wfvalues);
		denominator *=denominator;

		covariant_matrix_type fCovMatrix{};


        set_matrix(denominator,  fvalues, fCovMatrix);

      return fCovMatrix;
	}

	GReal_t    fCoefficients[nfunctors];
	functors_tuple_type fFunctors;
};

template<typename CovariantMatrixType>
struct CovMatrixBinary
{

	__hydra_host__ __hydra_device__
	inline CovariantMatrixType
	operator()( CovariantMatrixType const& x, CovariantMatrixType const& y )
	{
		return y + x;
	}
};

template<typename T, int I>
struct GetSWeight
{
	__hydra_host__ __hydra_device__
	double operator()( T const& sweights )	{
        double res =hydra::thrust::get<I>(sweights);
		return res;
	}
};


template<int W, typename F1, typename F2, typename ...Fs >
struct SWeights
{
	constexpr static size_t nfunctors = sizeof...(Fs)+2;
	typedef hydra::tuple<F1, F2, Fs...> functors_tuple_type;
	typedef hydra::Eigen::Matrix<double, nfunctors, nfunctors> cmatrix_t;
	typedef typename hydra::detail::tuple_type<nfunctors, double>::type tuple_t;


	SWeights() = delete;

	explicit SWeights(const  Parameter(&coeficients)[nfunctors], functors_tuple_type const& functors,
			cmatrix_t icmatrix ):
				fFunctors(functors),
				fICovMatrix( icmatrix )
	{

		for(size_t i=0;i<nfunctors; i++)
			fCoefficients[i] = coeficients[i].GetValue();
	}

	//dummy ctor just to bypass deleted default ctor
	explicit SWeights(functors_tuple_type const& functors, cmatrix_t icmatrix ):
				fFunctors(functors),
				fICovMatrix( icmatrix )
	{ }



	__hydra_host__ __hydra_device__
	inline SWeights(SWeights< W,F1, F2, Fs...> const& other ):
		fFunctors( other.fFunctors ),
		fICovMatrix( other.fICovMatrix )
	{

		for(size_t i=0;i<nfunctors; i++)
				fCoefficients[i] = other.fCoefficients[i];
	}


	__hydra_host__ __hydra_device__
	inline SWeights< W,F1, F2, Fs...>& operator=(SWeights< W,F1, F2, Fs...> const& other )
	{
		if(this == &other) return *this;

		fFunctors = other.fFunctors ;
		fICovMatrix = other.fICovMatrix ;
		for(size_t i=0;i<nfunctors; i++)
			fCoefficients[i] = other.fCoefficients[i];
		return *this;
	}


	template<typename Type, int V=W >
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< (V < 0), tuple_t >::type
	//inline tuple_t
	operator()(Type x)
	{
		auto fvalues  = detail::invoke_normalized(x, fFunctors);
		double values[nfunctors];
		detail::tupleToArray(fvalues, values);
		hydra::Eigen::Matrix<double, nfunctors,1> values_vector(hydra::Eigen::Map<Eigen::Matrix<double, nfunctors,1> >(values).eval());
		hydra::Eigen::Matrix<double, nfunctors,1> sweights(fICovMatrix*values_vector.eval());
		auto wfvalues = detail::multiply_array_tuple(fCoefficients, fvalues);
		GReal_t denominator   = 0.0;
		detail::add_tuple_values(denominator, wfvalues);
		sweights /=denominator;

		auto r = detail::arrayToTuple<double,nfunctors >(sweights.data());
		return r;
	}

	template<typename Type, int V=W>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< (V >= 0), double >::type
	//inline tuple_t
	operator()(Type x)
	{
		auto fvalues  = detail::invoke_normalized(x, fFunctors);
		double values[nfunctors];
		detail::tupleToArray(fvalues, values);
		hydra::Eigen::Matrix<double, nfunctors,1> values_vector(hydra::Eigen::Map<Eigen::Matrix<double, nfunctors,1> >(values).eval());
		hydra::Eigen::Matrix<double, nfunctors,1> sweights(fICovMatrix*values_vector.eval());
		auto wfvalues = detail::multiply_array_tuple(fCoefficients, fvalues);
		GReal_t denominator   = 0.0;
		detail::add_tuple_values(denominator, wfvalues);
		sweights /=denominator;

		return sweights.data()[V];
	}

	GReal_t    fCoefficients[nfunctors];
	functors_tuple_type fFunctors;
	cmatrix_t  fICovMatrix;
};

}  // namespace detail

}  // namespace hydra

#endif /* PROCESSSPLOT_H_ */
