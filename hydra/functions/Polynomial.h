/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * Polynomial.h
 *
 *  Created on: 12/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include <hydra/functions/Utils.h>

namespace hydra {

template< unsigned int Order, unsigned int ArgIndex=0>
class  Polynomial:public BaseFunctor<Polynomial<Order, ArgIndex>, double, Order>
{
	using BaseFunctor<Polynomial<Order, ArgIndex>, double, Order>::_par;

public:
	Polynomial() = delete;

	Polynomial(std::array<Parameter,Order> const& coeficients):
		BaseFunctor<Polynomial<Order, ArgIndex>, double, Order>( coeficients) {}

	__host__ __device__
	Polynomial(Polynomial<Order, ArgIndex> const& other):
		BaseFunctor<Polynomial<Order, ArgIndex>, double, Order>(other) {}

	__host__ __device__
	inline Polynomial<Order, ArgIndex>&
	operator=( Polynomial<ArgIndex, Order> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Polynomial,double, Order>::operator=(other);
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline double Evaluate(unsigned int n, T* x)  const
	{
		return polynomial(x[ArgIndex]);
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const
	{
		return polynomial(get<ArgIndex>(x));
	}

private:

	template<unsigned int I>
	__host__ __device__ inline
	typename std::enable_if<(I==Order), void >::type
	polynomial_helper( const double, double&)  const {}

	template<unsigned int I=0>
	__host__ __device__ inline
	typename std::enable_if<(I<Order), void >::type
	polynomial_helper( const double x, double& r)  const {

		r += _par[I]*pow<double,I>(x);
		polynomial_helper<I+1>(x, r);
	}

	__host__ __device__ inline double polynomial( double x) const {

		double r=0.0;
		polynomial_helper(x, r);
		return r;
	}

};


class PolynomialAnalyticalIntegral:public Integrator<PolynomialAnalyticalIntegral>
{

public:

	PolynomialAnalyticalIntegral(double min, double max):
	fLowerLimit(min),
	fUpperLimit(max)
	{
		assert(fLowerLimit >= fUpperLimit
				&& "hydra::PolynomialAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	}

	inline PolynomialAnalyticalIntegral(PolynomialAnalyticalIntegral const& other):
	fLowerLimit(other.GetLowerLimit()),
	fUpperLimit(other.GetUpperLimit())
	{}

	inline PolynomialAnalyticalIntegral&
	operator=( PolynomialAnalyticalIntegral const& other)
	{
		if(this == &other) return *this;
		this->fLowerLimit = other.GetLowerLimit();
		this->fUpperLimit = other.GetUpperLimit();
		return *this;
	}

	double GetLowerLimit() const {
		return fLowerLimit;
	}

	void SetLowerLimit(double lowerLimit) {
		fLowerLimit = lowerLimit;
	}

	double GetUpperLimit() const {
		return fUpperLimit;
	}

	void SetUpperLimit(double upperLimit) {
		fUpperLimit = upperLimit;
	}

	template<unsigned int Order, unsigned int ArgIndex >
	inline std::pair<double, double> Integrate(Polynomial<Order, ArgIndex> const& functor) const
	{
		double r   =  polynomial_integral<Order>(fUpperLimit) - polynomial_integral<Order>(fLowerLimit) ;
		return std::make_pair(r,0.0);
	}

private:

	template<unsigned int N, unsigned int I>
	__host__ __device__ inline
	typename std::enable_if<(I==N), void >::type
	polynomial_integral_helper( const double, const double(&coef)[N], double&) const {}

	template<unsigned int N, unsigned int I=0>
	__host__ __device__ inline
	typename std::enable_if<(I<N), void >::type
	polynomial_integral_helper( const double x, const double(&coef)[N], double& r) const {

		r += coef[I]*pow<double,I+1>(x)/(I+1);
		polynomial_integral_helper<N, I+1>(x, r);
	}

	template<unsigned int N>
	__host__ __device__ inline double polynomial_integral( double x) const {

		double r=0.0;
		polynomial_integral_helper<N,0>(x, r);
		return r;
	}

	double fLowerLimit;
	double fUpperLimit;
};

}  // namespace hydra



#endif /* POLYNOMIAL_H_ */
