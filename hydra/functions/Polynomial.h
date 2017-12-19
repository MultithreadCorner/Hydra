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

template<unsigned int ArgIndex=0, unsigned int Order=2>
struct Polynomial:public BaseFunctor<Polynomial<ArgIndex, Order>, double, Order>
{

	Polynomial() = delete;

	Polynomial(std::array<Parameter,Order> const& coeficients):
		BaseFunctor<Polynomial<ArgIndex, Order>, double, Order>( coeficients) {}

	__host__ __device__
	Polynomial(Polynomial<ArgIndex, Order> const& other):
		BaseFunctor<Polynomial<ArgIndex, Order>, double, Order>(other) {}

	__host__ __device__
	inline Polynomial<ArgIndex, Order>&
	operator=( Polynomial<ArgIndex>, Order const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Polynomial,double, Order>::operator=(other);
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline double Evaluate(unsigned int n, T* x)
	{
		return polynomial(x[ArgIndex]);
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)
	{
		return polynomial(get<ArgIndex>(x));
	}

private:

	template<unsigned int I>
	__host__ __device__ inline
	typename std::enable_if<(I==N), void >::type
	polynomial_helper( const double, double&){}

	template<unsigned int I=0>
	__host__ __device__ inline
	typename std::enable_if<(I<N), void >::type
	polynomial_helper( const double x, double& r){

		r += _par[I]*pow<double,I>(x);
		polynomial_helper<I+1>(x, r);
	}

	__host__ __device__ inline double polynomial( double x){

		double r=0.0;
		polynomial_helper(x, r);
		return r;
	}

};


struct PolynomialAnalyticalIntegral:public Integrator<PolynomialAnalyticalIntegral>
{

public:

	PolynomialAnalyticalIntegral(double min, double max):
	fLowerLimit(min),
	fUpperLimit(max)
	{
		std::assert(fLowerLimit >= fUpperLimit
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

	template<typename FUNCTOR>
	inline std::pair<double, double> Integrate(FUNCTOR const& functor)
	{
		double
		double r   =  (exp(fUpperLimit*tau) - exp(fLowerLimit*tau))/tau ;
		return std::make_pair(r,0.0);
	}

private:

	template<unsigned int I>
	__host__ __device__ inline
	typename std::enable_if<(I==N), void >::type
	polynomial_integral_helper( const double, double&){}

	template<unsigned int I=0>
	__host__ __device__ inline
	typename std::enable_if<(I<N), void >::type
	polynomial_integral_helper( const double x, const double(&coef)[N], double& r){

		r += coef[I]*pow<double,I+1>(x)/(I+1);
		polynomial_integral_helper<I+1>(x, r);
	}

	__host__ __device__ inline double polynomial_integal( double x){

		double r=0.0;
		polynomial_integal_helper(x, r);
		return r;
	}

	double fLowerLimit;
	double fUpperLimit;
};

}  // namespace hydra



#endif /* POLYNOMIAL_H_ */
