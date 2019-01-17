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
 * Polynomial.h
 *
 *  Created on: 12/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include <hydra/functions/Utils.h>

namespace hydra {

/**
 * \ingroup common_functions
 * \class  Polynomial
 *
 * From :  https://en.wikipedia.org/wiki/Polynomial
 *
 * A polynomial is an expression
 * that can be built from constants and
 *  symbols called indeterminates or variables by means of addition, multiplication and
 *  exponentiation to a non-negative integer power.
 *  Two such expressions that may be transformed, one to the other,
 *  by applying the usual properties of commutativity,
 *  associativity and distributivity of
 *   addition and multiplication are considered as defining the same polynomial.
 *   A polynomial in a single indeterminate \f$ x \f$ can always be written (or rewritten) in the form
\f[ a_n x^n + a_{n-1}x^{n-1} + \dotsb + a_2 x^2 + a_1 x + a_0, \f]
where \f$ a_0, \ldots, a_n \f$ are constants and \f$ x \f$ is the indeterminate. The word "indeterminate" means that
\f$ x \f$ represents no particular value, although any value may be substituted for it.
The mapping that associates the result of this substitution to the substituted value is a function, called a ''polynomial function''.

This can be expressed more concisely by using summation notation:
\f$ \sum_{k=0}^n a_k x^k \f$.
 *
 */
template< unsigned int Order, unsigned int ArgIndex=0>
class  Polynomial:public BaseFunctor<Polynomial<Order, ArgIndex>, double, Order+1>
{
	using BaseFunctor<Polynomial<Order, ArgIndex>, double, Order+1>::_par;

public:
	Polynomial() = delete;

	Polynomial(std::array<Parameter,Order+1> const& coeficients):
		BaseFunctor<Polynomial<Order, ArgIndex>, double, Order+1>( coeficients) {}

	__hydra_host__ __hydra_device__
	Polynomial(Polynomial<Order, ArgIndex> const& other):
		BaseFunctor<Polynomial<Order, ArgIndex>, double, Order+1>(other) {}

	__hydra_host__ __hydra_device__
	inline Polynomial<Order, ArgIndex>&
	operator=( Polynomial<ArgIndex, Order> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Polynomial< Order, ArgIndex>,double, Order+1>::operator=(other);
		return *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int , T* x)  const
	{
		double coefs[Order+1]{};
		for(unsigned int i =0; i<Order+1; i++)
			coefs[i]=CHECK_VALUE(_par[i], "par[%d]=%f", i, _par[i]) ;

		double r = polynomial(coefs, x[ArgIndex]);
		return  CHECK_VALUE(r, "result =%f", r) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const
	{
		double coefs[Order+1]{};
		for(unsigned int i =0; i<Order+1; i++)
			coefs[i]=CHECK_VALUE(_par[i], "par[%d]=%f", i, _par[i]) ;

		double r = polynomial(coefs, hydra::get<ArgIndex>(x));
		return  CHECK_VALUE(r, "result =%f", r) ;
	}

private:


	template<int I>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<(I==-1), void >::type
	polynomial_helper( const double(&)[Order+1],  const double, double&)  const {}

	template<int I>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< (I < Order) &&( I>=0), void >::type
	polynomial_helper( const double(&coef)[Order+1],  const double x, double& p)  const {

		 p = p*x + coef[I];
		 polynomial_helper<I-1>(coef, x,p);
	}

	template<int I=Order>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< I==(Order), void >::type
	polynomial_helper( const double(&coef)[Order+1],  const double x, double& p)  const {

		  p=coef[I];
		  polynomial_helper<I-1>(coef, x,p);
	}

	__hydra_host__ __hydra_device__
	inline double polynomial( const double(&coef)[Order+1],  const double x) const {

		double r=0.0;
		polynomial_helper<Order>( coef,x, r);
		return r;
	}

};

template<unsigned int Order, unsigned int ArgIndex>
class IntegrationFormula< Polynomial<Order, ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( Polynomial< Order, ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
	{
		double coefs[Order+1]{};

		for(unsigned int i =0; i<Order+1; i++)	coefs[i]=functor[i];

		double r = polynomial_integral<Order+1>(coefs, UpperLimit)
				      - polynomial_integral<Order+1>(coefs, LowerLimit) ;

		return std::make_pair(r,0.0);

	}

private:

	template<unsigned int N, unsigned int I>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<(I==N), void >::type
	polynomial_integral_helper( const double, const double(&)[N], double&) const {}

	template<unsigned int N, unsigned int I=0>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<(I<N), void >::type
	polynomial_integral_helper( const double x, const double(&coef)[N], double& r) const {

		r += coef[I]*hydra::pow<double,I+1>(x)/(I+1);
		polynomial_integral_helper<N, I+1>(x,coef, r);
	}

	template<unsigned int N>
	__hydra_host__ __hydra_device__
	inline double polynomial_integral(const double(&coef)[N], double x) const {

		double r=0.0;
		polynomial_integral_helper<N,0>(x,coef, r);
		return r;
	}


};


}  // namespace hydra



#endif /* POLYNOMIAL_H_ */
