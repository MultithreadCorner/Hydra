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
 * ChiSquare.h
 *
 *  Created on: 31/03/2018
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 18 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

/**
 * \ingroup common_functions
 *
 */
#ifndef CHISQUARE_H_
#define CHISQUARE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Distribution.h>
#include <hydra/Tuple.h>
#include <limits>
#include <utility>

namespace hydra {

/**
 * \ingroup common_functions
 * \class ChiSquare
 *
 * The chi-squared distribution (also chi-square or \f$ \chi^2 \f$ -distribution) with  \f$ k \f$ degrees of freedom is
 *  the distribution of a sum of the squares of k independent standard normal random variables.
 *  The chi-square distribution is a special case of the gamma distribution and
 *  is one of the most widely used probability distributions in inferential statistics,
 *   notably in hypothesis testing or in construction of confidence intervals.
 */
template< typename ArgType , typename Signature=double(ArgType) >
class ChiSquare: public BaseFunctor< ChiSquare<ArgType>, Signature, 1>
{
	using BaseFunctor<ChiSquare<ArgType>, Signature,  1>::_par;

	public:

		ChiSquare()=delete;

		ChiSquare(Parameter const& ndof ):
			BaseFunctor<ChiSquare<ArgType>, Signature,  1>({ndof}),
			fDenominator( (::tgamma(ndof/2.0)*::pow(2.0,ndof/2.0)) )
			{}

		__hydra_host__ __hydra_device__
		ChiSquare(ChiSquare<ArgType> const& other ):
			BaseFunctor<ChiSquare<ArgType>, Signature, 1>(other),
			fDenominator( other.GetDenominator())
			{}

		__hydra_host__ __hydra_device__
		inline ChiSquare<ArgType>&
		operator=(ChiSquare<ArgType> const& other ){
			if(this==&other) return  *this;

			BaseFunctor<ChiSquare<ArgType>, Signature, 1>::operator=(other);
			fDenominator = other.GetDenominator();

			return  *this;
		}

		__hydra_host__ __hydra_device__
		double GetDenominator() const {
			return fDenominator;
		}

		void Update(void ) override
		{
			double ndof  = 0.5*_par[0];
			fDenominator = ::tgamma(ndof)*::pow(2.0,ndof);
		}

		__hydra_host__ __hydra_device__
		inline double Evaluate(ArgType m)  const
		{
			double ndof  = _par[0];

			double r = (m > 0)?::pow(m,(ndof/2.0)-1.0) * ::exp(-m/2.0) / fDenominator:0.0;


			return CHECK_VALUE(r, "par[0]=%f", _par[0]) ;
		}

	private:

		double fDenominator;




};

template<typename ArgType>
class IntegrationFormula< ChiSquare<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( ChiSquare<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{

		double r = cumulative(functor[0], UpperLimit)
								 - cumulative(functor[0], LowerLimit);

		return std::make_pair(CHECK_VALUE(r," par[0] = %f LowerLimit = %f UpperLimit = %f",
				functor[0], LowerLimit,UpperLimit ), 0.0);

	}

private:

	inline double cumulative( const double ndof, const double x) const
	{

		return igam(ndof/2.0, x/2.0);
	}

	// borrowed from Cephes
	// left tail of incomplete gamma function:
	//inf.      k
	//  a  -x   -       x
	//  x  e     >   ----------
	//           -     -
	//         k=0   | (a+k+1)
	//
	//
	double igam( double a, double x ) const
	{
		double ans, ax, c, r;

		// LM: for negative values returns 1.0 instead of zero
		// This is correct if a is a negative integer since Gamma(-n) = +/- inf
		if (a <= 0)  return 1.0;

		if (x <= 0)  return 0.0;

		if( (x > 1.0) && (x > a ) )
			return( 1.0 - igamc(a,x) );

		// Compute  x**a * exp(-x) / gamma(a)
		ax = a * ::log(x) - x - ::lgamma(a);
		if( ax < -kMAXLOG )
			return( 0.0 );

		ax = std::exp(ax);

		// power series
		r = a;
		c = 1.0;
		ans = 1.0;

		do
		{
			r += 1.0;
			c *= x/r;
			ans += c;
		}
		while( c/ans > kMACHEP );

		return( ans * ax/a );
	}

	// incomplete gamma function (complement integral)
	//  igamc(a,x)   =   1 - igam(a,x)
	//
	//                            inf.
	//                              -
	//                     1       | |  -t  a-1
	//               =   -----     |   e   t   dt.
	//                    -      | |
	//                   | (a)    -
	//                             x
	//
	//
	// In this implementation both arguments must be positive.
	// The integral is evaluated by either a power series or
	// continued fraction expansion, depending on the relative
	// values of a and x.

	double igamc( double a, double x ) const
	{

		double ans, ax, c, yc, r, t, y, z;
		double pk, pkm1, pkm2, qk, qkm1, qkm2;

		// LM: for negative values returns 0.0
		// This is correct if a is a negative integer since Gamma(-n) = +/- inf
		if (a <= 0)  return 0.0;

		if (x <= 0) return 1.0;

		if( (x < 1.0) || (x < a) )
			return( 1.0 - igam(a,x) );

		ax = a * ::log(x) - x - ::lgamma(a);
		if( ax < -kMAXLOG )
			return( 0.0 );

		ax = ::exp(ax);

		//continued fraction
		y = 1.0 - a;
		z = x + y + 1.0;
		c = 0.0;
		pkm2 = 1.0;
		qkm2 = x;
		pkm1 = x + 1.0;
		qkm1 = z * x;
		ans = pkm1/qkm1;

		do
		{
			c += 1.0;
			y += 1.0;
			z += 2.0;
			yc = y * c;
			pk = pkm1 * z  -  pkm2 * yc;
			qk = qkm1 * z  -  qkm2 * yc;
			if(qk)
			{
				r = pk/qk;
				t = ::fabs( (ans - r)/r );
				ans = r;
			}
			else
				t = 1.0;
			pkm2 = pkm1;
			pkm1 = pk;
			qkm2 = qkm1;
			qkm1 = qk;
			if( ::fabs(pk) > kBig )
			{
				pkm2 *= kBiginv;
				pkm1 *= kBiginv;
				qkm2 *= kBiginv;
				qkm1 *= kBiginv;
			}
		}
		while( t > kMACHEP );

		return( ans * ax );
	}

	// the machine roundoff error
	static constexpr double kMACHEP = 1.11022302462515654042363166809e-16;

	// largest argument for TMath::Exp()
	static constexpr double kMAXLOG = 709.782712893383973096206318587;

	static constexpr  double kBig = 4.503599627370496e15;
	static constexpr  double kBiginv = 2.22044604925031308085e-16;
};



template<typename ArgType>
struct RngFormula< ChiSquare<ArgType> >
{

	typedef ArgType value_type;
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( ChiSquare<ArgType>const& functor) const
	{
		long int ndof  = ::lrint(functor[0]);

		if(ndof%2u == 0)
		{
			return  ndof;
		}
		else
		{
			return  ndof+1;
		}

	}

	template< typename T>
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( std::initializer_list<T> pars) const
	{
		long int ndof  = ::lrint(pars.begin()[0]);

		if(ndof%2u == 0)
		{
			return  ndof;
		}
		else
		{
			return  ndof+1;
		}
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	inline 	value_type Generate( Engine& rng, ChiSquare<ArgType>const& functor) const
	{
		long int ndof  = ::lrint(functor[0]);

	    double x = 1.0;
	if(ndof%2u == 0)
	{
		for(long i =0; i<ndof; ++i)
		x *= RngBase::uniform(rng);

		return static_cast<value_type>(-2.0*::log(x));
	}
	else{

		for(long i =0; i<ndof-1; ++i)
		    x *= RngBase::uniform(rng);

		double y = ::cos(2.0*PI*RngBase::uniform(rng));

		return static_cast<value_type>(-2.0*::log(x) - 2.0*::log(RngBase::uniform(rng))*y*y);
	}

	}


	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline value_type Generate( Engine& rng,  std::initializer_list<T> pars) const
	{
		long int ndof  = pars.begin()[0];



		double x = 1.0;

		if(ndof%2u == 0)
		{
			for(long i =0; i<ndof; ++i)
			x *= RngBase::uniform(rng);

			return static_cast<value_type>(-2.0*::log(x));
		}
		else{

			for(long i =0; i<ndof-1; ++i)
			    x *= RngBase::uniform(rng);

			double y = ::cos(2.0*PI*RngBase::uniform(rng));

			return static_cast<value_type>(-2.0*::log(x)
			                  - 2.0*::log(RngBase::uniform(rng))*y*y);
		}

	}



};

}  // namespace hydra



#endif /* CHISQUARE_H_ */
