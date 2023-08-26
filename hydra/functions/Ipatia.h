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
 * Ipatia.h
 *
 *  Created on: Jul 19, 2018
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 21 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef IPATIA_H_
#define IPATIA_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/cpp/System.h>

#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/utility/SafeCompare.h>
#include <hydra/GaussKronrodQuadrature.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include <gsl/gsl_sf_hyperg.h>

namespace hydra {

/**
 * \ingroup common_functions
 *
 * \class  hydra::Ipatia implements a version of the Ipatia distribution as described in
 * the reference https://doi.org/10.1016/j.nima.2014.06.081.
 *
 * \note hydra::Ipatia works only the parameters zeta=0 and l<0.
 * This limitation is mostly due the lack of fractional order Bessel function
 * in CUDA platform.
 */

template<typename ArgType, typename Signature=double(ArgType)>
class Ipatia : public BaseFunctor<Ipatia<ArgType>, Signature, 8>
{
	using BaseFunctor<Ipatia<ArgType>, Signature, 8>::_par;

public:

  Ipatia()=delete;

  Ipatia(Parameter const& mu, Parameter const& sigma,
		 Parameter const& A1, Parameter const& N1,
		 Parameter const& A2, Parameter const& N2,
		 Parameter const& l,  Parameter const& beta	):
		BaseFunctor<Ipatia<ArgType>, Signature, 8>({ mu, sigma, A1, N1, A2, N2, l, beta})
		{
	    if(this->GetParameter(6).GetValue() > 0.0 || this->GetParameter(6).GetUpperLim() > 0.0 || this->GetParameter(6).GetLowerLim() > 0.0 ){
	    	HYDRA_LOG(ERROR, "hydra::Ipatia's #6 is positive. This parameter needs be always negative. Exiting..." )
	    	exit(0);
	    }

		}

  __hydra_host__ __hydra_device__
  Ipatia( Ipatia<ArgType> const& other):
    BaseFunctor< Ipatia<ArgType>, Signature, 8>(other)
  		{}


  __hydra_host__ __hydra_device__
  Ipatia<ArgType>& operator=( Ipatia<ArgType> const& other)
   {
	  if(this ==&other) return *this;

	  BaseFunctor< Ipatia<ArgType>, Signature, 8>::operator=(other);
	  return *this;
    }


  __hydra_host__ __hydra_device__
  inline double Evaluate(ArgType x)  const	{

	  double mu    = _par[0];
	  double sigma = _par[1];
	  double A1    = _par[2];
	  double N1    = _par[3];
	  double A2    = _par[4];
	  double N2    = _par[5];
	  double l     = _par[6];
	  double beta  = _par[7];

	  return  CHECK_VALUE(ipatia(x, mu, sigma, A1, N1, A2, N2, l, beta), "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f , par[4]=%f, par[5]=%f,par[6]=%f,par[7]=%f\n",\
			  _par[0], _par[1],_par[2], _par[3], _par[4], _par[5],_par[6],_par[7]);

  }



private:

  __hydra_host__ __hydra_device__
  inline  double ipatia(const double x, const double mu,const double sigma,
	         const double A1, const double N1, const double A2, const double N2,
	         const double l, const double beta  ) const;

  __hydra_host__ __hydra_device__
  inline  double   left(const double d, const double sigma,
	         const double A1, const double N1, const double l, const double beta  ) const;

  __hydra_host__ __hydra_device__
  inline  double  right(const double d,const double sigma,
		  const double A2, const double N2,  const double l, const double beta  ) const;

  __hydra_host__ __hydra_device__
  inline  double center(const double d,const double sigma,
	         const double l, const double beta ) const;


};

template<typename ArgType>
class IntegrationFormula< Ipatia<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( Ipatia<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{

		double output = integral(LowerLimit-functor[0], UpperLimit-functor[0],
				functor[1], functor[2], functor[3], functor[4], functor[5], functor[6], functor[7]);

		if(::isnan(output)){

			if (WARNING >= Print::Level()  )
			{
				std::ostringstream stringStream;

				stringStream << "Detected NaN in analytical integration\n";
				stringStream << "Switching to numerical integration.\n";

				HYDRA_LOG(WARNING, stringStream.str().c_str() )

			}

			hydra::GaussKronrodQuadrature<61,500, hydra::cpp::sys_t> fNumIntegrator(LowerLimit, UpperLimit);

			return fNumIntegrator(functor);
		}
		else{

		return std::make_pair(
				CHECK_VALUE(output," par[0] = %f par[1] = %f par[2] = %f par[3] = %f par[4] = %f par[5] = %f par[6] = %f par[7] = %f LowerLimit = %f UpperLimit = %f",
						functor[0], functor[1], functor[2], functor[3],
						functor[4], functor[5], functor[6], functor[7],
						LowerLimit, UpperLimit ) ,0.0);
		}


	}
private:


	inline double integral(const double d0, const double d1, const double sigma,
			const double A1, const double N1, const double A2, const double N2,
			const double l, const  double beta ) const
	{


		double ASigma1 = A1*sigma;
		double ASigma2 = A2*sigma;
		double I0 = 0;
		double I1 = 0;
		double I1a = 0;
		double I1b = 0;

		double delta = (l<-1.0)? sigma*::sqrt(-2.0 -2.*l) : sigma;

		double delta2 = delta*delta;

		if ((d0 > -ASigma1) && (d1 < ASigma2)){
			return  d_hypergeometric(d1,delta, l) - d_hypergeometric(d0,delta, l);
		}

		if (d0 > ASigma2) {

			double cons1 = 1.;
			double phi = 1. + ASigma2*ASigma2/delta2;
			double k1 = cons1*::pow(phi,l-0.5);
			double k2 = beta*k1+ cons1*(l-0.5)*::pow(phi,l-1.5)*2.*ASigma2/delta2;
			double B = -ASigma2 - N2*k1/k2;
			double A = k1*::pow(B+ASigma2,N2);
			return A*(::pow(B+d1,1.0-N2)/(1.0-N2) -::pow(B+d0,1.0-N2)/(1.0-N2) );

		}

		if (d1 < -ASigma1) {

			double cons1 = 1.;
			double phi = 1. + ASigma1*ASigma1/delta2;
			double k1 = cons1*::pow(phi,l-0.5);
			double k2 = beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*ASigma1/delta2;
			double B = -ASigma1 + N1*k1/k2;
			double A = k1*::pow(B+ASigma1,N1);
			I0 = A*::pow(B-d0,1.0-N1)/(N1-1.0);
			I1 = A*::pow(B-d1,1.0-N1)/(N1-1.0);

			return I1 - I0;
		}



		if (d0 <-ASigma1) {

			double cons1 = 1.;
			double phi = 1. + ASigma1*ASigma1/delta2;
			double 	k1 = cons1*::pow(phi,l-0.5);
			double 	k2 = beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*ASigma1/delta2;
			double 	B = -ASigma1 + N1*k1/k2;
			double A = k1*::pow(B+ASigma1,N1);
			I0 = A*::pow(B-d0,1.0-N1)/(N1-1.0);
			I1a = A*::pow(B+ASigma1,1.0-N1)/(N1-1.0) - d_hypergeometric(-ASigma1,delta, l);

		}

		else {
			I0 = d_hypergeometric(d0,delta, l);
		}

		if (d1 > ASigma2) {

			double cons1 = 1.;
			double phi = 1. + ASigma2*ASigma2/delta2;
			double k1 = cons1*::pow(phi,l-0.5);
			double k2 = beta*k1+ cons1*(l-0.5)*::pow(phi,l-1.5)*2.*ASigma2/delta2;
			double B  = -ASigma2 - N2*k1/k2;
			double A  = k1*::pow(B+ASigma2,N2);
			I1b = A*(::pow(B+d1,1.0-N2)/(1.0-N2) -::pow(B+ASigma2,1.0-N2)/(1.0-N2) ) - d_hypergeometric(d1,delta, l) +  d_hypergeometric(ASigma2,delta, l);

		}

		I1 = d_hypergeometric(d1,delta, l) + I1a + I1b;

		return I1 - I0;


	}

	double hypergeometric_2F1(double a, double b, double c, double x) const {

		if ( detail::SafeLessThan(::fabs(x), 1.0, std::numeric_limits<double>::epsilon()) ){

			return hypergeometric_2f1(a,b,c,x); //gsl_sf_hyperg_2F1(a,b,c,x);
			}

		else {

			return   hypergeometric_2f1(c-a,b,c,1.0-1.0/(1.0-x))/::pow(1.0-x,b);//gsl_sf_hyperg_2F1(c-a,b,c,1.0-1.0/(1.0-x))/::pow(1.0-x,b);

		}
	}

	template<typename T>
	T hypergeometric_2f1(const T& AP, const T& BP, const T& CP, const T& ZM) const
	 {
	    // Implement a rational approximation of hypergeometric_2f1.
	    // This C++11 code uses a computational scheme similar to
	    // one shown in Y.L. Luke, "Algorithms for the Comuptation
	    // of Mathematical Functions", Academic Press, New York, 1977.
	    // The original Fortran77 source can be found in the subroutine
	    // R2F1() on pages 167-168. Slight corrections are made here
	    // in order to repair one or two type-setting errors in the
	    // original publication.

	    // Luke's original Fortran77 and algorithmic work are fascinating
	    // since they are generic and scalable --- decades ahead of their time.
	    // Here, in fact, the calculation is scaled down to 15 decimal digits.
	    // In other works of mine, the samp algorithm has been scaled up
	    // to 1,001 decimal digits of precision --- proof of the remarkable
	    // generic-ness of Luke's ground-breaking algorithms from a past
	    // time epoch.

	    // Retain some stylistic elements and comments from Luke's
	    // original Fortran77 coding style.

	    const T Z(-ZM); // NOLINT(readability-identifier-naming)

	    const T my_zero(static_cast<unsigned>(UINT8_C(0)));
	    const T my_one (static_cast<unsigned>(UINT8_C(1)));

	    // C
	    // C INITIALIZATION :
	    // C

	          T SABZ((AP + BP) * Z);             // NOLINT(readability-identifier-naming)
	    const T AB   (AP * BP);                  // NOLINT(readability-identifier-naming)
	    const T ABZ  (AB * Z);                   // NOLINT(readability-identifier-naming)
	    const T ABZ1((Z + ABZ) + SABZ);          // NOLINT(readability-identifier-naming)
	    const T ABZ2((ABZ1 + SABZ) + (3U * Z));  // NOLINT(readability-identifier-naming)

	    auto A = std::array<T, static_cast<std::size_t>(UINT8_C(4))> { }; // NOLINT(readability-identifier-naming)
	    auto B = std::array<T, static_cast<std::size_t>(UINT8_C(4))> { }; // NOLINT(readability-identifier-naming)

	    B[0U] = my_one;
	    A[0U] = my_one;

	    const T CP1(CP + my_one); // NOLINT(readability-identifier-naming)

	    B[1U] = my_one + (ABZ1 / (CP * 2U));
	    A[1U] = B[1U]  - (ABZ  / CP);

	    const T CT1(2U * CP1); // NOLINT(readability-identifier-naming)

	    B[2U] = my_one + ((ABZ2 / CT1) * (my_one + ABZ1 / ((-T(6U)) + (CT1 * 3U)))); // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    A[2U] = B[2U]  - ((ABZ  / CP)  * (my_one + ((ABZ2 - ABZ1) / CT1)));          // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

	    SABZ /= 4U;

	    const T Z2(Z / 2U); // NOLINT(readability-identifier-naming)

	    auto D = std::array<T, static_cast<std::size_t>(UINT8_C(9))> { }; // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers,readability-identifier-naming)

	    D[0U] = (((T(UINT8_C(7)) / 2U) - AB) * Z2) - SABZ;     // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[1U] = ABZ1 / 4U;                                     // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[2U] = D[1U] - (SABZ * 2U);                           // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[3U] = CP1 + my_one;                                  // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[4U] = CP1 * D[3U];                                   // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[5U] = CP  * D[4U];                                   // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[6U] = T(3U) / 2U;                                    // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[7U] = T(3U) / 4U;                                    // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	    D[8U] = D[7U] * Z;                                     // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

	    auto G = std::array<T, static_cast<std::size_t>(UINT8_C(3))> { }; // NOLINT(readability-identifier-naming)

	    // C
	    // C FOR I=3,...,N , THE VALUES A(I) AND B(I) ARE CALCULATED
	    // C USING THE RECURRENCE RELATIONS BELOW.
	    // C

	    for(auto   XI = static_cast<std::uint_fast16_t>(UINT16_C(3)); // NOLINT(readability-identifier-naming)
	               XI < static_cast<std::uint_fast16_t>(UINT16_C(10000));
	             ++XI)
	    {
	      G[2U]  = (D[2U] * D[1U]) / (D[7U] * D[5U]);             // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[1U] += (D[8U] + SABZ);                                // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[2U] += (D[8U] - SABZ);                                // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      G[2U] *= (D[1U] / D[6U]);                               // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      G[0U]  =  my_one + ((D[1U] + D[0U]) / (D[6U] * D[3U])); // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      G[1U]  =  D[1U] / (D[4U] * D[6U]);                      // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[7U] += (D[6U] * 2U);                                  // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      ++D[6U];                                                // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      G[1U] *= ((CP1 - XI) - ((D[2U] + D[0U]) / D[6U]));      // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

	      // C -----------------------------------------------------------------
	      // C THE RECURRENCE RELATIONS FOR A(I) and B(I) ARE AS FOLLOWS
	      // C -----------------------------------------------------------------

	      A[3U] = std::inner_product(G.crbegin(), G.crend(), A.cbegin(), my_zero);
	      B[3U] = std::inner_product(G.crbegin(), G.crend(), B.cbegin(), my_zero);

	      // Check if the iteration difference (delta) is within
	      // tolerance and break from the recursion if it is.
	      // Here we analyze the difference between this iteration
	      // result and the previous iteration result using:
	      //
	      //         |     |(A2/B2)]| |
	      // delta = | 1 - |--------| |
	      //         |     |(A3/B3) | |
	      //
	      //         |     |(A2*B3)| |
	      //       = | 1 - |-------| |,
	      //         |     |(A3*B2)| |
	      //
	      // where the absolute value of the ratio in the second term
	      // is used and the absolute value of delta is used.

	      using std::fabs;

	      const T ratio = fabs((A[2U] * B[3U]) / (A[3U] * B[2U]));
	      const T delta = fabs(my_one - ratio);

	      if((XI > static_cast<std::uint_fast16_t>(UINT8_C(7))) && (delta < std::numeric_limits<T>::epsilon()))
	      {
	        break;
	      }

	      // Shift the arrays for the next recursion.
	      std::copy(A.cbegin() + 1U, A.cend(), A.begin());
	      std::copy(B.cbegin() + 1U, B.cend(), B.begin());

	      D[8U] +=  Z2;          // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[0U] += (D[8U] * 2U); // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[5U] += (D[4U] * 3U); // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
	      D[4U] += (D[3U] * 2U); // NOLINT(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

	      ++D[3U];
	    }

	    // Return the rational approximation. This is given
	    // by the ratio of the final recursions of A and B.
	    return A.back() / B.back();
	  }

	double d_hypergeometric(double d1, double delta,double l) const {

		return d1*hypergeometric_2F1(0.5,0.5-l,1.5,-d1*d1/(delta*delta));

	}

};

}  // namespace hydra

#include <hydra/functions/detail/Ipatia.inl>

#endif /* IPATIA_H_ */
