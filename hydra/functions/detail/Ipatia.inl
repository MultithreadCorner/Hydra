/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * Ipatia.inl
 *
 *  Created on: 21/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef IPATIA_INL_
#define IPATIA_INL_

namespace hydra {

/*
 double Ipatia<ArgIndex>::ipatia(const double x, const double mu,const double sigma,
		                         const double a, const double n, const double a2, const double n2,
		                         const double l, const double fb) const
		{

	    double d = x-mu;
	    double cons0 = ::sqrt(zeta);
	    double asigma = a*sigma;
	    double a2sigma = a2*sigma;
	    double out = 0.;


	    double   beta = fb;
	    double  cons1 = -2.*l;


	    double   delta = sigma*( (l>-1.0)  + ::sqrt(-2+cons1)*(l<=-1.0) );

	    double   delta2 = delta*delta;

	    if (d < -asigma ) {
	    	double  cons1 = ::exp(-beta*asigma);
	    	double   phi = 1. + asigma*asigma/delta2;
	    	double   k1 = cons1*::pow(phi,l-0.5);
	    	double   k2 = beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2*asigma/delta2;
	    	double   B = -asigma + n*k1/k2;
	    	double   A = k1*::pow(B+asigma,n);
	    	out = A*::pow(B-d,-n);
	    }
	    else if (d > a2sigma) {
	    	double   cons1 = ::exp(beta*a2sigma);
	    	double    phi = 1. + a2sigma*a2sigma/delta2;
	    	double   k1 = cons1*::pow(phi,l-0.5);
	    	double   k2 = beta*k1+ cons1*(l-0.5)*::pow(phi,l-1.5)*2.*a2sigma/delta2;
	    	double   B = -a2sigma - n2*k1/k2;
	    	double   A = k1*::pow(B+a2sigma,n2);
	    	out =  A*::pow(B+d,-n2);

	    }
	    else { out = ::exp(beta*d)*::pow(1. + d*d/delta2,l-0.5);}


	    return out;

 }
 */


 double Ipatia<ArgIndex>::ipatia(const double x, const double mu,const double sigma,
         const double A1, const double N1, const double A2, const double N2,
         const double l, const double beta) const {

	 double d = x-mu;

	 return (d < -A1*sigma ) *  left(d, sigma, A1, N1, l, beta) +
			(d >  A2*sigma ) * right(d, sigma, A2, N2, l, beta) +
			(d >= -A1*sigma ) &&  (d <=  A2*sigma ) * center(d, sigma, l, beta);


 }


 double Ipatia<ArgIndex>::left(const double d, const double sigma,
	 const double A1, const double N1, const double l,  const double beta) const {

	 const double  asigma = A1*sigma;

	 double  delta2 = sigma*( (l>-1.0)  + ::sqrt(-2+cons1)*(l<=-1.0) );

	 delta2 *= delta2;

	 const double   cons1 = ::exp(-beta*asigma);
	 const double   phi = 1.0 + asigma*asigma/delta2;
	 const double   k1  = cons1*::pow(phi,l-0.5);
	 const double   k2  = beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2;
	 const double   B   = -asigma + n*k1/k2;
	 const double   A   = k1*::pow(B+asigma,N1);

	 return A*::pow(B-d,-n);

 }

 double Ipatia<ArgIndex>::right(const double d,const double sigma,
		const double A2, const double N2, const double l,  const double beta) const{

	 const  double asigma = A2*sigma;

	 double  delta2 = sigma*( (l>-1.0)  + ::sqrt(-2+cons1)*(l<=-1.0) );

	 delta2 *= delta2;

	 const double   cons1 = ::exp(beta*asigma);
	 const double   phi = 1.0 + asigma*asigma/delta2;
	 const double   k1  = cons1*::pow(phi,l-0.5);
	 const double   k2  = beta*k1 + cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2;
	 const double   B   = - asigma - N2*k1/k2;
	 const double   A   = k1*::pow(B+asigma,N2);




	 return  A*::pow(B+d,-n2);

 }

 double Ipatia<ArgIndex>::center(const double d,const double sigma,
		 const double l, const double beta) const {


	 const   double  cons1    = -2.*l;

	 double   delta2 = sigma*( double(l>-1.0)  + ::sqrt(-2+cons1)*(l<=-1.0) );

	 delta2 *= delta2;


	  return ::exp(beta*d)*::pow(1. + d*d/delta2,l-0.5);

 }

}  // namespace hydra

#endif /* IPATIA_INL_ */
