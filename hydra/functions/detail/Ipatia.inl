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
 * Ipatia.inl
 *
 *  Created on: 21/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef IPATIA_INL_
#define IPATIA_INL_

namespace hydra {

template<unsigned int ArgIndex>
__hydra_host__ __hydra_device__
 inline  double Ipatia<ArgIndex>::ipatia(const double x, const double mu,const double sigma,
         const double A1, const double N1, const double A2, const double N2,
         const double l, const double beta) const {

	 double d = x-mu;


	 double _left_   = (d < -A1*sigma );
	 double _right_  = (d > A2*sigma );
	 double _center_ = (d >= -A1*sigma) && (d <= A2*sigma );


	 return    _left_*left(d, sigma, A1, N1, l, beta) +
			  _right_*right(d, sigma, A2, N2, l, beta) +
			 _center_*center(d, sigma, l, beta );

 }

template<unsigned int ArgIndex>
__hydra_host__ __hydra_device__
 inline  double Ipatia<ArgIndex>::left(const double d, const double sigma,
	 const double A1, const double N1, const double l,  const double beta) const {

	 const double  asigma = A1*sigma;

	 double  delta2 = (l>=-1.0)? sigma : sigma *::sqrt(-2.0 - 2.*l) ;

	 delta2 *= delta2;

	 const double   cons1 = ::exp(-beta*asigma);
	 const double   phi = 1.0 + asigma*asigma/delta2;
	 const double   k1  = cons1*::pow(phi,l-0.5);
	 const double   k2  = beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2;
	 const double   B   = -asigma + N1*k1/k2;
	 const double   A   = k1*::pow(B+asigma,N1);

	 /*
	 std::cout << std::endl
			 << "d " <<  d << std::endl
			 << "asigma " <<  asigma << std::endl
			 << "delta2 " << delta2 << std::endl
			 << "cons1 " << ::exp(-beta*asigma) << std::endl
			 << "phi " << 1.0 + asigma*asigma/delta2 << std::endl
			 << "k1 "  << cons1*::pow(phi,l-0.5) << std::endl
	         << "k2 "  << beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2 << std::endl
	         << "B "    << -asigma + N1*k1/k2 << std::endl
	         << "A "    << k1*::pow(B+asigma,N1) << std::endl
	         << "A*::pow(B-d,-N1) " << A*::pow(B-d,-N1)
			 << std::endl;
	  */

	 return (d < -A1*sigma )? A*::pow(B-d,-N1):0.0;

 }

template<unsigned int ArgIndex>
__hydra_host__ __hydra_device__
 inline  double Ipatia<ArgIndex>::right(const double d,const double sigma,
		const double A2, const double N2, const double l,  const double beta) const{

	 const  double asigma = A2*sigma;

	 double  delta2 = (l>=-1.0)? sigma : sigma *::sqrt(-2.0 - 2.*l) ;

	 delta2 *= delta2;

	 const double   cons1 = ::exp(beta*asigma);
	 const double   phi = 1.0 + asigma*asigma/delta2;
	 const double   k1  = cons1*::pow(phi,l-0.5);
	 const double   k2  = beta*k1 + cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2;
	 const double   B   = - asigma - N2*k1/k2;
	 const double   A   = k1*::pow(B+asigma,N2);

	 /*

	 std::cout << std::endl
				 << "d " <<  d << std::endl
				 << "asigma " <<  asigma << std::endl
				 << "delta2 " << delta2 << std::endl
				 << "cons1 " << ::exp(-beta*asigma) << std::endl
				 << "phi " << 1.0 + asigma*asigma/delta2 << std::endl
				 << "k1 "  << cons1*::pow(phi,l-0.5) << std::endl
		         << "k2 "  << beta*k1- cons1*(l-0.5)*::pow(phi,l-1.5)*2.0*asigma/delta2 << std::endl
		         << "B "    << - asigma - N2*k1/k2 << std::endl
		         << "A "    <<  k1*::pow(B+asigma,N2)<< std::endl
		         << "A*::pow(B+d,-N2) " << A*::pow(B+d,-N2)
				 << std::endl;

	  */

	 return (d > A2*sigma )? A*::pow(B+d,-N2):0.0;

 }

template<unsigned int ArgIndex>
__hydra_host__ __hydra_device__
 inline  double Ipatia<ArgIndex>::center(const double d,const double sigma,
		 const double l, const double beta ) const {



	 double   delta2 = (l>=-1.0) ? sigma: sigma*::sqrt(-2.0 -2.*l);

	 delta2 *= delta2;


	  return  ::exp(beta*d)*::pow(1. + d*d/delta2,l-0.5)  ;

 }

}  // namespace hydra

#endif /* IPATIA_INL_ */
