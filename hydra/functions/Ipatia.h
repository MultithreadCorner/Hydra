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
 * Ipatia.h
 *
 *  Created on: Jul 19, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef IPATIA_H_
#define IPATIA_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

namespace hydra {

template<unsigned int ArgIndex=0>
class Ipatia : public BaseFunctor<  Ipatia<ArgIndex>, double, 7>
{
public:

  Ipatia()=delete;

  Ipatia(Parameter const& mu, Parameter const& sigma,
		 Parameter const& A1, Parameter const& N1,
		 Parameter const& A2, Parameter const& N2,
		 Parameter const& l,  Parameter const& beta,
	):
		BaseFunctor<Gaussian<ArgIndex>, double, 8>({ mu, sigma, A1, N1, A2, N2, l, beta}){}

  __hydra_host__ __hydra_device__
  Ipatia( Ipatia<ArgIndex> const& other):
    BaseFunctor< Ipatia<ArgIndex>, double, 7>(other)
  		{}


  __hydra_host__ __hydra_device__
  Ipatia<ArgIndex>& operator=( Ipatia<ArgIndex> const& other)
   {
	  if(this ==&other) return *this;

	  BaseFunctor< Ipatia<ArgIndex>, double, 7>::operator=(other);
	  return *this;
    }

  template<typename T>
  __hydra_host__ __hydra_device__
  inline double Evaluate(unsigned int, T*x)  const	{
	  double X = x[ArgIndex] ;
	  return  CHECK_VALUE(ipatia(X), "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f , par[4]=%f, par[5]=%f,par[6]=%f",\
			  _par[0], _par[1],_par[2], _par[3], _par[4], _par[5],_par[6]);

  }

  template<typename T>
  __hydra_host__ __hydra_device__
  inline double Evaluate(T x)  const {
	  double X =  get<ArgIndex>(x);
	  return  CHECK_VALUE(ipatia(X), "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f , par[4]=%f, par[5]=%f,par[6]=%f",\
				  _par[0], _par[1],_par[2], _par[3], _par[4], _par[5],_par[6]);

  }


private:

  double ipatia(const double x, const double mu,const double sigma,
	         const double A1, const double N1, const double A2, const double N2,
	         const double l, const double beta  ) const;

  double   left(const double x, const double mu,const double sigma,
	         const double A1, const double N1,
	         const double l, const double beta  ) const;

  double  right(const double x, const double mu,const double sigma,
	        const double A2, const double N2,  const double l, const double beta  ) const;

  double center(const double x, const double mu,const double sigma,
	         const double l, const double beta  ) const;


};


}  // namespace hydra

#include <detail/Ipatia.inl>

#endif /* IPATIA_H_ */
