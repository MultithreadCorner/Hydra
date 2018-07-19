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


namespace hydra {

template<unsigned int ArgIndex=0>
class Ipatia : public BaseFunctor<  Ipatia<ArgIndex>, double, 7>
{
public:

  Ipatia()=delete;

  Ipatia( Parameter const& l, Parameter const& zeta, Parameter const& fb,
	Parameter const& sigma, Parameter const& mu, Parameter const& a,Parameter const& n):
		BaseFunctor<Gaussian<ArgIndex>, double, 7>({l,zeta,fb,sigma,mu,a,n}){}

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




private:



};


}  // namespace hydra



#endif /* IPATIA_H_ */
