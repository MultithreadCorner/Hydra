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
 * PlanesDeltaAngle.h
 *
 *  Created on: 04/05/2018
 *      Author: Davide Brundu
.*      Address: davide.brundu@cern.ch
.*      GitHub:  dbrundu
 */

#ifndef PLANESDELTAANGLE_H_
#define PLANESDELTAANGLE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>

namespace hydra {

/**
 * \ingroup common_functions
 * \class PlanesDeltaAngle
 *
 *  This functor calculates the delta angle between decay plane of the particle with four-vector d2 and d3 (same plane)
 *  and h1 (other plane)
 */
class PlanesDeltaAngle:public BaseFunctor<PlanesDeltaAngle, double, 0>
{

public:

	__hydra_host__  __hydra_device__
	PlanesDeltaAngle(){};   //class_name ( const class_name & )

	__hydra_host__  __hydra_device__
	PlanesDeltaAngle( PlanesDeltaAngle const& other):
		BaseFunctor<PlanesDeltaAngle,double, 0>(other)
	{}

	__hydra_host__  __hydra_device__ inline
	PlanesDeltaAngle&		operator=( PlanesDeltaAngle const& other){
			if(this==&other) return  *this;
			BaseFunctor<PlanesDeltaAngle,double, 0>::operator=(other);
			return  *this;
		}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int , hydra::Vector4R* p)  const {

		hydra::Vector4R d2 = p[0];
		hydra::Vector4R d3 = p[1];
		hydra::Vector4R h1 = p[2];

		return chi_angle( d2, d3, h1);

	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T p)  const {

		hydra::Vector4R d2 = get<0>(p);
		hydra::Vector4R d3 = get<1>(p);
		hydra::Vector4R h1 = get<2>(p);

		return chi_angle( d2, d3, h1);
	}

	__hydra_host__ __hydra_device__ inline
	double operator()(Vector4R const& d2, Vector4R const& d3, Vector4R const& h1,  Vector4R const& h2) const {

		return chi_angle( d2, d3, h1);

	}

private:

	__hydra_host__ __hydra_device__ inline
	GReal_t chi_angle(Vector4R const& d2, Vector4R const& d3, Vector4R const& h1)const {


		hydra::Vector4R D = d2 + d3;
     
		hydra::Vector4R d1_perp = d2 - (D.dot(d2) / D.dot(D)) * D;
		hydra::Vector4R h1_perp = h1 - (D.dot(h1) / D.dot(D)) * D;

		// orthogonal to both D and d1_perp
		hydra::Vector4R d1_prime = D.cross(d1_perp);

		d1_perp = d1_perp / d1_perp.d3mag();
		d1_prime = d1_prime / d1_prime.d3mag();

		GReal_t x, y;

		x = d1_perp.dot(h1_perp);
		y = d1_prime.dot(h1_perp);

		GReal_t chi = ::atan2(y, x);
      

		return chi;

		}



};

}  // namespace hydra




#endif /* PLANESDELTAANGLE_H_ */
