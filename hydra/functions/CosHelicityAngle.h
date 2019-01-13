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
 * CosHelicityAngle.h
 *
 *  Created on: 29/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COSHELICITYANGLE_H_
#define COSHELICITYANGLE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/Config.h>
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
 * \class CosTheta
 *
 *  This functor calculates the cosine of the helicity angle of the particle with four-vector D,
 *  daughther of the particle with four-vector Q and grand daugther of particle  four-vector P .
 */
class CosHelicityAngle:public BaseFunctor<CosHelicityAngle, double, 0>
{

public:

	__hydra_host__  __hydra_device__
	CosHelicityAngle(){};

	__hydra_host__  __hydra_device__
	CosHelicityAngle( CosHelicityAngle const& other):
	BaseFunctor<CosHelicityAngle,double, 0>(other)
	{ }

	__hydra_host__  __hydra_device__ inline
	CosHelicityAngle&		operator=( CosHelicityAngle const& other){
			if(this==&other) return  *this;
			BaseFunctor<CosHelicityAngle,double, 0>::operator=(other);
			return  *this;
		}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int , hydra::Vector4R* p)  const {

		hydra::Vector4R P = p[0];
		hydra::Vector4R Q = p[1];
		hydra::Vector4R D = p[2];

		return cos_decay_angle( P, Q, D);

	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T p)  const {

		hydra::Vector4R P = get<0>(p);
		hydra::Vector4R Q = get<1>(p);
		hydra::Vector4R D = get<2>(p);

		return cos_decay_angle( P, Q, D);
	}

	__hydra_host__ __hydra_device__ inline
	double operator()(Vector4R const& p, Vector4R const& q, Vector4R const& d) const {

		return cos_decay_angle( p, q, d);

	}

private:

	__hydra_host__ __hydra_device__ inline
	GReal_t cos_decay_angle(Vector4R const& p, Vector4R const& q, Vector4R const& d)const {

		GReal_t pd = p*d;
		GReal_t pq = p*q;
		GReal_t qd = q*d;
		GReal_t mp2 = p.mass2();
		GReal_t mq2 = q.mass2();
		GReal_t md2 = d.mass2();

		return (pd * mq2 - pq * qd)
				/ ::sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));

		}



};

}  // namespace hydra



#endif /* COSHELICITYANGLE_H_ */
