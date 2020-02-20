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
 * Distribution.inl
 *
 *  Created on: Feb 19, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DISTRIBUTION_INL_
#define DISTRIBUTION_INL_

#include <hydra/detail/external/hydra_thrust/random.h>

namespace hydra {


template<typename Functor>
struct  Distribution: protected RngFormula<Functor>
{
	typedef typename RngFormula<Functor>::value_type  value_type;


	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type operator()(Functor const& functor, Engine& rng) const
	{
		return static_cast<const RngFormula<Functor>& >(*this).Generate(functor,rng);
	}


};


template<typename Functor>
class  RngBase
{

protected:
	typedef typename Functor::value_type  value_type;

	//most of the known distributions can be sampled using normal and uniform rngs.

	typedef hydra_thrust::uniform_real_distribution<double> uniform_rng_type;
	typedef hydra_thrust::normal_distribution<double>       normal_rng_type;

	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type uniform(Engine& rng) const
	{
		return uniform_rng_type(rng, {0.0, 1.0});
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type normal(Engine& rng) const
	{
		auto dist = normal_rng_type(0.0, 1.0);
		return dist(rng);
	}

};

}  // namespace hydra



#endif /* DISTRIBUTION_INL_ */
