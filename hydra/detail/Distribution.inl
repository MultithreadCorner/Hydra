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

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/StaticAssert.h>
#include <hydra/detail/external/hydra_thrust/random.h>

#include <initializer_list>

namespace hydra {


template<typename Class>
struct Distribution<Class, false >
{

 	HYDRA_STATIC_ASSERT( !(std::is_class<Class>::value), "There is no implemented method for"
 			"sampling this distribution.\n"
 			"Provide a hydra::RngFormula specialization for this distribution\n"
 			"or use the methods provided in hydra::Random.\n"
 			"Please inspect the error messages issued above to find the line generating the error.")

};


template<typename Functor>
struct  Distribution<Functor, true>: protected RngFormula<Functor>
{
	typedef typename RngFormula<Functor>::value_type  value_type;

	/**
	 * \brief In multi-thread environment, SetState needs to be called
	 * in order to avoid the generation of overlapping pseudo-random numbers.
	 * RngFormula specializations are required to implement the method
	 * NCalls, that returns the numbers of calls to the RNG engine
	 * to generate a point.
	 */
	template<typename Engine>
	__hydra_host__ __hydra_device__
	void SetState(Engine& rng, Functor const& functor, size_t ncall) const
	{
		rng.discard(
				static_cast<const RngFormula<Functor>& >(*this).NCalls(functor)*ncall) ;
		return;
	}

	/**
	 * \brief In multi-thread environment, SetState needs to be called
	 * in order to avoid the generation of overlapping pseudo-random numbers.
	 * RngFormula specializations are required to implement the method
	 * NCalls, that returns the numbers of calls to the RNG engine
	 * to generate a point.
	 */
	template<typename Engine, typename T=double>
	__hydra_host__ __hydra_device__
	void SetState(Engine& rng,  std::initializer_list<T> pars, size_t ncall) const
	{
		rng.discard(
				static_cast<const RngFormula<Functor>& >(*this).NCalls(pars)*ncall) ;
		return;
	}

	/**
	 * \brief The function call operator return a random number at each call.
	 */
	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type operator()(Engine& rng, Functor const& functor) const
	{
		return static_cast<const RngFormula<Functor>& >(*this).Generate(rng ,functor);
	}

	/**
	 * \brief The function call operator return a random number at each call.
	 */
	template<typename Engine, typename T=double>
	__hydra_host__ __hydra_device__
	value_type operator()(Engine& rng, std::initializer_list<T> pars ) const
	{
		return static_cast<const RngFormula<Functor>& >(*this).Generate(rng , pars);
	}


};

struct  RngBase
{

	//most of the known distributions can be sampled using normal and uniform rngs.

	typedef hydra_thrust::uniform_real_distribution<double> uniform_rng_type;
	typedef hydra_thrust::normal_distribution<double>        normal_rng_type;

	/**
	 * \brief Returns pseudo-random numbers uniformly distributed in the
	 * [0,1) range.
	 */
	template<typename Engine>
	__hydra_host__ __hydra_device__
   static double uniform(Engine& rng)
	{
		auto dist = uniform_rng_type(0.0, 1.0);
		return dist(rng);
	}

	/**
	 * \brief Returns a pseudo-random numbers normally distributed
	 */
	template<typename Engine>
	__hydra_host__ __hydra_device__
	static double normal(Engine& rng)
	{
		auto dist = normal_rng_type(0.0, 1.0);
		return dist(rng);
	}

};

}  // namespace hydra



#endif /* DISTRIBUTION_INL_ */
