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
 * DistributionSampler.h
 *
 *  Created on: 28/07/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DISTRIBUTIONSAMPLER_H_
#define DISTRIBUTIONSAMPLER_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/FormulaTraits.h>
#include <hydra/detail/RngFormula.h>
#include <hydra/Distribution.h>
#include <hydra/detail/PRNGTypedefs.h>

namespace hydra {

namespace detail {

template< typename Functor, typename Engine=hydra::default_random_engine>
struct Sampler
{
	typedef typename Distribution<Functor>::value_type  value_type;

	Sampler()=delete;

	Sampler(Functor const& functor, const size_t seed, const size_t jump) :
		fFunctor(functor),
		fSeed(seed),
		fJump(jump )
	{}

	__hydra_host__  __hydra_device__
	Sampler(Sampler<Functor, Engine> const& other) :
	fFunctor(other.fFunctor),
	fSeed(other.fSeed),
	 fJump(other.fJump )
	{}

	__hydra_host__  __hydra_device__
	Sampler<Functor, Engine>&
	operator=(Sampler<Functor, Engine> const& other){

		if(this != &other) return *this;

		fFunctor = other.fFunctor;
		fSeed    = other.fSeed;
		fJump     = other.fJump ;

		 return *this;
	}

	__hydra_host__  __hydra_device__
	value_type operator()(size_t index) {

		Engine rng(fSeed) ;

		auto distribution = hydra::Distribution<Functor>();
		distribution.SetState(rng, fFunctor, index+fJump);

		return distribution(rng, fFunctor);

	}

private:

	Functor fFunctor;
	size_t fSeed;
	size_t  fJump;
};

}  // namespace detail

}  // namespace hydra


#endif /* DISTRIBUTIONSAMPLER_H_ */
