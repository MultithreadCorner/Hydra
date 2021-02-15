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
 * DalitzSampler.h
 *
 *  Created on: 16/12/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZSAMPLER_H_
#define DALITZSAMPLER_H_

#include <hydra/detail/external/hydra_thrust/random.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Tuple.h>
#include <hydra/detail/functors/DalitzBase.h>

#include <iostream>
namespace hydra {

namespace detail {

template<typename GRND, typename Functor= dalitz::_unity_weight>
class DalitzSampler: public DalitzBase<GRND>
{

	typedef DalitzBase<GRND> super_type;
	typedef dalitz::_unity_weight ufunctor_type;

public:

	typedef typename tuple_type<4, double>::type event_type;

	DalitzSampler(double motherMass, std::array<double, 3> const& daughtersMasses,
			                      size_t seed=0xabc123, Functor functor= ufunctor_type{}):
	super_type(motherMass, daughtersMasses, seed),
	fFunctor(functor)
	{}


	DalitzSampler(double motherMass, const double (&daughtersMasses)[3],
			size_t seed=0xabc123,  Functor functor= ufunctor_type{}):
	super_type(motherMass, daughtersMasses, seed),
	fFunctor(functor)
	{}

	__hydra_host__ __hydra_device__
	DalitzSampler( DalitzSampler<GRND, Functor>const& other):
	super_type(other),
	fFunctor(other.GetFunctor())
	{}

	__hydra_host__ __hydra_device__
	inline   DalitzSampler<GRND, Functor>&
	operator=( DalitzSampler<GRND, Functor>const& other){

		if( this == other) return *this;

		super_type::operator=(other);
		fFunctor = other.GetFunctor();
		return *this;
	}

	__hydra_host__ __hydra_device__
	inline   event_type operator()(size_t i) const {

		auto masses_sq = this->MassesSq(i);
		auto weight = this->Weight(::sqrt(hydra::get<0>(masses_sq)))*fFunctor(hydra::get<0>(masses_sq),
	        	hydra::get<1>(masses_sq),  hydra::get<2>(masses_sq) );


        return event_type{weight, hydra::get<0>(masses_sq),
        	hydra::get<1>(masses_sq),  hydra::get<2>(masses_sq) };

	}

	Functor GetFunctor() const {
		return fFunctor;
	}

private:

	Functor fFunctor;
};

}  // namespace detail

}  // namespace hydra


#endif /* DALITZSAMPLER_H_ */
