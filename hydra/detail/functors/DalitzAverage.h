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
 * DalitzAverage.h
 *
 *  Created on: 22/12/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZAVERAGE_H_
#define DALITZAVERAGE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/functors/StatsPHSP.h>
#include <hydra/detail/functors/DalitzBase.h>


#include <type_traits>
#include <utility>

namespace hydra {

namespace detail {


template<typename FUNCTOR, typename GRND>
class DalitzAverager: public DalitzBase<GRND>
{
	typedef DalitzBase<GRND> super_type;

public:

	DalitzAverager(double motherMass, const double (&daughtersMasses)[3], FUNCTOR const& functor, size_t seed=0xabc123):
		super_type(motherMass, daughtersMasses, seed),
		fFunctor(functor)
	{ }

	__hydra_host__  __hydra_device__
	DalitzAverager( DalitzAverager<FUNCTOR,GRND>const& other):
		super_type( other),
	  	fFunctor(other.fFunctor)
	{ }

	__hydra_host__  __hydra_device__
	inline DalitzAverager<FUNCTOR,GRND>&
	operator=( DalitzAverager<FUNCTOR,GRND>const& other){

		if( this == other) return *this;

		super_type::operator=(other);
		fFunctor = other.fFunctor;

		return *this;
	}

	__hydra_host__  __hydra_device__
	inline StatsPHSP operator()(size_t i) const {

		StatsPHSP result;
		auto masses_sq = this->MassesSq(i);
		result.fMean = fFunctor(masses_sq  );
		result.fW    = this->Weight(::sqrt(hydra::get<0>(masses_sq)));
		result.fM2   = 0.0;

		return result;

	}

private:

	FUNCTOR fFunctor ;
};

}  // namespace detail

}  // namespace hydra



#endif /* AVERAGEDALITZ_H_ */
