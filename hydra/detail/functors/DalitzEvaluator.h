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
 * DalitzEvaluator.h
 *
 *  Created on: 13/01/2021
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZEVALUATOR_H_
#define DALITZEVALUATOR_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/functors/StatsPHSP.h>
#include <hydra/detail/functors/DalitzBase.h>


#include <type_traits>
#include <utility>

namespace hydra {

namespace detail {


template<typename FUNCTOR, typename GRND>
class DalitzEvaluator: public DalitzBase<GRND>
{
	typedef DalitzBase<GRND> super_type;

public:

	typedef hydra::tuple<double, typename FUNCTOR::return_type> return_type;

	DalitzEvaluator(double motherMass, const double (&daughtersMasses)[3], FUNCTOR const& functor, size_t seed=0xabc123):
		super_type(motherMass, daughtersMasses, seed),
		fFunctor(functor)
	{ }

	__hydra_host__  __hydra_device__
	DalitzEvaluator( DalitzEvaluator<FUNCTOR,GRND>const& other):
		super_type( other),
	  	fFunctor(other.fFunctor)
	{ }

	__hydra_host__  __hydra_device__
	inline DalitzEvaluator<FUNCTOR,GRND>&
	operator=( DalitzEvaluator<FUNCTOR,GRND>const& other){

		if( this == other) return *this;

		super_type::operator=(other);
		fFunctor = other.fFunctor;

		return *this;
	}

	__hydra_host__  __hydra_device__
	inline return_type operator()(size_t i) const {

		auto masses_sq = this->MassesSq(i);

		return return_type{ this->Weight(::sqrt(hydra::get<0>(masses_sq))), fFunctor( masses_sq  )};

	}

private:

	FUNCTOR fFunctor ;
};

}  // namespace detail

}  // namespace hydra




#endif /* DALITZEVALUATOR_H_ */
