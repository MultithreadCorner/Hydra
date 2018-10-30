/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * Integrator.h
 *
 *  Created on: 31/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <utility>

namespace hydra {

template<typename Functor, size_t N=1>
class IntegrationFormula;

template<typename Functor, size_t N=1>
class AnalyticalIntegral;

template<typename Functor, size_t N=1>
class NumericalIntegral;

template<typename Algorithm, size_t N=1>
struct Integral;



template<typename Algorithm, size_t N>
struct Integral{

	typedef void hydra_integrator_tag;

	template<typename Functor>
	inline std::pair<GReal_t, GReal_t> operator()( Functor  const & functor)
	{

		auto result = static_cast< Algorithm&>(*this).Integrate(functor);

		return result;
	}

	template<typename Functor>
	inline std::pair<GReal_t, GReal_t> operator()( Functor  const & functor, double (&min)[N], double (&max)[N])
	{

		auto result = static_cast< Algorithm&>(*this).Integrate(functor, min, max);

		return result;
	}


};


template<typename Algorithm>
struct Integral<Algorithm,1>{

	typedef void hydra_integrator_tag;

	template<typename Functor>
	inline std::pair<GReal_t, GReal_t> operator()( Functor  const & functor)
	{

		auto result = static_cast< Algorithm&>(*this).Integrate(functor);

		return result;
	}

	template<typename Functor>
	inline std::pair<GReal_t, GReal_t> operator()( Functor  const & functor, double min, double max)
	{

		auto result = static_cast< Algorithm&>(*this).Integrate(functor, min, max);

		return result;
	}


};






template<typename ALGORITHM>
struct Integrator{

	typedef void hydra_integrator_tag;

	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t> operator()( FUNCTOR  const & functor)
	{
	//functor.SetNormalized(0);
	auto result = static_cast<ALGORITHM*>(this)->Integrate(functor);
	//functor.SetNormalized(1);
	return result;
	}
};



}  // namespace hydra



#endif /* INTEGRATOR_H_ */
