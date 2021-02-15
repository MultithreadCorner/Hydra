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
 * DalitzIntegrator.h
 *
 *  Created on: 13/01/2021
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZINTEGRATOR_H_
#define DALITZINTEGRATOR_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Integrator.h>
#include <hydra/DalitzPhaseSpace.h>
#include <hydra/detail/Print.h>
#include <tuple>
#include <hydra/Random.h>

namespace hydra {

/**
 * \ingroup phsp
 *
 */
template <typename Backend,  typename GRND=hydra::default_random_engine>
class DalitzIntegrator;

template <hydra::detail::Backend BACKEND,  typename GRND>
class DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>:
public Integral< DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND> >
{

public:
	//tag
	typedef void hydra_integrator_tag;


	DalitzIntegrator(double motherMass, double (&daughtersMasses)[3], size_t n):
		fGenerator(motherMass,  daughtersMasses),
		fNSamples(n)
	{}


	DalitzIntegrator(double motherMass, std::array<double, 3> const& daughtersMasses, size_t n):
		fGenerator(motherMass, daughtersMasses),
		fNSamples(n)
	{}


	DalitzIntegrator(double motherMass, std::initializer_list<double> const& daughtersMasses, size_t n):
		fGenerator(motherMass, daughtersMasses),
		fNSamples(n)
	{}

	DalitzIntegrator( DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>const& other):
		fGenerator( other.GetGenerator()),
		fNSamples(other.GetNSamples())
	{}

	template < hydra::detail::Backend BACKEND2,  typename GRND2>
	DalitzIntegrator( DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND2>, GRND2>const& other):
	fGenerator( other.GetGenerator()),
	fNSamples(other.GetNSamples())
	{}

	DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>const& other)
	{
		if(this==&other) return *this;

		fGenerator = other.GetGenerator() ;
		fNSamples  = other.GetNSamples() ;

		return *this;
	}

	template < hydra::detail::Backend  BACKEND2,  typename GRND2>
	DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( DalitzIntegrator<hydra::detail::BackendPolicy<BACKEND2>, GRND2>const& other)
	{
		if(this==&other) return *this;

		fGenerator = other.GetGenerator() ;
		fNSamples  = other.GetNSamples() ;

		return *this;
	}

	const DalitzPhaseSpace<GRND>& GetGenerator() const {
		return fGenerator;
	}

	DalitzPhaseSpace<GRND>& GetGenerator()  {
		return fGenerator;
	}

	void SetGenerator(const DalitzPhaseSpace<GRND>& generator) {
		fGenerator = generator;
	}


	size_t GetNSamples() const {
		return fNSamples;
	}

	void SetNSamples(size_t nSamples) {
		fNSamples = nSamples;
	}

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);

private:

	DalitzPhaseSpace<GRND> fGenerator;
	size_t fNSamples;

};


}  // namespace hydra

#include <hydra/detail/DalitzIntegrator.inl>

#endif /* DALITZINTEGRATOR_H_ */
