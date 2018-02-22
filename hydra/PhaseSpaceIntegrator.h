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
 * PhaseSpaceIntegrator.h
 *
 *  Created on: 23/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHASESPACEINTEGRATOR_H_
#define PHASESPACEINTEGRATOR_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Integrator.h>

#include <hydra/detail/Print.h>
#include <tuple>

namespace hydra {

template <size_t N, typename Backend,  typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
class PhaseSpaceIntegrator;

template <size_t N, hydra::detail::Backend BACKEND,  typename GRND>
class PhaseSpaceIntegrator<N,  hydra::detail::BackendPolicy<BACKEND>, GRND>:
public Integrator<PhaseSpaceIntegrator<N,  hydra::detail::BackendPolicy<BACKEND>, GRND>>
{
public:
	//tag
	typedef void hydra_integrator_tag;


	PhaseSpaceIntegrator(const GReal_t motherMass, const GReal_t (&daughtersMasses)[N], size_t n):
		fGenerator( daughtersMasses),
		fMother(motherMass,0,0,0),
		fNSamples(n)
	{}


	PhaseSpaceIntegrator(const GReal_t motherMass, std::array<GReal_t,N> const& daughtersMasses, size_t n):
		fGenerator(daughtersMasses),
		fMother(motherMass,0,0,0),
		fNSamples(n)
	{}



	PhaseSpaceIntegrator(const GReal_t motherMass, std::initializer_list<GReal_t> const& daughtersMasses, size_t n):
		fGenerator(daughtersMasses),
		fMother(motherMass,0,0,0),
		fNSamples(n)
	{}

	PhaseSpaceIntegrator( PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND>, GRND>const& other):
		fGenerator( other.GetGenerator()),
		fMother( other. GetMother()  ),
		fNSamples(other.GetNSamples())
	{}

	template < hydra::detail::Backend BACKEND2,  typename GRND2>
	PhaseSpaceIntegrator( PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND2>, GRND2>const& other):
	fGenerator( other.GetGenerator()),
	fMother( other. GetMother()  ),
	fNSamples(other.GetNSamples())
	{}

	PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND>, GRND>const& other)
	{
		if(this==&other) return *this;

		fGenerator = other.GetGenerator() ;
		fMother      =  other. GetMother()  ;
		fNSamples  = other.GetNSamples() ;

		return *this;
	}

	template < hydra::detail::Backend  BACKEND2,  typename GRND2>
	PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( PhaseSpaceIntegrator<N,hydra::detail::BackendPolicy<BACKEND2>, GRND2>const& other)
	{
		if(this==&other) return *this;

		fGenerator = other.GetGenerator() ;
		fMother =  other. GetMother()  ;
		fNSamples  = other.GetNSamples() ;

		return *this;
	}


	const PhaseSpace<N, GRND>& GetGenerator() const {
		return fGenerator;
	}

	PhaseSpace<N, GRND>& GetGenerator()  {
		return fGenerator;
	}

	void SetGenerator(const PhaseSpace<N, GRND>& generator) {
		fGenerator = generator;
	}

	const Vector4R& GetMother() const {
		return fMother;
	}

	void SetMother(const Vector4R& mother) {
		fMother = mother;
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

	PhaseSpace<N,GRND> fGenerator;
	Vector4R  fMother;
	size_t fNSamples;

};



}  // namespace hydra

#include <hydra/detail/PhaseSpaceIntegrator.inl>

#endif /* PHASESPACEINTEGRATOR_H_ */
