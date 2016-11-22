/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * Vegas.h
 *
 *  Created on: 19/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */

#ifndef VEGAS_H_
#define VEGAS_H_

#include <iostream>
#include "boost/format.hpp"

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/VegasState.h>
#include <hydra/detail/functors/ProcessCallsVegas.h>
#include <hydra/detail/Integrator.h>


namespace hydra {



template<typename FUNCTOR, size_t N,  typename GRND=thrust::random::default_random_engine >
class Vegas : public Integrator<Vegas<FUNCTOR,N,GRND>, N>
{
public:

	//tag
	typedef void hydra_numerical_integrator_tag;

	Vegas(){};

	template<typename FUNCTOR, size_t N, typename GRND2>
	Vegas( Vegas<FUNCTOR, N, GRND2> const& other):
	Integrator<Vegas<FUNCTOR,N,GRND>, N>(other),
	fState(other.GetState())
	{}


	void PrintLimits() const ;
	void PrintHead() const  ;
	void PrintDistribution()  const ;
	void PrintGrid()  const ;


	VegasState<N>& GetState()  {
		return fState;
	}

	void SetState(VegasState<N> const& state) {
		fState = state;
	}

private:

	thrust::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor,
			std::array<GReal_t,N> const& xlower,
			std::array<GReal_t,N> const& xupper,
			size_t calls );

	void PrintResults(GReal_t integral, GReal_t sigma,
			GReal_t cumulated_integral, GReal_t cumulated_sigma,
			GReal_t time) const  ;

	void InitGrid();
	void ResetGridValues();
	void InitBoxCoordinates();

	void ResizeGrid(const GInt_t bins);
	void RefineGrid();

	void ProcessFuncionCalls(FUNCTOR const& functor, GReal_t& integral, GReal_t& tss);


	inline GReal_t GetCoordinate(const GUInt_t i, const GUInt_t j) const {
		return fState.GetXi()[i * N + j];
	}

	inline void SetCoordinate(const GUInt_t i, const GUInt_t j, const GReal_t x) {
		fState.SetXi(i * N + j, x);
	}

	inline GReal_t GetNewCoordinate(const GUInt_t i) const {
		return fState.GetXin()[i];
	}

	inline void SetNewCoordinate(const GUInt_t i, const GReal_t x) {
		fState.SetXin(i, x);
	}

	inline GReal_t GetDistributionValue(const GUInt_t i, const GUInt_t j) const {
		return fState.GetDistribution()[i * N + j];
	}

	inline void SetDistributionValue(const GUInt_t i, const GUInt_t j,
			const GReal_t x) {
		fState.SetDistribution(i * N + j, x);
	}

	VegasState<N> fState;

};

}

#include <hydra/detail/Vegas.inl>
#endif /* VEGAS_H_ */
