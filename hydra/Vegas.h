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


#include "boost/format.hpp"

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/VegasState.h>
#include <hydra/detail/functors/ProcessCallsVegas.h>
#include <hydra/detail/Integrator.h>
#include <iostream>

//using boost::format;

namespace hydra {



template<size_t N, typename GRND=thrust::random::default_random_engine >
class Vegas : public Integrator<Vegas<N,GRND>, N>{
public:

	//tag
	typedef void hydra_numerical_integrator_tag;

	Vegas()=delete;

	/**
	 *\brief Vegas ctor taking the region of integration and the number of calls
	 * Look the documentation of VegasState to see the state parameters initialization
	 */
	Vegas(std::array<GReal_t,N> const& xlower,
			std::array<GReal_t,N> const& xupper, size_t calls);

	/**
	 *\brief Vegas ctor taking a VegasState object and the number of calls
	 * Look the documentation of VegasState to see how to set the state parameters
	 */
	Vegas(VegasState<N> const& state, size_t calls);

	/**
	 *\brief Vegas copy-ctor
	 */
	template<typename GRND2>
	Vegas( Vegas<N, GRND2> const& other):
		fNCalls(other.GetNCalls()),
		fResult(other.GetResult()),
		fAbsError(other.GetAbsError()),
		fState(other.GetState())
	{}

	/**
	 *\brief Integrate the functor in the volume defined in construction time.
	 *\param functor: integrand.
	 *\param reset: reset the integrator state between calls,
	 * it is the desiderable behavior of the integrator in during fits.
	 */
	template<typename FUNCTOR >
	GInt_t Integrate(FUNCTOR const& functor, GBool_t reset=0);
	void PrintLimits() ;
	void PrintHead() ;
	void PrintResults(GReal_t integral, GReal_t sigma,
			GReal_t cumulated_integral, GReal_t cumulated_sigma,
			GReal_t time) ;
	void PrintDistribution() ;
	void PrintGrid() ;

	inline GReal_t GetAbsError() const {
		return fAbsError;
	}

	inline void SetAbsError(GReal_t absError) {
		fAbsError = absError;
	}

	inline size_t GetNCalls() const {
		return fNCalls;
	}

	inline void SetNCalls(size_t nCalls) {
		fNCalls = nCalls;
	}


	inline GReal_t GetResult() const {
		return fResult;
	}

	inline void SetResult(GReal_t result) {
		fResult = result;
	}

	inline VegasState<N>& GetState()  {
		return fState;
	}

	inline void SetState(const VegasState<N>* state) {
		fState = state;
	}

	__host__
		inline const GReal_t* GetLowerLimit() const {
			return fState.GetXLow().data();
		}

	__host__
	inline const GReal_t* GetUpperLimit() const {
		return fState.GetXUp().data();
	}

VegasState<N> fState;
private:

	void InitGrid();
	void ResetGridValues();
	void InitBoxCoordinates();

	void ResizeGrid(const GInt_t bins);
	void RefineGrid();
	template<typename FUNCTOR >
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


	size_t fNCalls;
	GReal_t fResult;
	GReal_t fAbsError;

};

}

#include <hydra/detail/Vegas.inl>
#endif /* VEGAS_H_ */
