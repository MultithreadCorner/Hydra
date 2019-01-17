/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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


#ifndef VEGAS_H_
#define VEGAS_H_

#include <iostream>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/VegasState.h>
#include <hydra/detail/functors/ProcessCallsVegas.h>
#include <hydra/Integrator.h>
#include <utility>
#include <hydra/detail/external/thrust/random.h>

namespace hydra {

template<size_t N, typename  BACKEND,  typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine >
class Vegas ;

/**
 * \ingroup numerical_integration
 *
 * \brief Class to perform numerical integration using Vegas algorithm.
 *
 *  The VEGAS algorithm of Lepage is based on importance sampling.
 *  It samples points from the probability distribution described by the
 *  function |f|, so that the points are concentrated in the regions that
 *  make the largest contribution to the integral.
 *
 *  *Find a more complete documentation* [here](https://www.gnu.org/software/gsl/doc/html/montecarlo.html#vegas) .
 *
 */
template<size_t N,  hydra::detail::Backend  BACKEND,  typename GRND>
class Vegas<N, hydra::detail::BackendPolicy<BACKEND>, GRND >
: public Integral<Vegas<N,hydra::detail::BackendPolicy<BACKEND>,GRND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t>  rvector_backend;
	typedef typename system_t::template container<GUInt_t>  uvector_backend;

	typedef typename rvector_backend::iterator rvector_iterator;
	typedef typename uvector_backend::iterator uvector_iterator;


public:


	Vegas()=delete;

	Vegas(std::array<GReal_t,N> const& xlower,	std::array<GReal_t,N> const& xupper, size_t ncalls):
		Integral<Vegas<N, hydra::detail::BackendPolicy<BACKEND>,GRND>>(),
		fState(xlower,xupper)
		{
		fState.SetCalls(ncalls);
		}


	Vegas(VegasState<N, hydra::detail::BackendPolicy<BACKEND>> const& state):
		Integral<Vegas<N, hydra::detail::BackendPolicy<BACKEND>,GRND>>(),
		fState(state)
		{}



	Vegas( Vegas< N, hydra::detail::BackendPolicy<BACKEND>, GRND> const& other):
	Integral<Vegas<N, hydra::detail::BackendPolicy<BACKEND>,GRND>>(),
	fState(other.GetState())
	{}

	Vegas< N, hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( Vegas< N, hydra::detail::BackendPolicy<BACKEND>, GRND> const& other)
	{
		if(this == &other) return *this;

		this->fState =other.GetState();
		return *this;

	}


	template< hydra::detail::Backend  BACKEND2, typename GRND2>
	Vegas( Vegas< N, hydra::detail::BackendPolicy<BACKEND2>, GRND2> const& other):
	Integral<Vegas<N, hydra::detail::BackendPolicy<BACKEND>,GRND>>(),
	fState(other.GetState())
	{}

	template< hydra::detail::Backend  BACKEND2, typename GRND2>
	Vegas< N, hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( Vegas< N, hydra::detail::BackendPolicy<BACKEND2>, GRND2> const& other)
	{
		if(this == &other) return *this;

		this->fState =other.GetState();
		return *this;

	}


	void PrintLimits()  ;
	void PrintHead()   ;
	void PrintDistribution()  ;
	void PrintGrid()  ;

	void PrintResults(GReal_t integral, GReal_t sigma,
			GReal_t cumulated_integral, GReal_t cumulated_sigma,
			GReal_t time) ;

	inline VegasState<N,hydra::detail::BackendPolicy<BACKEND>>& GetState()  {
		return fState;
	}

	inline VegasState<N,hydra::detail::BackendPolicy<BACKEND>> const& GetState() const {
			return fState;
		}

	inline void SetState(VegasState<N,hydra::detail::BackendPolicy<BACKEND>> const& state) {
		fState = state;
	}

	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& fFunctor);

private:



	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> IntegIterator(FUNCTOR const& functor, GBool_t training);

	void InitGrid();
	void ResetGridValues();
	void InitBoxCoordinates();

	void ResizeGrid(const GInt_t bins);
	void RefineGrid();

	template<typename FUNCTOR>
	void ProcessFuncionCalls(FUNCTOR const& functor, GBool_t training,GReal_t& integral, GReal_t& tss);


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

	VegasState<N,hydra::detail::BackendPolicy<BACKEND>> fState;
	rvector_backend fFValInput;
	uvector_backend fGlobalBinInput;
	rvector_backend fFValOutput;
	uvector_backend fGlobalBinOutput;
};

}

#include <hydra/detail/Vegas.inl>

#endif /* VEGAS_H_ */
