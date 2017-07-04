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
 * Plain.h
 *
 *  Created on: 29/07/2016
 *      Author: Antonio Augusto Alves Junior
 */




#ifndef PLAIN_H_
#define PLAIN_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <hydra/PlainState.h>
#include <hydra/detail/functors/ProcessCallsPlain.h>
#include <utility>

namespace hydra {

template<size_t N, typename BACKEND, typename GRND=thrust::random::default_random_engine>
struct Plain;



/**
 * @ingroup numerical_integration
 * @brief This class implements the Plain MC numerical integration algorithm in Hydra.
 * The plain Monte Carlo algorithm samples points randomly from the integration region
 * to estimate the integral and its error. Using this algorithm the estimate of the integral \f$E(f|N)\f$
 * for N randomly distributed points \f$x_i\f$ is given by,
 * \f[E(f|N) ==  V <f> = (V / N) \sum_i^N f(x_i)\f]
 * where \f$V\f$ is the volume of the integration region.
 * The error on this estimate \f$\sigma(E|N)\f$ is calculated from the estimated variance of the mean,
 * \$[\sigma^2(E; N) = (V^2 / N^2)\sum_i^N (f(x_i) - <f>)^2\$].
 * For large \f$N\f$ this variance decreases asymptotically as \f$ Var(f)/N \f$, where \f$ Var(f)\f$
 * is the true variance of the function over the integration region.
 * The error estimate itself should decrease as \f$\sigma(f)/\sqrt{N}\f$.
 * The familiar law of errors decreasing as \f$1/\sqrt{N}\f$ applies—to
 * reduce the error by a factor of 10 requires a 100-fold increase in the number of sample points.
 */
template<size_t N, hydra::detail::Backend BACKEND, typename GRND>
class Plain<N, hydra::detail::BackendPolicy<BACKEND>, GRND>:
public Integrator<Plain<N,hydra::detail::BackendPolicy<BACKEND>,GRND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t> vector_t;

public:

	//tag
	typedef void hydra_integrator_tag;

	Plain()=delete;

	Plain( std::array<GReal_t,N> const& LowLim,
		   std::array<GReal_t,N> const& UpLim, size_t calls):
				fNCalls(calls),
				fResult(0),
				fAbsError(0),
				fVolume(1.0)
	{

		fVolume=1.0;
		for(size_t i=0; i<N; i++)
		{
			fDeltaX.push_back( -LowLim[i] + UpLim[i]);
			fXLow.push_back( LowLim[i]);
			fVolume *= (-LowLim[i] + UpLim[i]);
		}


	}

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t>  Integrate(FUNCTOR const& fFunctor );

	GReal_t GetSigma() const {
			return fAbsError;
		}

	GReal_t GetAbsError() const {
		return fAbsError;
	}

	void SetAbsError(GReal_t absError) {
		fAbsError = absError;
	}

	const vector_t& GetDeltaX() const {
		return fDeltaX;
	}

	void SetDeltaX(const mc_device_vector<GReal_t>& deltaX) {
		fDeltaX = deltaX;
	}

	size_t GetNCalls() const {
		return fNCalls;
	}

	void SetNCalls(size_t nCalls) {
		fNCalls = nCalls;
	}

	GReal_t GetResult() const {
		return fResult;
	}

	void SetResult(GReal_t result) {
		fResult = result;
	}


	GReal_t GetVolume() const {
		return fVolume;
	}

	void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	const vector_t& GetXLow() const {
		return fXLow;
	}

	void SetXLow(const mc_device_vector<GReal_t>& xLow) {
		fXLow = xLow;
	}

	//PlainState *fState;

private:
	size_t  fNCalls;
	GReal_t fResult;
	GReal_t fAbsError;
	GReal_t fVolume;
	vector_t fDeltaX;
	vector_t fXLow;

};

}
#include <hydra/detail/Plain.inl>
#endif /* PLAIN_H_ */
