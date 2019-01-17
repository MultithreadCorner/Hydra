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
 * Plain.h
 *
 *  Created on: 29/07/2016
 *      Author: Antonio Augusto Alves Junior
 */




#ifndef PLAIN_H_
#define PLAIN_H_
#include <iostream>
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/external/thrust/device_vector.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/PlainState.h>
#include <hydra/detail/functors/ProcessCallsPlain.h>
#include <utility>
#include <vector>

#include <hydra/detail/external/thrust/random.h>

namespace hydra {

template<size_t N, typename BACKEND, typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
struct Plain;



/**
 * \ingroup numerical_integration
 *
 * \brief This class implements the Plain MC numerical integration algorithm in Hydra.
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
public Integral<Plain<N,hydra::detail::BackendPolicy<BACKEND>,GRND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t> vector_t;

public:

	Plain()=delete;

	/**
	 * @brief Constructor for  Plain MC numerical integration algorithm.
	 * @param LowLim  is std::array<GReal_t,N> with the lower limits of the integration region.
	 * @param UpLim std::array<GReal_t,N>  with the upper limits of the integration region.
	 * @param calls Number of calls.
	 */
	Plain( std::array<GReal_t,N> const& LowLim, std::array<GReal_t,N> const& UpLim,
			size_t calls, size_t seed=159753456852):
				fSeed(seed),
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

	/**
	 * @brief Constructor for  Plain MC numerical integration algorithm.
	 * @param LowLim  is std::array<GReal_t,N> with the lower limits of the integration region.
	 * @param UpLim std::array<GReal_t,N>  with the upper limits of the integration region.
	 * @param calls Number of calls.
	 */
	Plain( const double LowLim[N] , const double  UpLim[N], size_t calls, size_t seed=159753456852):
        fSeed(seed),
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

	Plain( Plain<N, hydra::detail::BackendPolicy<BACKEND>, GRND> const& other):
		fSeed(other.GetSeed() ),
		fNCalls(other.GetNCalls()),
		fResult(other.GetResult()),
		fAbsError(other.GetAbsError() ),
		fVolume(other.GetVolume()),
		fDeltaX(other.GetDeltaX()),
		fXLow(other.GetXLow())
	{ }

	Plain<N, hydra::detail::BackendPolicy<BACKEND>, GRND>&
	operator=( Plain<N, hydra::detail::BackendPolicy<BACKEND>, GRND> const& other)
	{
		if( this==&other) return *this;

		this->fSeed     = other.GetSeed() ;
		this->fNCalls   = other.GetNCalls();
		this->fResult   = other.GetResult();
		this->fAbsError = other.GetAbsError() ;
		this->fVolume = other.GetVolume();
		this->fDeltaX = other.GetDeltaX();
		this->fXLow   = other.GetXLow();

		return *this;
	}

	template<hydra::detail::Backend BACKEND2>
	Plain( Plain<N, hydra::detail::BackendPolicy<BACKEND2>, GRND> const& other):
	fSeed(other.GetSeed() ),
	fNCalls(other.GetNCalls()),
	fResult(other.GetResult()),
	fAbsError(other.GetAbsError() ),
	fVolume(other.GetVolume()),
	fDeltaX(other.GetDeltaX()),
	fXLow(other.GetXLow())
	{ }

	template<hydra::detail::Backend BACKEND2>
	Plain<N, hydra::detail::BackendPolicy<BACKEND>, GRND>&
    operator=( Plain<N, hydra::detail::BackendPolicy<BACKEND2>, GRND> const& other)
	{
		if( this==&other) return *this;

		this->fSeed   = other.GetSeed() ;
		this->fNCalls = other.GetNCalls() ;
		this->fResult = other.GetResult();
		this->fAbsError = other.GetAbsError() ;
		this->fVolume = other.GetVolume() ;
		this->fDeltaX = other.GetDeltaX() ;
		this->fXLow   = other.GetXLow();

		return *this;
	}



	/**
	 * @brief This method performs the actual integration.
	 * @param fFunctor functor (integrand).
	 * @return std::pair<GReal_t, GReal_t> with the integration result and error.
	 */
	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t>  Integrate(FUNCTOR const& fFunctor );

	/**
	 * @brief Get the absolute error of integration.
	 * @return error of integration.
	 */
	inline GReal_t GetSigma() const {
			return fAbsError;
		}

	/**
	 * @brief Get the absolute error of integration.
	 * @return error of integration.
	 */
	inline GReal_t GetAbsError() const {
		return fAbsError;
	}

	/**
	 * Set the absolute error of integration.
	 * @param absError error of integration.
	 */
	inline void SetAbsError(GReal_t absError) {
		fAbsError = absError;
	}

	/**
	 * @brief Get a vector with the width of the integration region in each direction.
	 * @return HYDRA_EXTERNAL_NS::thrust::[backend]::vector<double>.
	 */
	inline const vector_t& GetDeltaX() const {
		return fDeltaX;
	}

	/**
	 * @brief Set a vector with the width of the integration region in each direction.
	 * @param deltaX
	 */
	inline void SetDeltaX(const std::vector<GReal_t>& deltaX) {
		fDeltaX = deltaX;
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


	inline GReal_t GetVolume() const {
		return fVolume;
	}

	inline void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	inline const vector_t& GetXLow() const {
		return fXLow;
	}

	inline void SetXLow(const std::vector<GReal_t>&  xLow) {
		fXLow = xLow;
	}

	inline size_t GetSeed() const {
		return fSeed;
	}

	inline void SetSeed(const size_t& seed) {
		fSeed = seed;
	}

	//PlainState *fState;

private:

	size_t  fSeed;
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
