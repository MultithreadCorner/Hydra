/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * Exponential.h
 *
 *  Created on: Dec 11, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef EXPONENTIAL_H_
#define EXPONENTIAL_H_

#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>

namespace hydra {

template<unsigned int ArgIndex>
struct Exponential:public BaseFunctor<Exponential<ArgIndex>, double, 1>
{

	Exponential() = delete;

	Exponential(Parameter const& tau):
		BaseFunctor<Exponential<ArgIndex>, double, 1>({tau}) {}

	__host__ __device__
	Exponential(Exponential<ArgIndex> const& other):
		BaseFunctor<Exponential<ArgIndex>, double, 1>(other) {}

	__host__ __device__
	inline Exponential<ArgIndex>&
	operator=( Exponential<ArgIndex> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Exponential,GReal_t,1>::operator=(other);
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(unsigned int n, T* x)
	{
		return exp(x[ ArgIndex]*_par[0] );
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)
	{
		return exp(get<ArgIndex>(x)*_par[0] );
	}

};


class ExponentialAnalyticalIntegral:public Integrator<ExponentialAnalyticalIntegral>
{

public:

	ExponentialAnalyticalIntegral(double min, double max):
	fLowerLimit(min),
	fUpperLimit(max)
	{}

	inline ExponentialAnalyticalIntegral(ExponentialAnalyticalIntegral const& other):
	fLowerLimit(other.GetLowerLimit()),
	fUpperLimit(other.GetUpperLimit())
	{}

	inline ExponentialAnalyticalIntegral&
	operator=( ExponentialAnalyticalIntegral const& other)
	{
		if(this == &other) return *this;
		this->fLowerLimit = other.GetLowerLimit();
		this->fUpperLimit = other.GetUpperLimit();
		return *this;
	}

	GReal_t GetLowerLimit() const {
		return fLowerLimit;
	}

	void SetLowerLimit(GReal_t lowerLimit) {
		fLowerLimit = lowerLimit;
	}

	GReal_t GetUpperLimit() const {
		return fUpperLimit;
	}

	void SetUpperLimit(GReal_t upperLimit) {
		fUpperLimit = upperLimit;
	}

	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor)
	{
		GReal_t tau = functor[0];
		GReal_t r   =  (exp(fUpperLimit*tau) - exp(fLowerLimit*tau))/tau ;
		return std::make_pair(r,0.0);
	}

private:

	double fLowerLimit;
	double fUpperLimit;

};

}  // namespace hydra

#endif /* EXPONENTIAL_H_ */
