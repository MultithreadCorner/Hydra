/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
 * BreitWignerNR.h
 *
 *  Created on: Dec 13, 2017
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 18 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef BREITWIGNERNR_H_
#define BREITWIGNERNR_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Function.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Distribution.h>
#include <hydra/Tuple.h>


/**
 * \ingroup common_functions
 *
 * \class BreitWignerNR implements a non-relativistic Breit-Wigner shape
 */

namespace hydra {

template<typename ArgType, typename Signature=double(ArgType) >
class BreitWignerNR: public BaseFunctor<BreitWignerNR<ArgType>, Signature, 2>
{
	using BaseFunctor<BreitWignerNR<ArgType>, Signature, 2>::_par;

public:

	BreitWignerNR()=delete;

	BreitWignerNR(Parameter const& mean, Parameter const& lambda ):
		BaseFunctor<BreitWignerNR<ArgType>, Signature, 2>({mean, lambda})
		{}

	__hydra_host__ __hydra_device__
	BreitWignerNR(BreitWignerNR<ArgType> const& other ):
		BaseFunctor<BreitWignerNR<ArgType>, Signature, 2>(other)
		{}

	__hydra_host__ __hydra_device__ inline
	BreitWignerNR<ArgType>&
	operator=(BreitWignerNR<ArgType> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<BreitWignerNR<ArgType>, Signature, 2>::operator=(other);
		return  *this;
	}


	__hydra_host__ __hydra_device__ inline
	double Evaluate(ArgType m)  const
	{
		double mean  = _par[0];
		double width = _par[1];

		double m2 = (m - mean)*(m - mean);
		double w2 = width*width;

		return CHECK_VALUE(1.0/(m2 + 0.25*w2), "par[0]=%f, par[1]=%f", _par[0], _par[1]) ;
	}


};

template<typename ArgType>
class IntegrationFormula< BreitWignerNR<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( BreitWignerNR<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{

		double r = cumulative(functor[0], functor[1], UpperLimit)
							 - cumulative(functor[0], functor[1], LowerLimit);

			return std::make_pair(	CHECK_VALUE(r," par[0] = %f par[1] = %f LowerLimit = %f UpperLimit = %f",
					functor[0], functor[1], LowerLimit,UpperLimit ), 0.0);

	}
private:

	inline double cumulative( const double mean,  const double width,  const double x) const
	{
		double c = 2.0/width;
		return c*( ::atan( c*( x - mean)));
	}


};

template<typename ArgType>
struct RngFormula< BreitWignerNR<ArgType> >
{

	typedef ArgType value_type;

	__hydra_host__ __hydra_device__
	unsigned NCalls( BreitWignerNR<ArgType>const&) const
	{
		return 1;
	}

	template< typename T>
	__hydra_host__ __hydra_device__
	unsigned NCalls( std::initializer_list<T>) const
	{
		return 1;
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type Generate( Engine& rng, BreitWignerNR<ArgType>const& functor) const
	{
		double mean  = functor[0];
		double width = functor[1];
	    double x = mean + 0.5 * width * ::tan(PI*(RngBase::uniform(rng) -0.5));

		return static_cast<value_type>(x);
	}


	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	value_type Generate( Engine& rng,  std::initializer_list<T> pars) const
	{
		double mean  = pars.begin()[0];
		double width = pars.begin()[1];
	    double x = mean + 0.5 * width * ::tan(PI*(RngBase::uniform(rng) -0.5));

		return static_cast<value_type>(x);
	}



};


}  // namespace hydra



#endif /* BREITWIGNERNR_H_ */
