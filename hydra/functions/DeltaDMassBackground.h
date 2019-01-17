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
 * DeltaDMassBackground.h
 *
 *  Created on: Jul 31, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DELTADMASSBACKGROUND_H_
#define DELTADMASSBACKGROUND_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/cpp/System.h>
#include <hydra/GaussKronrodQuadrature.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/utility/SafeCompare.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include <gsl/gsl_sf_gamma.h>

namespace hydra {

/**
 * \ingroup common_functions
 * \class DeltaDMassBackground
 */
template<unsigned int ArgIndex=0>
class DeltaDMassBackground: public BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>
{
	using BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>::_par;

public:

	DeltaDMassBackground() = delete;

	DeltaDMassBackground(Parameter const& threshold, Parameter const& A, Parameter const& B, Parameter const& C):
		BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>({threshold, A, B, C})
		{}

	__hydra_host__ __hydra_device__
	DeltaDMassBackground(DeltaDMassBackground<ArgIndex> const& other ):
	BaseFunctor<DeltaDMassBackground<ArgIndex>, double,4>(other)
	{}

	__hydra_host__ __hydra_device__
	DeltaDMassBackground<ArgIndex>&
	operator=(DeltaDMassBackground<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<DeltaDMassBackground<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T* x)  const	{

		double delta   = (x[ArgIndex] - _par[0]);
		double ratio   = (x[ArgIndex] / _par[0]);

		// (1.0- exp(-x/c)))*pow(x/m, a) + b*(x/m-1.0)
		double val   = delta > 0.0 ? (1.0- ::exp(-delta/_par[3]))*::pow(ratio, _par[1]) + _par[2]*(ratio-1.0) : 0.0;

		double r = val > 0.0 ? val : 0.0;

		return  CHECK_VALUE( r , "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f ", _par[0], _par[1], _par[2], _par[3]);

	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const {

		double arg   = get<ArgIndex>(x) - _par[0];

		double delta   = (get<ArgIndex>(x) - _par[0]);
		double ratio   = (get<ArgIndex>(x) / _par[0]);

		// (1.0- exp(-x/c)))*pow(x/m, a) + b*(x/m-1.0)
		double val   = delta >0 ? (1.0- ::exp(-delta/_par[3]))*::pow(ratio, _par[1]) + _par[2]*(ratio-1.0) : 0.0;

		double r = val > 0.0 ? val : 0.0;

		return  CHECK_VALUE( r, "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f ", _par[0], _par[1], _par[2], _par[3]);

	}

};

template<unsigned int ArgIndex>
class IntegrationFormula<  DeltaDMassBackground<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula(  DeltaDMassBackground<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
	{
		const double value_at_max =  functor(UpperLimit);
		const double ratio = functor[0]/functor[3];

		bool flag_unsafe_b   = !detail::SafeGreaterThan(functor[2], 0.0);
		bool flag_unsafe_max = !detail::SafeGreaterThan(value_at_max, 0.0) ;
		bool flag_unsafe_a   = !detail::SafeGreaterThan(functor[1], -1.0) ;
		bool flag_unsafe_ratio  = !detail::SafeLessThan(ratio, 500.0) ;

		if( flag_unsafe_max ||  flag_unsafe_a||flag_unsafe_ratio) {

			if (WARNING >= Print::Level()  )
			{
				std::ostringstream stringStream;

				stringStream << "Detected potentially problematic parameters values for analytical integration:\n";

				if( flag_unsafe_max) {
					stringStream << "Diagnosis: Functor value at UpperLimit is negative or zero.\n"
							<< "Diagnosis: UpperLimit=" << UpperLimit << "Functor(" << UpperLimit << ")=" << value_at_max << ".\n";
				}

				if( flag_unsafe_a ) {
					stringStream << "Diagnosis: parameter "<< functor.GetParameter(1).GetName() << " is less than -1.0.\n"
							<< "Diagnosis: parameter value=" << functor.GetParameter(1).GetValue() << ".\n";
				}

				if( flag_unsafe_ratio ) {
					stringStream << "Diagnosis: ratio of parameters "<< functor.GetParameter(0).GetName()
											 << " and " << functor.GetParameter(3).GetName()
											 << " is much greater than 1.0.\n"
											 << "Diagnosis: parameter #0 value="
											 << functor.GetParameter(0).GetValue()
											 << " parameter #3 value="
											 << functor.GetParameter(3).GetValue()
											 << " ratio= " <<ratio << ".\n";
				}
				stringStream << "Switching to numerical integration.\n";
				HYDRA_LOG(WARNING, stringStream.str().c_str() )

			}


			//HYDRA_CALLER ;
			//HYDRA_MSG << "Detected potentially problematic parameters values for analytical integration:"<< HYDRA_ENDL;
			//if( flag_unsafe_max)  HYDRA_MSG << "Diagnosis: functor(UpperLimit) < 0" << HYDRA_ENDL;
			//if(	flag_unsafe_a	) HYDRA_MSG << "Diagnosis: parameter #1 is less than -1 "  << HYDRA_ENDL;
			//if(	flag_unsafe_ratio	) HYDRA_MSG << "Diagnosis: ratio (parameter #0)/ (parameter #3) > 200 "  << HYDRA_ENDL;
			//HYDRA_MSG << "Switching to numerical integration." << HYDRA_ENDL;
			hydra::GaussKronrodQuadrature<61,500, hydra::cpp::sys_t> NumIntegrator(LowerLimit,UpperLimit);
			return NumIntegrator(functor);

		} else {


			double min = LowerLimit > functor[0] ? LowerLimit : functor[0];

			double def_integral = cumulative(functor[0], functor[1],functor[2], functor[3], UpperLimit)
											- cumulative(functor[0], functor[1],functor[2], functor[3], min);

			return std::make_pair(
					CHECK_VALUE( def_integral," par[0] = %f par[1] = %f par[2] = %f par[3] = %f LowerLimit = %f UpperLimit = %f",
							functor[0], functor[1],functor[2], functor[3], min,UpperLimit ) ,0.0);

		}

	}

private:

	inline double cumulative(const double M0, const double A, const double B, const double C, const double x) const
		{

			return C * ::pow(::exp(1.0), M0/C) * ::pow(C/M0, A) * inc_gamma(A+1.0, x/C) +
					x * ::pow(x/M0, A)/(A + 1.0) +
					B * x * (0.5*x/M0 - 1.0);
		}

		inline double inc_gamma( const double a, const double x) const {

			return gsl_sf_gamma_inc(a, x);
		}




};




}// namespace hydra

#endif /* DELTADMASSBACKGROUND_H_ */
