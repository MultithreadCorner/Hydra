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
 * AnalyticalIntegral.inl
 *
 *  Created on: 30/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ANALYTICALINTEGRAL_INL_
#define ANALYTICALINTEGRAL_INL_

#include <stdexcept>
#include <cassert>

namespace hydra {


template<typename Functor, size_t N>
class AnalyticalIntegral: protected IntegrationFormula<Functor,N>,
							public Integral< AnalyticalIntegral<Functor, N> >
{

public:

	typedef void hydra_analytical_integral_tag;

	AnalyticalIntegral()=delete;

	AnalyticalIntegral(GReal_t (&lower_limit)[N], GReal_t (&upper_limit)[N])
	{
		for(size_t i =0; i<N; i++ ){

			fLowerLimit[i] = lower_limit[i];
			fUpperLimit[i] = upper_limit[i];
			if( fLowerLimit[i] > fUpperLimit[i])
				throw std::invalid_argument("hydra::AnalyticalIntegral: Illegal integration domain definition  fLowerLimit > fUpperLimit.");
		}
	}

	AnalyticalIntegral(AnalyticalIntegral<Functor, N>const& other):
		IntegrationFormula<Functor,N>(other)
	{

		for(size_t i =0; i<N; i++ ){

			fLowerLimit[i] = other.GetLowerLimit(i);
			fUpperLimit[i] = other.GetUpperLimit(i);
		}

	}


	AnalyticalIntegral<Functor, N>&
	operator=(AnalyticalIntegral<Functor, N>const& other)
	{
		if(this == &other) return *this;

		IntegrationFormula<Functor,1>::operator=(other);

		for(size_t i =0; i<N; i++ ){

			fLowerLimit[i] = other.GetLowerLimit(i);
			fUpperLimit[i] = other.GetUpperLimit(i);
		}

		return *this;
	}

	inline std::pair<GReal_t, GReal_t> operator()(Functor const& functor) const
	{
			return  static_cast<const IntegrationFormula<Functor,N>& >(*this).EvalFormula(functor,
					fLowerLimit, fUpperLimit );
	}

	inline std::pair<GReal_t, GReal_t> Integrate(Functor const& functor) const
	{
		return  static_cast<const IntegrationFormula<Functor,N>& >(*this).EvalFormula(functor,
				fLowerLimit, fUpperLimit );
	}

	inline std::pair<GReal_t, GReal_t> Integrate(Functor const& functor,
			double (&LowerLimit)[N], double (&UpperLimit)[N] ) const
	{
			return  static_cast<const IntegrationFormula<Functor,N>& >(*this).EvalFormula(functor,
					LowerLimit, UpperLimit );
	}

	double GetLowerLimit(size_t i) const {
		return fLowerLimit[i];
	}

	void SetLowerLimit(size_t i, double value) {

		fLowerLimit[i]=value;
	}

	double GetUpperLimit(size_t i) const {
		return fUpperLimit[i];
	}

	void SetUpperLimit(size_t i, double value){

		fUpperLimit[i]=value;
	}

private:

   double fLowerLimit[N];
   double fUpperLimit[N];
};


template<typename Functor>
class AnalyticalIntegral<Functor, 1>:protected IntegrationFormula<Functor,1>,
								   public Integral< AnalyticalIntegral<Functor, 1> >
{

public:

	AnalyticalIntegral()=delete;

	AnalyticalIntegral(GReal_t lower_limit, GReal_t upper_limit ):
		fLowerLimit(lower_limit),
		fUpperLimit(upper_limit)
	{
		if( fLowerLimit > fUpperLimit)
			throw std::invalid_argument("hydra::AnalyticalIntegral: Illegal integration domain definition  fLowerLimit > fUpperLimit.");
	}

	AnalyticalIntegral( AnalyticalIntegral<Functor, 1> const& other ):
		IntegrationFormula<Functor,1>(other),
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	AnalyticalIntegral<Functor, 1>& operator=( AnalyticalIntegral<Functor, 1> const& other )
	{
		if(this == &other) return *this;
		IntegrationFormula<Functor,1>::operator=(other);
		fLowerLimit = other.GetLowerLimit();
		fUpperLimit = other.GetUpperLimit();

		return *this;
	}


	inline std::pair<GReal_t, GReal_t>
	Integrate(Functor const& functor) const
	{
		return  this->EvalFormula(functor,	fLowerLimit, fUpperLimit );
	}

	inline std::pair<GReal_t, GReal_t>
	Integrate(Functor const& functor, double LowerLimit, double UpperLimit) const
	{
		return this->EvalFormula(functor, LowerLimit, UpperLimit );
	}



	double GetLowerLimit() const {
		return fLowerLimit;
	}

	void SetLowerLimit(double lowerLimit) {
		fLowerLimit = lowerLimit;
	}

	double GetUpperLimit() const {
		return fUpperLimit;
	}

	void SetUpperLimit(double upperLimit) {
		fUpperLimit = upperLimit;
	}

private:

   double fLowerLimit;
   double fUpperLimit;
};



}  // namespace hydra


#endif /* ANALYTICALINTEGRAL_INL_ */
