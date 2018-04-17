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
 * GaussianKDE.h
 *
 *  Created on: Apr 12, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSIANKDE_H_
#define GAUSSIANKDE_H_

#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

namespace hydra {

template< size_t NBins, size_t ArgIndex=0>
class GaussianKDE: public BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>
{
	using BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>::_par;

public:

	struct Kernel
	{
		Kernel()=delete;

		Kernel(double h, double x):
			BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>(),
			fX(x), fH(h){}

		Kernel(Kernel const& other):
			BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>(other),
			fX(other.fX), fH(other.fH){}

		Kernel& operator=(Kernel const& other){

			if(this == &other) return *this;

			fX = other.fX ;
			fH = other.fH ;

			return *this;
		}

		double operator()(double x){

			double m = (x - fX)/fH;
			return  hydra::math_constants::inverse_sqrt2Pi*exp(-0.5*m*m);
		}

		double fX;
		double fH;

	};

	//-------------------------

	GaussianKDE() = delete;

	template<typename Iterator>
	GaussianKDE(double min, double max, Iterator begin, Iterator end, double h)
	{
		BuildKDE(min, max, h, begin, end);
	}


	GaussianKDE(GaussianKDE<NBins, ArgIndex> const& other):
	fSpiline(other.GetSpiline())
	{}

	GaussianKDE<NBins, ArgIndex>&
	operator=(GaussianKDE<NBins, ArgIndex> const& other)
	{
		if(this == &other) return *this;
		fH=other.GetH();
		fSpiline=other.GetSpiline();
		return *this;
	}


	double GetH() const {
		return fH;
	}

	void SetH(double h) {
		fH = h;
		BuildKDE(begin, end);
	}

	double GetMaximum() const {
		return fMaximum;

	}

	void SetMaximum(double maximum) {
		fMaximum = maximum;
		BuildKDE(begin, end);
	}

	double GetMinimum() const {
		return fMinimum;
	}

	void SetMinimum(double minimum) {
		fMinimum = minimum;
	}


	const CubicSpiline<NBins, ArgIndex>& GetSpiline() const {
		return fSpiline;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int n, T*x)  const {

		GReal_t X  = x[ArgIndex];

		GReal_t r = fSpiline( X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const {

		GReal_t X  = hydra::get<ArgIndex>(x); //mass

		GReal_t r = fSpiline( X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

private:

	template<typename Iterator>
	void BuildKDE(Iterator begin, Iterator end, double h);

	double fH;
	double fMinimum;
	double fMaximum;
	CubicSpiline<NBins,ArgIndex> fSpiline;


};


}  // namespace hydra

#include <hydra/functions/detail/GaussianKDE.inl>

#endif /* GAUSSIANKDE_H_ */
