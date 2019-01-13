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
 * GaussianKDE.h
 *
 *  Created on: Apr 12, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSIANKDE_H_
#define GAUSSIANKDE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/CubicSpiline.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

namespace hydra {

/**
 *  \ingroup common_functions
 *  \class GaussianKDE
 */
template< size_t NBins, size_t ArgIndex=0>
class GaussianKDE: public BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>
{
	using BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>::_par;

public:

	struct Kernel
	{
		Kernel()=delete;

		__hydra_host__ __hydra_device__
		Kernel(double h, double x):
		  fX(x),
		  fH(h)
		{}

		__hydra_host__ __hydra_device__
		Kernel(Kernel const& other):
			fX(other.fX),
			fH(other.fH)
		{}

		__hydra_host__ __hydra_device__
		inline 	Kernel& operator=(Kernel const& other){

			if(this == &other) return *this;

			fX = other.fX ;
			fH = other.fH ;

			return *this;
		}

		__hydra_host__ __hydra_device__
		inline 	double operator()(double x){

			double m = (x - fX)/fH;
			return  hydra::math_constants::inverse_sqrt2Pi*exp(-0.5*m*m);
		}

		double fX;
		double fH;

	};

	//-------------------------

	GaussianKDE() = delete;

	template<typename Iterator>
	GaussianKDE(double min, double max, double h, Iterator begin, Iterator end):
	BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>()
	{
		fSpiline=BuildKDE(min, max, h, begin, end);
	}


	__hydra_host__ __hydra_device__
	GaussianKDE(GaussianKDE<NBins, ArgIndex> const& other):
	BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>(other),
	fSpiline(other.GetSpiline())
	{}

	__hydra_host__ __hydra_device__
	inline 	GaussianKDE<NBins, ArgIndex>&
	operator=(GaussianKDE<NBins, ArgIndex> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<GaussianKDE<NBins, ArgIndex>, double, 0>::operator=(other);
		fSpiline=other.GetSpiline();
		return *this;
	}

	__hydra_host__ __hydra_device__
	inline 	const CubicSpiline<NBins, ArgIndex>& GetSpiline() const {
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
	__hydra_host__ __hydra_device__
	inline 	CubicSpiline<NBins>  BuildKDE(double min, double max, double h, Iterator begin, Iterator end);

	CubicSpiline<NBins> fSpiline;


};


}  // namespace hydra

#include <hydra/functions/detail/GaussianKDE.inl>

#endif /* GAUSSIANKDE_H_ */
