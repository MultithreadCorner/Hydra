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
 * BlattWeisskopfFunctions.h
 *
 *  Created on: 14/01/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BLATTWEISSKOPFFUNCTIONS_H_
#define BLATTWEISSKOPFFUNCTIONS_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Complex.h>
#include <hydra/functions/Utils.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>

namespace hydra {


/**
 * \ingroup common_functions
 *
 * Blatt-Weisskopf B' functions
 *
 * These functions are normalized to give to give
 * \f$ B'_L = 1 \f$ for \f$ z = z_0 = (|p_0 |/d)^2\f$ where \f$ p_0\f$ is the value of p when
 * \f$ m_ab = m_r \f$. The centrifugal barrier is to be moved to the dynamical functions.
 *
 *	\f[
 *	B'_{0}(d,p_0,p) =1\\
 *	B'_{1}(d,p_0,p) = \sqrt{ \frac{1 +(p_0d)^2 }{ 1 +(pd)^2 } \\
 *	B'_{2}(d,p_0,p) = \sqrt{ \frac{9 + 3(p_0d)^2 + (p_0d)^4 }{9 + 3(pd)^2 + (pd)^4} \\
 *	B'_{3}(d,p_0,p) = \sqrt{ \frac{225 + 45(p_0d)^2 + 6(p_0d)^4 + (p_0d)^6 }{225 + 45(pd)^2 + 6(pd)^4 + (pd)^6 } \\
 *	B'_{4}(d,p_0,p) = \sqrt{ \frac{11025 + 1575(p_0d)^2 + 135(p_0d)^4 + 10(p_0d)^6 + (p_0 d)^8}{11025 + 1575(p_0d)^2 + 135(p_0d)^4 + 10(p_0d)^6 + (p_0 d)^8} \\
 *	B'_{5}(d,p_0,p) = \sqrt{ \frac{893025 + 99225(p_0d)^2 + 6300(p_0d)^4 + 315(p_0d)^6 + 15(p_0d)^8 + (p_0d)^10 }{893025 + 99225(pd)^2 + 6300(pd)^4 + 315(pd)^6 + 15(pd)^8 + (pd)^10}\\
 *	\f]
 *
 * @tparam L hydra::Wave vertex wave
 * @param radi decay vertex radio
 * @param p0 momentum of the resonance at the decaying particle rest frame with nominal mass.
 * @param p p momentum of the resonance at the decaying particle rest frame with a given invariant mass.
 * @return real number
 *
 * References:
 *  - J. Blatt and V. Weisskopf, Theoretical  Nuclear  Physics , New York: John Wiley & Sons (1952)
 *  - M. Lax, H. Feshbach J. Acoust Soc.Am. 1948 20:2, 108-124 (https://doi.org/10.1121/1.1906352)
 *  - S. U. Chung, Formulas for Angular-Momentum Barrier Factors, BNL-QGS-06-101
 */
template<hydra::Wave L>
__hydra_host__ __hydra_device__   inline
double BarrierFactor(const double d, const double p0, const double p);


template<> __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::SWave>(const double, const double, const double) {
	return 1.0;
}

template<> __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::PWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (1 + z0*z0)/(1 + z*z) );
}

template<> __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::DWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt((9 + 3*z0*z0 + z0*z0*z0*z0)/(9 + 3*z*z + z*z*z*z));
}

template<> __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::FWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (225 + 45*z0*z0 + 6*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0)
			  / (225 + 45*z*z   + 6*z*z*z*z + z*z*z*z*z*z) );
}


template<>  __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::GWave>(const double radi, const double p0, const double p)  {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (11025 + 1575*z0*z0 + 135*z0*z0*z0*z0
			+ 10*z0*z0*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0*z0*z0)
			/ (11025 + 1575*z*z   + 135*z*z*z*z
					+ 10*z*z*z*z*z*z  + z*z*z*z*z*z*z*z) );

}


template<> __hydra_host__ __hydra_device__   inline
double BarrierFactor<hydra::HWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (893025 + 99225*z0*z0 + 6300*z0*z0*z0*z0  + 315*z0*z0*z0*z0*z0*z0
			+ 15*z0*z0*z0*z0*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0*z0*z0*z0*z0)
			/ (893025 + 99225*z*z   + 6300*z*z*z*z  + 315*z*z*z*z*z*z
					+ 15*z*z*z*z*z*z*z*z  + z*z*z*z*z*z*z*z*z*z) );

}


}  // namespace hydra

#endif /* BLATTWEISSKOPFFUNCTIONS_H_ */
