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
 * BreitWignerLineShape.h
 *
 *  Created on: 26/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BREITWIGNERLINESHAPE_H_
#define BREITWIGNERLINESHAPE_H_

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
__host__ __device__   inline
double BarrierFactor(const double d, const double p0, const double p);

/**
 * @BreitWignerLineShape
 *
 * Breit-Wigner line shape for 3 body resonant decays \f$ A -> r c , r-> a b\f$ ,
 * where A is a "long-lived" particle and \f$ a, b\f$ and \f$c\f$ are the final states.
 * The lineshape is defined by the expression:
 *
 * \f[
 *  R(m_{a,b}|m_0,\Lambda_0) = B'_{L_A}(d, p_0, p)(\frac{p}{m_A})^{L_A} \times \\
 *  		BW(m_{a,b}|m_0,\Lambda_0) \times B'_{L_r}(d, q_0, q)(\frac{q}{q_r})^{L_r}
 * \f]
 *
 * where Breit-Wigner amplitude is given by:
 *
 *\f[ BW(m_{ab}|m_0,\Lambda_0)= \frac{1}{m_0^2 - m_{ab}^2 - im_0\Lambda(m_{ab})} \f]
 *
 *and
 *\f[  \Lambda(m_{ab}) = \Lambda_0(\frac{q}{q_0})^{2L_{r}+1}\frac{m_0}{m}B'_{L_r}(d, q_0, q)\f]
 *
 *@tparam ResonanceWave hydra::Wave resonance decay vertex wave
 *@tparam MotherWave hydra::Wave mother particle decay vertex wave
 */
template<Wave ResonanceWave, Wave MotherWave=SWave, unsigned int ArgIndex=0>
class BreitWignerLineShape : public BaseFunctor<BreitWignerLineShape< ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>
{
	using BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>::_par;

public:

	BreitWignerLineShape()=delete;

	/**
	 *
	 * @param mass resonance mass.
	 * @param width resonance width.
	 * @param mother_mass resonance mother mass.
	 * @param daugther1_mass resonance daughter particle 1 mass
	 * @param daugther2_mass resonance daughter particle 2 mass
	 * @param daugther3_mass daughter particle 2 mass
	 * @param radi decay vertex radio.
	 */
	BreitWignerLineShape(Parameter const& mass, Parameter const& width,
			double mother_mass,
			double daugther1_mass, double daugther2_mass, double daugther3_mass,
			double radi):
		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>{mass,width},
		fDaughter1Mass(daugther1_mass),
		fDaughter2Mass(daugther2_mass),
		fDaughter3Mass(daugther3_mass),
		fMotherMass(mother_mass),
		fRadi(radi)
	{}

	__host__  __device__
	BreitWignerLineShape(BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>  const& other):
		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>(other),
		fDaughter1Mass(other.GetDaughter1Mass()),
		fDaughter2Mass(other.GetDaughter2Mass()),
		fDaughter3Mass(other.GetDaughter3Mass()),
		fMotherMass(other.GetMotherMass()),
		fRadi(other.GetRadi())
		{}

	__host__  __device__
	BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>&
	operator=(BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>  const& other)
	{
		if(this==&other) return  *this;

		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>,
			hydra::complex<double>, 2>::operator=(other);

		fDaughter1Mass= other.GetDaughter1Mass();
		fDaughter2Mass= other.GetDaughter2Mass();
		fDaughter3Mass= other.GetDaughter3Mass();
		fMotherMass= other.GetMotherMass();
		fRadi= other.GetRadi();

		 return  *this;
	}

	__host__  __device__ inline
	double GetDaughter1Mass() const {
		return fDaughter1Mass;
	}

	__host__  __device__ inline
	void SetDaughter1Mass(double daughter1Mass) {
		fDaughter1Mass = daughter1Mass;
	}

	__host__  __device__ inline
	double GetDaughter2Mass() const {
		return fDaughter2Mass;
	}

	__host__  __device__ inline
	void SetDaughter2Mass(double daughter2Mass) {
		fDaughter2Mass = daughter2Mass;
	}

	__host__  __device__ inline
	double GetDaughter3Mass() const {
		return fDaughter3Mass;
	}

	__host__  __device__ inline
	void SetDaughter3Mass(double daughter3Mass) {
		fDaughter3Mass = daughter3Mass;
	}

	__host__  __device__ inline
	double GetMotherMass() const {
		return fMotherMass;
	}

	__host__  __device__ inline
	void SetMotherMass(double motherMass) {
		fMotherMass = motherMass;
	}

	__host__  __device__ inline
	double GetRadi() const {
		return fRadi;
	}

	__host__  __device__ inline
	void SetRadi(double radi) {
		fRadi = radi;
	}

	template<typename T>
	__host__ __device__ inline
	hydra::complex<double> Evaluate(unsigned int n, T*x)  const	{

		const double m = x[ArgIndex] ;

		const double resonance_mass  = _par[0];
		const double resonance_width = _par[1];

		return  m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fDaughter3Mass) ?
				LineShape(m,resonance_mass, resonance_width): hydra::complex<double>(0.0, 0.0) ;

	}

	template<typename T>
	__host__ __device__ inline
	hydra::complex<double> Evaluate(T x)  const {

		double m =  get<ArgIndex>(x);

		const double resonance_mass  = _par[0];
		const double resonance_width = _par[1];

		return  m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fDaughter3Mass) ?
				LineShape(m,resonance_mass, resonance_width): hydra::complex<double>(0.0, 0.0) ;
	}

private:


	   __host__ __device__  inline
	 double  Width( const double m, const double resonance_mass, const double resonance_width,
			   const double  p0, const double  p) const {

		 const double  B = BarrierFactor<ResonanceWave>( fRadi, p0,  p);

	  	 return resonance_width*\
	  			pow<double, 2*ResonanceWave+1>(p/p0)*\
	  			(resonance_mass/m)*\
	  			B*B;

	   }

	 __host__ __device__   inline
	 hydra::complex<double> LineShape(const double m, const double resonance_mass, const double resonance_width ) const {

		 const double p0 = pmf( fMotherMass, resonance_mass, fDaughter3Mass);
		 const double q0 = pmf( resonance_mass, fDaughter1Mass, fDaughter2Mass);

		 const double p  = pmf( fMotherMass, m, fDaughter3Mass);
		 const double q  = pmf( m, fDaughter1Mass, fDaughter2Mass);

		 const double width = Width( m, resonance_mass, resonance_width, q0, q);

		 hydra::complex<double> numerator( BarrierFactor<MotherWave>(fRadi, p0, p)*\
				 pow<double, MotherWave>(p/p0)*\
				 BarrierFactor<ResonanceWave>(fRadi, q0, q)*\
				 pow<double,ResonanceWave>(q/q0) , 0);

		 hydra::complex<double> denominator(m*m - resonance_mass*resonance_mass,  -resonance_mass*width);

		 return hydra::complex<double>(numerator/denominator) ;

	 }

	double fDaughter1Mass;
	double fDaughter2Mass;
	double fDaughter3Mass;
	double fMotherMass;
	double fRadi;

};



template<> __host__ __device__   inline
double BarrierFactor<hydra::SWave>(const double radi, const double p0, const double p) {
	return 1.0;
}

template<> __host__ __device__   inline
double BarrierFactor<hydra::PWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (1 + z0*z0)/(1 + z*z) );
}

template<> __host__ __device__   inline
double BarrierFactor<hydra::DWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt((9 + 3*z0*z0 + z0*z0*z0*z0)/(9 + 3*z*z + z*z*z*z));
}

template<> __host__ __device__   inline
double BarrierFactor<hydra::FWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (225 + 45*z0*z0 + 6*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0)
			  / (225 + 45*z*z   + 6*z*z*z*z + z*z*z*z*z*z) );
}


template<>  __host__ __device__   inline
double BarrierFactor<hydra::GWave>(const double radi, const double p0, const double p)  {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (11025 + 1575*z0*z0 + 135*z0*z0*z0*z0
			+ 10*z0*z0*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0*z0*z0)
			/ (11025 + 1575*z*z   + 135*z*z*z*z
					+ 10*z*z*z*z*z*z  + z*z*z*z*z*z*z*z) );

}


template<> __host__ __device__   inline
double BarrierFactor<hydra::HWave>(const double radi, const double p0, const double p) {

	double z  =  radi*p;
	double z0 =  radi*p0;

	return ::sqrt( (893025 + 99225*z0*z0 + 6300*z0*z0*z0*z0  + 315*z0*z0*z0*z0*z0*z0
			+ 15*z0*z0*z0*z0*z0*z0*z0*z0 + z0*z0*z0*z0*z0*z0*z0*z0*z0*z0)
			/ (893025 + 99225*z*z   + 6300*z*z*z*z  + 315*z*z*z*z*z*z
					+ 15*z*z*z*z*z*z*z*z  + z*z*z*z*z*z*z*z*z*z) );

}


}  // namespace hydra


#endif /* BREITWIGNERLINESHAPE_H_ */
