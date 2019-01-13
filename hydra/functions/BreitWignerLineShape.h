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
 * BreitWignerLineShape.h
 *
 *  Created on: 26/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BREITWIGNERLINESHAPE_H_
#define BREITWIGNERLINESHAPE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Complex.h>
#include <hydra/functions/Utils.h>
#include <hydra/functions/BlattWeisskopfFunctions.h>
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
 * \class BreitWignerLineShape
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
			double daugther1_mass, double daugther2_mass, double bachelor_mass,
			double radi):
		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>{mass,width},
		fDaughter1Mass(daugther1_mass),
		fDaughter2Mass(daugther2_mass),
		fBachelorMass(bachelor_mass),
		fMotherMass(mother_mass),
		fRadi(radi)
	{}

	__hydra_host__  __hydra_device__
	BreitWignerLineShape(BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>  const& other):
		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>(other),
		fDaughter1Mass(other.GetDaughter1Mass()),
		fDaughter2Mass(other.GetDaughter2Mass()),
		fBachelorMass(other.GetBachelorMass()),
		fMotherMass(other.GetMotherMass()),
		fRadi(other.GetRadi())
		{}

	__hydra_host__  __hydra_device__
	BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>&
	operator=(BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>  const& other)
	{
		if(this==&other) return  *this;

		BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>,
			hydra::complex<double>, 2>::operator=(other);

		fDaughter1Mass= other.GetDaughter1Mass();
		fDaughter2Mass= other.GetDaughter2Mass();
		fBachelorMass= other.GetBachelorMass();
		fMotherMass= other.GetMotherMass();
		fRadi= other.GetRadi();

		 return  *this;
	}

	__hydra_host__  __hydra_device__ inline
	double GetDaughter1Mass() const {
		return fDaughter1Mass;
	}

	__hydra_host__  __hydra_device__ inline
	void SetDaughter1Mass(double daughter1Mass) {
		fDaughter1Mass = daughter1Mass;
	}

	__hydra_host__  __hydra_device__ inline
	double GetDaughter2Mass() const {
		return fDaughter2Mass;
	}

	__hydra_host__  __hydra_device__ inline
	void SetDaughter2Mass(double daughter2Mass) {
		fDaughter2Mass = daughter2Mass;
	}

	__hydra_host__  __hydra_device__ inline
	double GetBachelorMass() const {
		return fBachelorMass;
	}

	__hydra_host__  __hydra_device__ inline
	void SetBachelorMass(double daughter3Mass) {
		fBachelorMass = daughter3Mass;
	}

	__hydra_host__  __hydra_device__ inline
	double GetMotherMass() const {
		return fMotherMass;
	}

	__hydra_host__  __hydra_device__ inline
	void SetMotherMass(double motherMass) {
		fMotherMass = motherMass;
	}

	__hydra_host__  __hydra_device__ inline
	double GetRadi() const {
		return fRadi;
	}

	__hydra_host__  __hydra_device__ inline
	void SetRadi(double radi) {
		fRadi = radi;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	hydra::complex<double> Evaluate(unsigned int, const T*x)  const	{

		const double m = x[ArgIndex] ;

		const double resonance_mass  = _par[0];
		const double resonance_width = _par[1];

		return  m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fBachelorMass) ?
				LineShape(m,resonance_mass, resonance_width): hydra::complex<double>(0.0, 0.0) ;

	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	hydra::complex<double> Evaluate(T& x)  const {

		double m =  get<ArgIndex>(x);

		const double resonance_mass  = _par[0];
		const double resonance_width = _par[1];

		return  m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fBachelorMass) ?
				LineShape(m,resonance_mass, resonance_width): hydra::complex<double>(0.0, 0.0) ;
	}



private:


	   __hydra_host__ __hydra_device__  inline
	 double  Width( const double m, const double resonance_mass, const double resonance_width,
			   const double  p0, const double  p) const {

		 const double  B = BarrierFactor<ResonanceWave>( fRadi, p0,  p);

	  	 return resonance_width*\
	  			pow<double, 2*ResonanceWave+1>(p/p0)*\
	  			(resonance_mass/m)*\
	  			B*B;

	   }

	 __hydra_host__ __hydra_device__   inline
	 hydra::complex<double> LineShape(const double m, const double resonance_mass, const double resonance_width ) const {

		 const double p0 = pmf( fMotherMass, resonance_mass, fBachelorMass);
		 const double q0 = pmf( resonance_mass, fDaughter1Mass, fDaughter2Mass);

		 const double p  = pmf( fMotherMass, m, fBachelorMass);
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
	double fBachelorMass;
	double fMotherMass;
	double fRadi;

};



}  // namespace hydra


#endif /* BREITWIGNERLINESHAPE_H_ */
