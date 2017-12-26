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

enum Wave{ SWave=0, PWave, DWave, FWave, GWave, HWave };

template<hydra::Wave L>
__host__ __device__   inline
double BarrierFactor(const double radi, const double p0, const double p);

template<Wave ResonanceWave, Wave MotherWave=SWave, unsigned int ArgIndex=0>
class BreitWignerLineShape : public BaseFunctor<BreitWignerLineShape< ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>
{
	using BaseFunctor<BreitWignerLineShape<ResonanceWave,MotherWave,ArgIndex>, hydra::complex<double>, 2>::_par;

public:

	BreitWignerLineShape()=delete;

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

		return  LineShape(m,resonance_mass, resonance_width);

	}

	template<typename T>
	__host__ __device__ inline
	hydra::complex<double> Evaluate(T x)  const {

		double m =  get<ArgIndex>(x);

		const double resonance_mass  = _par[0];
		const double resonance_width = _par[1];

		return  LineShape(m,resonance_mass, resonance_width);
	}

private:

	   __host__  __device__
	   inline double pmf( const double mother_mass, const double daughter1_mass, const double daughter2_mass) const{
		   double mother_mass_sq  = mother_mass*mother_mass;

		   return  ::sqrt( ( mother_mass_sq - ( daughter1_mass + daughter2_mass)*( daughter1_mass + daughter2_mass))
				   *( mother_mass_sq - ( daughter1_mass - daughter2_mass)*( daughter1_mass - daughter2_mass)) )/2*mother_mass;
	   }



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
				 pow<double, MotherWave>(p/fMotherMass)*\
				 BarrierFactor<ResonanceWave>(fRadi, q0, q)*\
				 pow<double,ResonanceWave>(q/resonance_mass) , 0);

		 hydra::complex<double> denominator(resonance_mass*resonance_mass - m*m, - resonance_mass*width);

		 return hydra::complex<double>(numerator/denominator);

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
