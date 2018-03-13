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
 * DalitzAngularDistribution.h
 *
 *  Created on: 09/03/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZANGULARDISTRIBUTION_H_
#define DALITZANGULARDISTRIBUTION_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/functions/Utils.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>


namespace hydra {

/**
 * \ingroup functors
 *
 */

template<Wave L>
double zemach_function(const double x);

template<Wave L, unsigned int CHANNEL=1>
class DalitzAngularDistribution:public BaseFunctor<DalitzAngularDistribution<L,CHANNEL >, double, 0>
{
	static_assert(CHANNEL>0 || CHANNEL<3, "[Hydra::DalitzAngularDistribution]:  CHANNEL template parameter allowed values are {1,2,3}" );
	// 1 -> (0,1)
	// 2 -> (1,2)
	// 3 -> (2,0)
	static constexpr unsigned int _M12=CHANNEL-1;
	static constexpr unsigned int _M23= CHANNEL==3?0:CHANNEL;

public:


	DalitzAngularDistribution()=delete;

	DalitzAngularDistribution(double mother_mass, double daughter1_mass, double daughter2_mass, double bachelor_mass):
		fDaughter1Mass(daughter1_mass),
		fDaughter2Mass(daughter2_mass),
		fBachelorMass(bachelor_mass),
		fMotherMass(mother_mass)
    {}

	__hydra_host__  __hydra_device__
	DalitzAngularDistribution( DalitzAngularDistribution<L,CHANNEL> const& other):
	BaseFunctor<DalitzAngularDistribution<L,CHANNEL>,double, 0>(other),
	fDaughter1Mass(other.GetDaughter1Mass()),
	fDaughter2Mass(other.GetDaughter2Mass()),
	fBachelorMass(other.GetBachelorMass()),
	fMotherMass(other.GetMotherMass())
	{}

	__hydra_host__  __hydra_device__ inline
	DalitzAngularDistribution<L,CHANNEL>&
	operator=( DalitzAngularDistribution<L,CHANNEL> const& other){

		if(this==&other) return  *this;

		BaseFunctor<DalitzAngularDistribution<L,CHANNEL>,double, 0>::operator=(other);

		this->fDaughter1Mass = other.GetDaughter1Mass();
		this->fDaughter2Mass = other.GetDaughter2Mass();
		this->fBachelorMass  = other.GetBachelorMass();
		this->fMotherMass    = other.GetMotherMass();

		return  *this;
	}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int , double* invariant_masses)  const {

		double M12Sq  = pow<double,2>(invariant_masses[_M12]);
		double M23Sq  = pow<double,2>(invariant_masses[_M23]);

		return zemach_function<L>(cos_decay_angle(M12Sq, M23Sq));
	}


	__hydra_host__ __hydra_device__ inline
	double operator()(double M12, double M23 )  const {

			double M12Sq  = pow<double,2>(M12);
			double M23Sq  = pow<double,2>(M23);

			return zemach_function<L>(cos_decay_angle(M12Sq, M23Sq));
	}

	__hydra_host__ __hydra_device__ inline
	double GetBachelorMass() const {
		return fBachelorMass;
	}

	__hydra_host__ __hydra_device__ inline
	void SetBachelorMass(double bachelorMass) {
		fBachelorMass = bachelorMass;
	}

	__hydra_host__ __hydra_device__ inline
	double GetDaughter1Mass() const {
		return fDaughter1Mass;
	}

	__hydra_host__ __hydra_device__ inline
	void SetDaughter1Mass(double daughter1Mass) {
		fDaughter1Mass = daughter1Mass;
	}

	__hydra_host__ __hydra_device__ inline
	double GetDaughter2Mass() const {
		return fDaughter2Mass;
	}

	__hydra_host__ __hydra_device__ inline
	void SetDaughter2Mass(double daughter2Mass) {
		fDaughter2Mass = daughter2Mass;
	}

	__hydra_host__ __hydra_device__ inline
	double GetMotherMass() const {
		return fMotherMass;
	}

	__hydra_host__ __hydra_device__ inline
	void SetMotherMass(double motherMass) {
		fMotherMass = motherMass;
	}

private:

	 //Kajantie & Bycling Particle Kinematics pg 24, eq 6.10
	__hydra_host__ __hydra_device__ inline
	double triangular_function(const double M12){

		return (pow<double,2>(M12) - pow<double,2>(fDaughter1Mass + fDaughter2Mass))*(pow<double,2>(M12) - pow<double,2>(fDaughter1Mass - fDaughter2Mass));
	}

	__hydra_host__ __hydra_device__ inline
	double cos_decay_angle(const double M12Sq, const double M23Sq){

		return ((pow<double,2>(fMotherMass) - M23Sq - pow<double,2>(fDaughter1Mass))*\
				( M23Sq + pow<double,2>(fDaughter2Mass) - pow<double,2>(fBachelorMass) )\
				+ 2*M23Sq*(pow<double,2>(fDaughter1Mass) + pow<double,2>(fDaughter2Mass) - M12Sq))/\
				::sqrt(triangular_function(pow<double,2>(fMotherMass), M23Sq, pow<double,2>(fDaughter1Mass) ))\
				 *::sqrt(triangular_function(M23Sq, pow<double,2>(fDaughter2Mass), pow<double,2>(fBachelorMass)));
	}

	double fDaughter1Mass;
	double fDaughter2Mass;
	double fBachelorMass;
	double fMotherMass;

};



/*
 * Zemach angular distribution specializations
 */
template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<SWave>(const double ){

	return 1.0;
}

template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<PWave>(const double x){

	return -x;
}

template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<DWave>(const double x){

	return 0.5*(3.0*pow<double, 2>(x) -1) ;
}

template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<FWave>(const double x){

	return -0.5*(5.0*pow<double, 3>(x) - 3.0*x);
}

template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<GWave>(const double x){

	return 0.125*(35.0*pow<double, 4>(x) - 30.0*pow<double, 2>(x) + 3);
}

template<>
__hydra_host__ __hydra_device__ inline
double zemach_function<HWave>(const double x){

	return -0.125*(63.0*pow<double, 5>(x) - 70.0*pow<double, 3>(x) + 15.0*x);
}



}  // namespace hydra




#endif /* DALITZANGULARDISTRIBUTION_H_ */
