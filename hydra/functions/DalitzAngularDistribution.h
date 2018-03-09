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

template<unsigned int P1, unsigned int P2>
double cos_decay_angle(const double M0Sq, const double M1Sq, const double M2Sq, const double M3Sq,
		const double M12, const double M23, const double M31);


template<Wave L>
class DalitzAngularDistribution<L>:public BaseFunctor<DalitzAngularDistribution<L>, double, 0>
{

public:

	DalitzAngularDistribution(double mother_mass, double daughter1_mass, double daughter2_mass, double bachelor_mass):
		fDaughter1Mass(daughter1_mass),
		fDaughter2Mass(daughter2_mass),
		fBachelorMass(bachelor_mass),
		fMotherMass(mother_mass)
    {}

	__hydra_host__  __hydra_device__
	DalitzAngularDistribution( DalitzAngularDistribution<L> const& other):
	BaseFunctor<DalitzAngularDistribution<L>,double, 0>(other),
	fDaughter1Mass(other.GetDaughter1Mass()),
	fDaughter2Mass(other.GetDaughter2Mass()),
	fBachelorMass(other.GetBachelorMass()),
	fMotherMass(other.GetMotherMass())
	{}

	__hydra_host__  __hydra_device__ inline
	DalitzAngularDistribution<L>&
	operator=( DalitzAngularDistribution<L> const& other){

		if(this==&other) return  *this;

		BaseFunctor<DalitzAngularDistribution<L>,double, 0>::operator=(other);

		this->fDaughter1Mass = other.GetDaughter1Mass();
		this->fDaughter2Mass = other.GetDaughter2Mass();
		this->fBachelorMass  = other.GetBachelorMass();
		this->fMotherMass = other.GetMotherMass();

		return  *this;
	}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int , double* invariant_masses)  const {

		double M12Sq  = pow<2>(invariant_masses[0]);
		double M23Sq  = pow<2>(invariant_masses[1]);
		double M31Sq  = pow<2>(invariant_masses[2]);

		return zemach_function<L>(cos_decay_angle(M12Sq, M23Sq, M31Sq));

	}


	double GetBachelorMass() const {
		return fBachelorMass;
	}

	void SetBachelorMass(double bachelorMass) {
		fBachelorMass = bachelorMass;
	}

	double GetDaughter1Mass() const {
		return fDaughter1Mass;
	}

	void SetDaughter1Mass(double daughter1Mass) {
		fDaughter1Mass = daughter1Mass;
	}

	double GetDaughter2Mass() const {
		return fDaughter2Mass;
	}

	void SetDaughter2Mass(double daughter2Mass) {
		fDaughter2Mass = daughter2Mass;
	}

	double GetMotherMass() const {
		return fMotherMass;
	}

	void SetMotherMass(double motherMass) {
		fMotherMass = motherMass;
	}

private:

	 //Kajantie & Bycling Particle Kinematics pg 24, eq 6.10
	__hydra_host__ __hydra_device__ inline
	double triangular_function(const double M12){

		return (pow<2>(M12) - pow<2>(fDaughter1Mass + fDaughter2Mass))*(pow<2>(M12) - pow<2>(fDaughter1Mass - fDaughter2Mass));
	}

	__hydra_host__ __hydra_device__ inline
	double cos_decay_angle(const double M0Sq, const double M1Sq, const double M2Sq, const double M3Sq,
			const double M12Sq, const double M23Sq, const double M31Sq){

		return ((M0Sq - M23Sq - M1Sq)*( M23Sq + M2Sq -M3Sq ) + 2*M23*(M1Sq + M2Sq - M12Sq))/
				::sqrt(triangular_function(M0Sq, M23Sq, M1Sq ))*::sqrt(triangular_function(M23Sq, M2Sq, M3Sq ));
	}

	double fDaughter1Mass;
	double fDaughter2Mass;
	double fBachelorMass;
	double fMotherMass;

};



template<>
__hydra_host__ __hydra_device__ inline
double cos_decay_angle<2,3>(const double M0Sq, const double M1Sq, const double M2Sq, const double M3Sq,
		const double M12, const double M23, const double M31){

	return ((M0Sq - M31 - M2Sq)*( M31 + M3Sq -M1Sq ) + 2*M31*(M2Sq + M3Sq - M23))/
			::sqrt(triangular_function(M0Sq, M31, M2Sq ))*::sqrt(triangular_function(M31, M3Sq, M1Sq ));
}

template<>
__hydra_host__ __hydra_device__ inline
double cos_decay_angle<3,1>(const double M0Sq, const double M1Sq, const double M2Sq, const double M3Sq,
		const double M12Sq, const double M23Sq, const double M31Sq){

	return ((M0Sq - M12Sq - M3Sq)*( M12Sq + M1Sq -M2Sq ) + 2*M12Sq*(M3Sq + M1Sq - M31Sq))/
			::sqrt(triangular_function(M0Sq, M12Sq, M3Sq ))*::sqrt(triangular_function(M12Sq, M1Sq, M2Sq ));

}

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
