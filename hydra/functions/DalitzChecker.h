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
 * DalitzChecker.h
 *
 *  Created on: Mar 15, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZCHECKER_H_
#define DALITZCHECKER_H_

namespace hydra {

class DalitzChecker:public BaseFunctor<DalitzChecker, bool, 0>
{

public:

	DalitzChecker()=delete;

	DalitzChecker(double mother_mass, double daughter1_mass, double daughter2_mass, double daughter3_mass):
		fDaughter1Mass(daughter1_mass),
		fDaughter2Mass(daughter2_mass),
		fDaughter3Mass(daughter3_mass),
		fMotherMass(mother_mass)
    {}

	__hydra_host__  __hydra_device__
	DalitzChecker( DalitzChecker const& other):
	BaseFunctor<DalitzChecker,double, 0>(other),
	fDaughter1Mass(other.GetDaughter1Mass()),
	fDaughter2Mass(other.GetDaughter2Mass()),
	fDaughter3Mass(other.GetDaughter3Mass()),
	fMotherMass(other.GetMotherMass())
	{}

	__hydra_host__  __hydra_device__ inline
	DalitzChecker&
	operator=( DalitzChecker const& other){

		if(this==&other) return  *this;

		BaseFunctor<DalitzChecker,double, 0>::operator=(other);

		this->fDaughter1Mass = other.GetDaughter1Mass();
		this->fDaughter2Mass = other.GetDaughter2Mass();
		this->fDaughter3Mass = other.GetDaughter3Mass();
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
	double GetDaughter3Mass() const {
		return fDaughter3Mass;
	}

	__hydra_host__ __hydra_device__ inline
	void SetDaughter3Mass(double bachelorMass) {
		fDaughter3Mass = bachelorMass;
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
				( M23Sq + pow<double,2>(fDaughter2Mass) - pow<double,2>(fDaughter3Mass) )\
				+ 2*M23Sq*(pow<double,2>(fDaughter1Mass) + pow<double,2>(fDaughter2Mass) - M12Sq))/\
				::sqrt(triangular_function(pow<double,2>(fMotherMass), M23Sq, pow<double,2>(fDaughter1Mass) ))\
				 *::sqrt(triangular_function(M23Sq, pow<double,2>(fDaughter2Mass), pow<double,2>(fDaughter3Mass)));
	}

	double fDaughter1Mass;
	double fDaughter2Mass;
	double fDaughter3Mass;
	double fMotherMass;

};



}  // namespace hydra

#endif /* DALITZCHECKER_H_ */
