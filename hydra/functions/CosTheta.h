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
 * CosTheta.h
 *
 *  Created on: 07/03/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COSTHETA_H_
#define COSTHETA_H_


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
double cos_decay_angle(const double M12, const double M23, const double M31);

template<Wave L, unsigned int P1, unsigned int P2>
class DalitzAngularDistribution;

template<Wave L, unsigned int P1, unsigned int P2>
class DalitzAngularDistribution<L, P1,P2>:public BaseFunctor<DalitzAngularDistribution<P1,P2>, double, 0>
{

public:

	DalitzAngularDistribution(double M0, double M1, double M2, double M3):
		fM0(M0),
		fM1(M1),
		fM2(M2),
		fM3(M3)
    {}

	__hydra_host__  __hydra_device__
	DalitzAngularDistribution( DalitzAngularDistribution<L,P1,P2> const& other):
	BaseFunctor<DalitzAngularDistribution<L,P1,P2>,double, 0>(other),
	fM0(other.GetM0()),
	fM1(other.GetM1()),
	fM2(other.GetM2()),
	fM3(other.GetM3())
	{}

	__hydra_host__  __hydra_device__ inline
	DalitzAngularDistribution<L,P1,P2>&	operator=( DalitzAngularDistribution<L,P1,P2> const& other){

		if(this==&other) return  *this;

		BaseFunctor<DalitzAngularDistribution<L,P1,P2>,double, 0>::operator=(other);

		this->fM0 = other.GetM0();
		this->fM1 = other.GetM1();
		this->fM2 = other.GetM2();
		this->fM3 = other.GetM3();

		return  *this;
	}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int , double* invariant_masses)  const {

		double M12  = invariant_masses[0];
		double M23  = invariant_masses[1];
		double M31  = invariant_masses[2];

		return cos_decay_angle<P1,P2>( M12, M23, M31);

	}


	__hydra_host__ __hydra_device__ inline
	double operator()(Vector4R const& p, Vector4R const& q, Vector4R const& d) const {

		return cos_decay_angle( p, q, d);

	}

	__hydra_host__ __hydra_device__ inline
	double GetM0() const {
		return fM0;
	}

	__hydra_host__ __hydra_device__ inline
	void SetM0(double m0) {
		fM0 = m0;
	}

	__hydra_host__ __hydra_device__ inline
	double GetM1() const {
		return fM1;
	}

	__hydra_host__ __hydra_device__ inline
	void SetM1(double m1) {
		fM1 = m1;
	}

	__hydra_host__ __hydra_device__ inline
	double GetM2() const {
		return fM2;
	}

	__hydra_host__ __hydra_device__ inline
	void SetM2(double m2) {
		fM2 = m2;
	}

	__hydra_host__ __hydra_device__ inline
	double GetM3() const {
		return fM3;
	}

	__hydra_host__ __hydra_device__ inline
	void SetM3(double m3) {
		fM3 = m3;
	}

private:

	double fM0;
	double fM1;
	double fM2;
	double fM3;

};

template<>
__hydra_host__ __hydra_device__ inline
double cos_decay_angle<1,2>(const double M12, const double M23, const double M31){


}
template<>
__hydra_host__ __hydra_device__ inline
double cos_decay_angle<2,3>(const double M12, const double M23, const double M31){


}
template<>
__hydra_host__ __hydra_device__ inline
double cos_decay_angle<3,1>(const double M12, const double M23, const double M31){


}


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



#endif /* COSTHETA_H_ */
