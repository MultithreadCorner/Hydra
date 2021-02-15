/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * DalitzBase.h
 *
 *  Created on: 04/01/2021
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZBASE_H_
#define DALITZBASE_H_


#include <hydra/detail/external/hydra_thrust/random.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Tuple.h>
#include <iostream>

namespace hydra {

namespace detail {

namespace dalitz {

struct _unity_weight
{
	template<typename... Args>
	__hydra_host__ __hydra_device__
	inline double operator()(Args const&...) const
	{
		return 1.0 ;
	}
};

}  // namespace dalitz

template<typename GRND>
struct DalitzBase
{


	typedef typename tuple_type<3, double>::type events_tuple_type;

	DalitzBase(double motherMass, std::array<double, 3> const& daughtersMasses, size_t seed=0xabc123):
			fSeed(seed)
		{

			fSumMSq      = motherMass*motherMass;

			fMasses[0]   = motherMass;
			fMassesSq[0] = motherMass*motherMass;

			for(size_t i=1; i<4; i++){

				fMasses[i]   = daughtersMasses[i-1];
				fMassesSq[i] = daughtersMasses[i-1]*daughtersMasses[i-1];
				fSumMSq += fMassesSq[i];

			}

		}

	DalitzBase(double motherMass, const double (&daughtersMasses)[3], size_t seed=0xabc123):
		fSeed(seed)
	{

		fSumMSq      = motherMass*motherMass;

		fMasses[0]   = motherMass;
		fMassesSq[0] = motherMass*motherMass;

		for(size_t i=1; i<4; i++){

			fMasses[i]   = daughtersMasses[i-1];
			fMassesSq[i] = daughtersMasses[i-1]*daughtersMasses[i-1];
			fSumMSq += fMassesSq[i];

		}

	}

	__hydra_host__ __hydra_device__
	DalitzBase( DalitzBase<GRND>const& other):
		fSeed(other.fSeed)
	{
		fSumMSq      = other.fSumMSq;

		for(size_t i=0; i<4; i++){
			fMasses[i]   = other.fMasses[i];
			fMassesSq[i] = other.fMassesSq[i];
		}
	}

	__hydra_host__ __hydra_device__
	inline   DalitzBase<GRND>& operator=( DalitzBase<GRND>const& other){

		if( this == other) return *this;

		fSeed = other.fSeed;

		fSumMSq      = other.fSumMSq;

		const double* massesSq = other.fMassesSq;
		const double* masses   = other.fMasses;

		for(size_t i=0; i<4; i++){
				fMasses[i]   = other.fMasses[i];
				fMassesSq[i] = other.fMassesSq[i];
			}

		return *this;
	}

	__hydra_host__ __hydra_device__
	inline   events_tuple_type MassesSq(size_t i) const {

		GRND rng(fSeed);

		rng.discard(2*i);

		double fM12Min = fMasses[1] + fMasses[2];
		double fM12Max = fMasses[0] - fMasses[3];

		hydra_thrust::uniform_real_distribution<double> M12_RNG( fM12Min,  fM12Max );

		double m12 = M12_RNG(rng);

		double m12_sq = m12*m12;

		hydra_thrust::uniform_real_distribution<double> M23_RNG( M23SqMin(m12),  M23SqMax(m12)) ;

		double m23_sq = M23_RNG(rng );

		double m13_sq = fSumMSq - m23_sq - m12_sq;

        return events_tuple_type{ m12_sq, m13_sq, m23_sq };

	}

	__hydra_host__ __hydra_device__
	inline  double Weight(double m12) const {

		double weight = 1.0;

		weight *= ::sqrt( (m12*m12 - ::pow( fMasses[1]+fMasses[2], 2.0))*
				          (m12*m12 - ::pow( fMasses[1]-fMasses[2], 2.0)) )/(2.0*m12);

		weight *= ::sqrt( ( fMasses[0]*fMasses[0] - ::pow( m12 + fMasses[3], 2.0))*
						  ( fMasses[0]*fMasses[0] - ::pow( m12 - fMasses[3], 2.0)))/(2.0*fMasses[0]);

		return weight;
	}

private:

	__hydra_host__ __hydra_device__
	inline   double E2( double m12 ) const  {

		return (m12*m12 - fMassesSq[1] +  fMassesSq[2] )/(2.0*m12);
	}

	__hydra_host__ __hydra_device__
	inline  double E3( double m12 ) const  {


		return (fMassesSq[0] - m12*m12 - fMassesSq[3] )/(2.0*m12) ;
	}

	__hydra_host__ __hydra_device__
	inline   double M23SqMax( double m12  ) const  {

		double E[2]{E2(m12), E3(m12)};

		double ESq[2]{ E[0]*E[0],  E[1]*E[1]};

		return  ::pow( E[0] + E[1], 2.0 ) -
				::pow( ::sqrt(ESq[0] - fMassesSq[2] ) - ::sqrt(ESq[1] - fMassesSq[3]), 2.0 );
	}

	__hydra_host__ __hydra_device__
	inline   double M23SqMin( double m12  ) const  {

		double E[2]{E2(m12), E3(m12)};

		double ESq[2]{ E[0]*E[0],  E[1]*E[1]};

		return  ::pow( E[0] + E[1], 2.0 ) -
				::pow( ::sqrt(ESq[0] - fMassesSq[2] ) + ::sqrt(ESq[1] - fMassesSq[3]), 2.0 );
	}



	size_t fSeed;
	double fSumMSq;
	double fMasses[4];
	double fMassesSq[4];
};

}  // namespace detail

}  // namespace hydra




#endif /* DALITZBASE_H_ */
