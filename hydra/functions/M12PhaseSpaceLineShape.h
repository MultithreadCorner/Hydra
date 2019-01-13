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
 * M12PhaseSpaceLineShape.h
 *
 *  Created on: 27/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef M12PhaseSpaceLineShape_H_
#define M12PhaseSpaceLineShape_H_

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
 * @class M12PhaseSpaceLineShape
 *
 * Two-body phase-space distribution for \f$ m_{12}\f$:
 * \f[ \frac{dN}{dm_{m_12}} \propto q.p \f]
 *
 */

template<unsigned int ArgIndex=0>
class M12PhaseSpaceLineShape: public BaseFunctor<M12PhaseSpaceLineShape<ArgIndex>, double, 0>{

public:
	M12PhaseSpaceLineShape()=delete;

		M12PhaseSpaceLineShape(double mother_mass, double daugther1_mass,
				double daugther2_mass, double daugther3_mass):
			BaseFunctor<M12PhaseSpaceLineShape<ArgIndex>, double, 0>{},
			fDaughter1Mass(daugther1_mass),
			fDaughter2Mass(daugther2_mass),
			fDaughter3Mass(daugther3_mass),
			fMotherMass(mother_mass)
			{}

		__hydra_host__  __hydra_device__
		M12PhaseSpaceLineShape(M12PhaseSpaceLineShape<ArgIndex>  const& other):
			BaseFunctor<M12PhaseSpaceLineShape<ArgIndex>, double, 0>(other),
			fDaughter1Mass(other.GetDaughter1Mass()),
			fDaughter2Mass(other.GetDaughter2Mass()),
			fDaughter3Mass(other.GetDaughter3Mass()),
			fMotherMass(other.GetMotherMass())
			{}

		__hydra_host__  __hydra_device__
		M12PhaseSpaceLineShape<ArgIndex>&
		operator=(M12PhaseSpaceLineShape<ArgIndex>  const& other)
		{
			if(this==&other) return  *this;

			BaseFunctor<M12PhaseSpaceLineShape<ArgIndex>,
				double, 0>::operator=(other);

			fDaughter1Mass= other.GetDaughter1Mass();
			fDaughter2Mass= other.GetDaughter2Mass();
			fDaughter3Mass= other.GetDaughter3Mass();
			fMotherMass= other.GetMotherMass();

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
		double GetDaughter3Mass() const {
			return fDaughter3Mass;
		}

		__hydra_host__  __hydra_device__ inline
		void SetDaughter3Mass(double daughter3Mass) {
			fDaughter3Mass = daughter3Mass;
		}

		__hydra_host__  __hydra_device__ inline
		double GetMotherMass() const {
			return fMotherMass;
		}

		__hydra_host__  __hydra_device__ inline
		void SetMotherMass(double motherMass) {
			fMotherMass = motherMass;
		}


		template<typename T>
		__hydra_host__ __hydra_device__ inline
		double Evaluate(unsigned int, T*x)  const	{

			const double m = x[ArgIndex] ;

			return  m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fDaughter3Mass)?LineShape(m):0.0;

		}

		template<typename T>
		__hydra_host__ __hydra_device__ inline
		double Evaluate(T x)  const {

			double m =  get<ArgIndex>(x);

			return   m > (fDaughter1Mass+fDaughter2Mass) && m<(fMotherMass-fDaughter3Mass)?LineShape(m):0.0;
		}

private:


		__hydra_host__ __hydra_device__   inline
		double LineShape(const double m ) const {

			double p = pmf( fMotherMass, m, fDaughter3Mass);
			double q = pmf( m, fDaughter1Mass, fDaughter2Mass);

			return (p*q);

		}

		double fDaughter1Mass;
		double fDaughter2Mass;
		double fDaughter3Mass;
		double fMotherMass;
};

}  // namespace hydra


#endif /* NONRESONANTLINESHAPE_H_ */
