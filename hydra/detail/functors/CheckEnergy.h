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
 * CheckEnergy.h
 *
 *  Created on: 24/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CHECKENERGY_H_
#define CHECKENERGY_H_

namespace hydra {

namespace detail {


template <size_t N>
struct CheckEnergy
{
	GReal_t fMasses[N];

	//constructor
	CheckEnergy(const GReal_t (&masses)[N] )
	{
		for(size_t i=0; i<N; i++)
			fMasses[i] = masses[i];
	}

	//copy
	__hydra_host__      __hydra_device__
	CheckEnergy(CheckEnergy<N> const& other)
	{
		for(size_t i=0; i<N; i++)
			fMasses[i] = other.fMasses[i];
	}

	template<typename Type>
	__hydra_host__ __hydra_device__
	inline bool operator()(Type& particle)
	{

		Vector4R mother = particle;
		GReal_t fTeCmTm =  mother.mass();

//#pragma unroll N
		for (size_t n = 0; n < N; n++)
		{
			fTeCmTm -= fMasses[n];
		}

		return (bool) fTeCmTm > 0.0;
	}
};

}//namespace detail

}//namespace hydra

#endif /* CHECKENERGY_H_ */
