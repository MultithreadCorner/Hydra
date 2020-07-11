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
 * SeedRNG.h
 *
 *  Created on: 30/06/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SEEDRNG_H_
#define SEEDRNG_H_


#include <hydra/detail/Config.h>
#include <stdint.h>


namespace hydra {

class SeedRNG
{
public:

	 __hydra_host__ __hydra_device__
	 SeedRNG(size_t state=1337 ):
	fState(state)
	{}

	 __hydra_host__ __hydra_device__
	 SeedRNG(SeedRNG const& other):
	fState(other.GetState())
	{}

	 __hydra_host__ __hydra_device__
	 inline SeedRNG& operator=(SeedRNG const& other)
	{
		if(this==&other) return *this;

		fState=other.GetState();

		return *this;
	}

	 __hydra_host__ __hydra_device__
	size_t GetState() const {
		return fState;
	}

	__hydra_host__ __hydra_device__
	void SetState(size_t state) {
		fState = state;
	}

	__hydra_host__ __hydra_device__
	size_t operator()()
	{
		size_t z = (fState += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}


private:

	size_t fState;

};

}  // namespace hydra

#endif /* SEEDRNG_H_ */
