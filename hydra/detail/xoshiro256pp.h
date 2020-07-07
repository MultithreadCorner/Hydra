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
 * xoshiro256pp.h
 *
 *  Created on: 05/07/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef XOSHIRO256PP_H_
#define XOSHIRO256PP_H_

#include <stdint.h>
#include <limits>


namespace hydra {

namespace random {

class xoshiro256pp
{
public:

	__hydra_host__ __hydra_device__
	xoshiro256pp(uint64_t seed=0xb56c4feeef1b ):
	fSplitMix64State(seed),
	fSeed(seed)
	{
		for(size_t i=0; i<4; ++i)
			fState[i]=splitmix64();
	}

	__hydra_host__ __hydra_device__
	inline uint64_t operator()(void)
	{
		const uint64_t result = rotl(fState[0] + fState[3], 23) + fState[0];

		const uint64_t t = fState[1] << 17;

		fState[2] ^= fState[0];
		fState[3] ^= fState[1];
		fState[1] ^= fState[2];
		fState[0] ^= fState[3];

		fState[2] ^= t;

		fState[3] = rotl(fState[3], 45);

		return result;

	}

	__hydra_host__ __hydra_device__
	void discard(unsigned long long z)
	{
	    for(; z > 0; --z)
	    {
          this->operator()();
	    }
	 }

	__hydra_host__ __hydra_device__
	void jump_ahead(void)
	{

		static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba,
				0xd5a61266f0c9392c, 0xa9582618e03fc9aa,
				0x39abdc4529b1661c };

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		uint64_t s2 = 0;
		uint64_t s3 = 0;

		for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
			for(int b = 0; b < 64; b++) {
				if (JUMP[i] & UINT64_C(1) << b) {
					s0 ^= fState[0];
					s1 ^= fState[1];
					s2 ^= fState[2];
					s3 ^= fState[3];
				}
				this->operator()();
			}

		fState[0] = s0;
		fState[1] = s1;
		fState[2] = s2;
		fState[3] = s3;
	}

	static const  uint64_t min = 0;

	static const  uint64_t max = std::numeric_limits<uint64_t>::max();

private:

	__hydra_host__ __hydra_device__
	inline uint64_t rotl(const uint64_t x, int k)
	{
		return (x << k) | (x >> (64 - k));
	}

	__hydra_host__ __hydra_device__
	inline uint64_t splitmix64()
	{
		uint64_t z = (fSplitMix64State += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}


	uint64_t fState[4];
	uint64_t fSplitMix64State;
	uint64_t fSeed;
};

}  // namespace random

}  // namespace hydra


#endif /* XOSHIRO256PP_H_ */
