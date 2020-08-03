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
 * philox.h
 *
 *  Created on: 10/07/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHILOX_H_
#define PHILOX_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/hydra_R123/philox.h>
#include <stdint.h>

namespace hydra {

namespace random {

class philox
{
	typedef hydra_r123::Philox2x64 engine_type;

	typedef bool                     trigger_type;

public:

	typedef typename engine_type::ctr_type  state_type;
	typedef typename engine_type::ukey_type  seed_type;
	typedef unsigned long long advance_type;
	typedef uint64_t  result_type;

	__hydra_host__ __hydra_device__
	philox()=delete;

	__hydra_host__ __hydra_device__
	philox(uint64_t  s):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(state_type{}),
      fSeed(seed_type{s}),
      fTrigger(true)
    {}


	__hydra_host__ __hydra_device__
	philox( philox const& other):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(other.GetState() ),
      fSeed(other.GetSeed() ),
      fTrigger(true)
    {}

	__hydra_host__ __hydra_device__
	inline philox& operator=( philox const& other)
	{
		if(this==&other) return *this;

	  fEngine = engine_type{};
      fCache  = state_type{};
      fState  = other.GetState();
      fSeed   = other.GetSeed();
      fTrigger = true;
      return *this;
    }


	__hydra_host__ __hydra_device__
	inline uint64_t operator()(void)
	{
		uint64_t result = 0;

		if(fTrigger)
		{
			fCache = fEngine(fState,  fSeed);
			result = fCache[0];
			fTrigger=false;
		}
		else
		{
			result = fCache[ 1 ];
			fTrigger=true;
		}

		fState.incr();

		return result;
	}

	__hydra_host__ __hydra_device__
	inline void discard( unsigned long long n){

		fState.incr(n);
	}

	__hydra_host__ __hydra_device__
	inline const seed_type& GetSeed() const {
		return fSeed;
	}

	__hydra_host__ __hydra_device__
	inline void SetSeed(seed_type seed) {
		fSeed = seed;
	}

	__hydra_host__ __hydra_device__
	inline const state_type& GetState() const {
		return fState;
	}

	__hydra_host__ __hydra_device__
	inline void SetState(const state_type& state) {
		fState = state;
	}

	static const  uint64_t HYDRA_PREVENT_MACRO_SUBSTITUTION min  = 0;

	static const  uint64_t HYDRA_PREVENT_MACRO_SUBSTITUTION max = std::numeric_limits<uint64_t>::max();

private:

	engine_type  fEngine;
	state_type fCache;
	state_type fState;
	seed_type     fSeed;
	trigger_type fTrigger;
};

}  // namespace random

}  // namespace hydra




#endif /* PHILOX_H_ */
