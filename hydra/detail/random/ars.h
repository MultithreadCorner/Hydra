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
 * ars.h
 *
 *  Created on: Jul 10, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ARS_H_
#define ARS_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/external/hydra_R123/ars.h>

namespace hydra {

namespace random {

#if R123_USE_AES_NI

class ars
{

	typedef hydra_r123::ARS4x32 engine_type;

	typedef bool                  trigger_type;

	typedef union result_union
	{
		typename engine_type::ctr_type state32;
		uint64_t state64[2];

	} result_type;

public:



	typedef typename engine_type::ctr_type  state_type;
	typedef typename engine_type::ukey_type  seed_type;

	ars()=delete;

	__hydra_host__ __hydra_device__
	ars(uint32_t  s):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(state_type{}),
      fSeed(seed_type{s}),
      fTrigger(true)
    {}


	__hydra_host__ __hydra_device__
	ars( ars const& other):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(other.GetState() ),
      fSeed(other.GetSeed() ),
      fTrigger(true)
    {}

	__hydra_host__ __hydra_device__
	inline ars& operator=( ars const& other)
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
		result_type temp;

		if(fTrigger)
		{
			fCache = fEngine(fState,  fSeed);
			temp.state32 = fCache;
			result = temp.state64[0];
			fTrigger=false;
		}
		else
		{
			temp.state32 = fCache;
			result = temp.state64[ 1 ];
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

#else

#error ">>> [Hydra]: NVCC has no AES-IN instructions. hydra::ars. hydra::ars does not support CUDA backend. "

#endif

#endif /* ARS_H_ */
