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
 * EngineR123.h
 *
 *  Created on: 09/09/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ENGINER123_H_
#define ENGINER123_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/RandomTraits.h>
#include <hydra/detail/external/hydra_R123/array.h>
#include <hydra/detail/external/hydra_R123/philox.h>
#include <hydra/detail/external/hydra_R123/threefry.h>
#include <hydra/detail/external/hydra_R123/ReinterpretCtr.hpp>
#if R123_USE_AES_NI
#include <hydra/detail/external/hydra_R123/ars.h>
#endif

#include <stdint.h>

namespace hydra {

namespace random {

template<typename UIntType>
__hydra_host__ __hydra_device__
inline UIntType splitmix( UIntType& );


template<>
__hydra_host__ __hydra_device__
inline uint32_t splitmix<uint32_t>(uint32_t& x) {
	uint32_t z = (x += 0x6D2B79F5UL);
	z = (z ^ (z >> 15)) * (z | 1UL);
	z ^= z + (z ^ (z >> 7)) * (z | 61UL);
	return z ^ (z >> 14);
}

template<>
__hydra_host__ __hydra_device__
inline uint64_t  splitmix<uint64_t>(uint64_t& x){
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}


template<typename Engine>
class EngineR123
{
	typedef unsigned               trigger_type;

public:

	typedef Engine   engine_type;
	typedef typename hydra::detail::random_traits<engine_type>::state_type     state_type;
	typedef typename hydra::detail::random_traits<engine_type>::seed_type       seed_type;
	typedef typename hydra::detail::random_traits<engine_type>::advance_type advance_type;
	typedef typename hydra::detail::random_traits<engine_type>::init_type       init_type;
	typedef typename hydra::detail::random_traits<engine_type>::result_type   result_type;

	static const unsigned arity= hydra::detail::random_traits<engine_type>::arity;

	__hydra_host__ __hydra_device__
	EngineR123()=delete;

	//typedef typename	init_type::dummy tt;


	__hydra_host__ __hydra_device__
	EngineR123(result_type  seed):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(state_type{}),
      fSeed(seed_type{}),
      fTrigger(0)
    {
		init_type temp= {{}};

		for(unsigned i=0; i<temp.size()-1; ++i)
			temp[i] = splitmix<result_type>(seed);

		fSeed =temp;
    }

	__hydra_host__ __hydra_device__
	EngineR123(init_type  seed):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(state_type{}),
      fSeed(seed),
      fTrigger(0)
    {}


	__hydra_host__ __hydra_device__
	EngineR123( EngineR123<Engine> const& other):
	  fEngine(engine_type{}),
      fCache(state_type{}),
      fState(other.GetState() ),
      fSeed(other.GetSeed() ),
      fTrigger(other.GetTrigger())
    {}

	__hydra_host__ __hydra_device__
	inline EngineR123<Engine>&
	operator=( EngineR123<Engine> const& other)
	{
		if(this==&other) return *this;

	  fEngine = engine_type{};
      fCache  = state_type{};
      fState  = other.GetState();
      fSeed   = other.GetSeed();
      fTrigger =other.GetTrigger();
      return *this;
    }


	__hydra_host__ __hydra_device__
	inline result_type operator()(void)
	{
		result_type result = 0;

		if(fTrigger==arity) fTrigger=0;

		if(fTrigger==0){

			fCache = fEngine(fState.incr(),  fSeed);
			result = fCache[fTrigger++];
		}
		else {
			result = fCache[fTrigger++];
		}

		return result;
	}

	__hydra_host__ __hydra_device__
	inline void discard( advance_type n){

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

	static const result_type HYDRA_PREVENT_MACRO_SUBSTITUTION min  = 0;

	static const result_type HYDRA_PREVENT_MACRO_SUBSTITUTION max = std::numeric_limits<result_type>::max();

private:

	__hydra_host__ __hydra_device__
	inline trigger_type GetTrigger() const {
		return fTrigger;
	}


	engine_type   fEngine;
	state_type     fCache;
	state_type     fState;
	seed_type       fSeed;
	trigger_type fTrigger;
};

#if R123_USE_AES_NI
typedef EngineR123<hydra_r123::ARS4x32>           ars;
#else
typedef void  ars;
#endif

typedef EngineR123<hydra_r123::Threefry4x64> threefry;

typedef EngineR123<hydra_r123::Philox4x64>     philox;


}  // namespace random

}  // namespace hydra



#endif /* ENGINER123_H_ */
