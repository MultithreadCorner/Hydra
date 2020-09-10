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
 * RandomTraits.h
 *
 *  Created on: 09/09/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANDOMTRAITS_H_
#define RANDOMTRAITS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/hydra_R123/philox.h>
#include <hydra/detail/external/hydra_R123/threefry.h>
#include <hydra/detail/external/hydra_R123/ars.h>
#include <stdint.h>

namespace hydra {

namespace detail {

template<typename Engine>
struct random_traits;
/*
 * random_traits<T>::state_type { counter, state}
 * random_traits<T>::advance_type;
 * random_traits<T>::init_type;
 * random_traits<T>::result_type;
 */

//philox
template<> struct random_traits<hydra_r123::Philox2x64>
{
	typedef union {
		typename hydra_r123::Philox2x64::ctr_type  counter;
		typename hydra_r123::Philox2x64::ctr_type    state;
	} state_type;

	typedef typename hydra_r123::Philox2x64::key_type  seed_type;
	typedef uint64_t advance_type;
	typedef typename hydra_r123::Philox2x64::ukey_type    init_type;
	typedef uint64_t  result_type;

	enum{arity=2};
};

//
template<> struct random_traits<hydra_r123::Threefry2x64>
{
	typedef union {
		typename hydra_r123::Threefry2x64::ctr_type  counter;
		typename hydra_r123::Threefry2x64::ctr_type    state;
	} state_type;

	typedef typename hydra_r123::Threefry2x64::key_type  seed_type;
	typedef uint64_t advance_type;
	typedef typename hydra_r123::Threefry2x64::ukey_type    init_type;
	typedef uint64_t  result_type;

	enum{arity=2};
};


#if R123_USE_AES_NI
template<> struct random_traits<hydra_r123::ARS4x32>
{
	typedef union {
		typename hydra_r123::ARS4x32::ctr_type  counter;
		uint64_t                               state[2];
	} state_type;

	typedef typename hydra_r123::ARS4x32::key_type  seed_type;
	typedef uint64_t advance_type;
	typedef typename hydra_r123::ARS4x32::ukey_type    init_type;
	typedef uint64_t  result_type;

	enum{arity=2};
};
#endif


}  // namespace detail

}  // namespace hydra



#endif /* RANDOMTRAITS_H_ */
