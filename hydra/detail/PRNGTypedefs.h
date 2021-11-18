/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * PRNGTypedefs.h
 *
 *  Created on: 29/07/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PRNGTYPEDEFS_H_
#define PRNGTYPEDEFS_H_

#include <hydra/detail/Config.h>

#include <hydra/detail/external/hydra_thrust/random.h>
/*
#include <hydra/detail/random/philox.h>
#include <hydra/detail/random/threefry.h>
#include <hydra/detail/random/ars.h>
*/
#include <hydra/detail/random/EngineR123.h>
#include <hydra/detail/random/squares3.h>
//#include <hydra/detail/random/squares3_long.h>
#include <hydra/detail/random/squares4.h>
//#include <hydra/detail/random/squares4_long.h>

namespace hydra {

/*! \typedef default_random_engine
 *  \brief An implementation-defined "default" random number engine.
 *  \note \p default_random_engine is currently an alias for \p hydra::random::squares3, and may change
 *        in a future version.
 */

//typedef  hydra::random::squares3 default_random_engine;

//typedef hydra_thrust::random::default_random_engine default_random_engine;
//typedef hydra::random::philox default_random_engine;
typedef hydra::random::philox_long  default_random_engine;
//typedef hydra::random::threefry default_random_engine;
//typedef hydra::random::ars default_random_engine;
//typedef hydra::random::squares3 default_random_engine;
//typedef hydra::random::squares3_long  default_random_engine;
//typedef hydra::random::squares4 default_random_engine;

/*! \typedef minstd_rand0
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand0
 *        shall produce the value \c 1043618065 .
 */
typedef hydra_thrust::random::minstd_rand0 minstd_rand0;

/*! \typedef minstd_rand
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand
 *        shall produce the value \c 399268537 .
 */
typedef hydra_thrust::random::minstd_rand minstd_rand;


/*! \typedef ranlux24
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-3 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux24
 *        shall produce the value \c 9901578 .
 */
typedef hydra_thrust::random::ranlux24	ranlux24;

/*! \typedef ranlux48
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-4 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux48
 *        shall produce the value \c 88229545517833 .
 */
typedef hydra_thrust::random::ranlux48	ranlux48;

/*! \typedef taus88
 *  \brief A random number engine with predefined parameters which implements
 *         L'Ecuyer's 1996 three-component Tausworthe random number generator.
 *
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p taus88
 *        shall produce the value \c 3535848941 .
 */
typedef hydra_thrust::random::taus88 	taus88;


/*! \typedef philox
 *  \brief The Philox family of counter-based RNGs use integer multiplication, xor and permutation of W-bit words
 *         to scramble its N-word input key.  Philox is a mnemonic for Product HI LO Xor).
 *         This generator has a period of 2^128
 */
typedef hydra::random::philox philox;

/*! \typedef philox
 *  \brief The Philox family of counter-based RNGs use integer multiplication, xor and permutation of W-bit words
 *         to scramble its N-word input key.  Philox is a mnemonic for Product HI LO Xor).
 *         This generator has a period of 2^256
 */
typedef hydra::random::philox_long philox_long;

/*! \typedef threefry
 *  \brief Threefry uses integer addition, bitwise rotation, xor and permutation of words to randomize its output.
 *   This generator has a period of 2^128
 */
typedef hydra::random::threefry threefry;

/*! \typedef threefry_long
 *  \brief Threefry uses integer addition, bitwise rotation, xor and permutation of words to randomize its output.
 *  This generator has a period of 2^256
 */
typedef hydra::random::threefry_long threefry_long;

/*! \typedef ars
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::ars ars;

/*! \typedef squares3
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::squares3 squares3;

/*! \typedef squares3_long
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */

//typedef hydra::random::squares3_long squares3_long;

/*! \typedef squares4_long
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
//typedef hydra::random::squares4_long squares4_long;

/*! \typedef squares4
 *  \brief Ars uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space..
 *
 */
typedef hydra::random::squares4 squares4;


}  // namespace hydra


#endif /* PRNGTYPEDEFS_H_ */
