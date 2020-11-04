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
 * benchmark_cbprng.cpp
 *
 *  Created on: 04/11/2020
 *      Author: Antonio Augusto Alves Junior
 */

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch/catch.hpp>

#include <cstdio>
#include <cstdint>

//hydra
#include <hydra/Random.h>

//set a global seed
static const uint64_t default_seed= 0x548c9decbce65295  ;

TEST_CASE("Performance of Hydra's counter based PRNGs") {

	hydra::squares3        RNG_squares3(default_seed);
	hydra::squares4        RNG_squares4(default_seed);
	hydra::squares3_128bit RNG_squares3_128bit(default_seed);
	hydra::threefry        RNG_threefry(default_seed);
	hydra::threefry_long   RNG_threefry_long(default_seed);
	hydra::philox          RNG_philox(default_seed);
	hydra::philox_long     RNG_philox_long(default_seed);

	BENCHMARK("hydra::squares3") {
		return RNG_squares3();
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4();
	};

	BENCHMARK("hydra::squares3_128bit") {
		return RNG_squares3_128bit();
	};

	BENCHMARK("hydra::threefry") {
		return RNG_threefry();
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long();
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox();
	};

	BENCHMARK("RNG_philox_long") {
		return RNG_philox_long();
	};
}
