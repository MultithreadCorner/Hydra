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
#include <random>

//set a global seed
static const uint64_t default_seed= 0x548c9decbce65295  ;

TEST_CASE("Performance of Hydra's counter based PRNGs") {


	hydra::squares3        RNG_squares3(default_seed);
	hydra::squares4        RNG_squares4(default_seed);
	//hydra::squares3_long   RNG_squares3_long(default_seed);
	//hydra::squares4_long   RNG_squares4_long(default_seed);
	hydra::threefry        RNG_threefry(default_seed);
	hydra::threefry_long   RNG_threefry_long(default_seed);
	hydra::philox          RNG_philox(default_seed);
	hydra::philox_long     RNG_philox_long(default_seed);
	  std::mt19937         RNG_mt19937(default_seed );
	  std::ranlux48        RNG_ranlux48(default_seed);
/*
	SECTION( "hydra::squares3_long: test sequence" )
	{
		hydra::squares3_long   RNG_squares3_long_1(default_seed);
		RNG_squares3_long_1.discard(0);
		hydra::squares3_long   RNG_squares3_long_2(default_seed);
		RNG_squares3_long_2.discard( RNG_squares3_long_2.max);
		std::cout<<"hydra::squares3_long output: counter | RNG(counter) | RNG (counter+2^64-1 )" << std::endl << std::endl ;
		for(size_t n=0; n< 10; ++n){
			auto x= RNG_squares3_long_1();
			auto y= RNG_squares3_long_2();
			std::cout<< n <<" | "<< std::hex << x <<" | " << y<< std::dec << std::endl ;
	  		REQUIRE( x != y );
		}
	 }
*/
/*
	SECTION( "hydra::squares4_long: test sequence" )
	{
		hydra::squares4_long   RNG_squares4_long_1(default_seed);
		RNG_squares4_long_1.discard(0);
		hydra::squares4_long   RNG_squares4_long_2(default_seed);
		RNG_squares4_long_2.discard( RNG_squares4_long_2.max);
		std::cout<<"hydra::squares4_long output: counter | RNG(counter) | RNG (counter+2^64-1 )" << std::endl << std::endl ;
		for(size_t n=0; n< 10; ++n){
			auto x= RNG_squares4_long_1();
			auto y= RNG_squares4_long_2();
			std::cout<< n <<" | "<< std::hex << x <<" | " << y<< std::dec << std::endl ;
	  		REQUIRE( x != y );
		}
	 }
*/
	SECTION( "Benchmarks: RNG::operator()" )
	{
	BENCHMARK("hydra::squares3") {
		return RNG_squares3();
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4();
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long();
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long();
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry();
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long();
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox();
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long();
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937();
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48();
	};
	}


	SECTION( "Benchmarks: RNG::discard(1)" )
	{
	static uint32_t jump_size=1;

	BENCHMARK("hydra::squares3") {
		RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
	/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}

	SECTION( "Benchmarks: RNG::discard(10)" )
	{
	static uint32_t jump_size=10;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}


	SECTION( "Benchmarks: RNG::discard(100)" )
	{
	static uint32_t jump_size=100;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}

	SECTION( "Benchmarks: RNG::discard(1000)" )
	{
	static uint32_t jump_size=1000;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}


	SECTION( "Benchmarks: RNG::discard(10000)" )
	{
	static uint32_t jump_size=10000;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}


	SECTION( "Benchmarks: RNG::discard(100000)" )
	{
	static uint32_t jump_size=100000;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}


	SECTION( "Benchmarks: RNG::discard(1000000)" )
	{
	static uint32_t jump_size=1000000;

	BENCHMARK("hydra::squares3") {
		return RNG_squares3.discard(jump_size);
	};

	BENCHMARK("hydra::squares4") {
		return RNG_squares4.discard(jump_size);
	};
/*
	BENCHMARK("hydra::squares3_long") {
		return RNG_squares3_long.discard(jump_size);
	};

	BENCHMARK("hydra::squares4_long") {
		return RNG_squares4_long.discard(jump_size);
	};
*/
	BENCHMARK("hydra::threefry") {
		return RNG_threefry.discard(jump_size);
	};

	BENCHMARK("hydra::threefry_long") {
		return RNG_threefry_long.discard(jump_size);
	};

	BENCHMARK("hydra::philox") {
		return RNG_philox.discard(jump_size);
	};

	BENCHMARK("hydra::philox_long") {
		return RNG_philox_long.discard(jump_size);
	};

	BENCHMARK("std::mt19937") {
		return RNG_mt19937.discard(jump_size);
	};

	BENCHMARK("std::ranlux48") {
		return RNG_ranlux48.discard(jump_size);
	};
	}
}
