/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * ars_bigcrush.cpp
 *
 *  Created on: 18/09/2020
 *      Author: Antonio Augusto Alves Junior
 */


#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>


extern "C"
{
    #include "unif01.h"
    #include "bbattery.h"
    #include "util.h"
}

//set a global seed
static const uint64_t seed= 0x548c9decbce65297 ;

static  std::mt19937 RNG32(seed);
static  std::mt19937_64 RNG64(seed);


uint32_t mersenne32(void){

	return RNG32();
}

uint32_t mersenne64(void){

	return RNG64();
}



int main(int argv, char** argc)
{

   unif01_Gen* gen_a ;

   std::ostringstream filename;
   filename << "hydra_timing_baseline_TestU01_log.txt" ;

   std::cout << "------------------- [ Measuring timing for 1G events using std::mt19937 ] -------------------"  << std::endl;

   freopen(filename.str().c_str(), "w", stdout);

   gen_a = unif01_CreateExternGenBits(const_cast<char*>("mersenne32"), mersenne32 );

   unif01_TimerGenWr(gen_a, 1000000000, 0);
   unif01_DeleteExternGenBits(gen_a);

   gen_a = unif01_CreateExternGenBits(const_cast<char*>("mersenne64"), mersenne64 );
   unif01_TimerGenWr(gen_a, 1000000000, 0);
   unif01_DeleteExternGenBits(gen_a);

   fclose(stdout);


   return 0;

}
