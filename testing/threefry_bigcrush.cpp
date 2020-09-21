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
 * philox_bigcrush.cpp
 *
 *  Created on: 18/09/2020
 *      Author: Antonio Augusto Alves Junior
 */


#include <stdio.h>
#include <cstdlib>
#include <vector>

//hydra
#include <hydra/Random.h>
//command line
#include <tclap/CmdLine.h>

extern "C"
{
    #include "unif01.h"
    #include "bbattery.h"
    #include "util.h"
    #include "sstring.h"
}

//set a global seed
static const uint64_t seed= 0x123abdf3 ;

static hydra::threefry RNG(seed);

uint32_t threefry(void){

	return RNG();
}


int main(int argv, char** argc)
{
	unsigned battery = 0;
	std::vector<unsigned> allowed{0,1,2};
	TCLAP::ValuesConstraint<unsigned> allowedVals( allowed );

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<unsigned> EArg("b", "battery","TestU01's battery: 0 - SmallCrush (default) / 1 - Crush / 2 - BigCrush", false, 0, &allowedVals);
		cmd.add(EArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		battery = EArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

   unif01_Gen* gen_a = unif01_CreateExternGenBits(const_cast<char*>("hydra::threefry"), threefry );

   if(battery==0) std::cout <<
		   "[Testing hydra::threefry] : "
		   "Running TestU01's SmallCrush on hydra::threefry.\n"
		   "Find the test report on 'hydra_threefry_TestU01_log.txt'\n"
		   "It is going to take from seconds to minutes."
		   << std::endl;
   if(battery==1) std::cout<<
		   "[Testing hydra::threefry] : "
		   "Running TestU01's Crush on hydra::threefry.\n"
		   "Find the test report on 'hydra_threefry_TestU01_log.txt'\n"
		   "It is going to take from dozens of minutes to hours."
		    << std::endl;
   if(battery==2) std::cout<<
		   "[Testing hydra::threefry] : "
		   "Running TestU01's BigCrush on hydra::threefry.\n"
		   "Find the test report on 'hydra_threefry_TestU01_log.txt'\n"
		   "It is going to take many hours."
		   << std::endl;


   freopen("hydra_threefry_TestU01_log.txt", "w", stdout);

   //
   if(battery==0)bbattery_SmallCrush(gen_a);
   /*
   {
	   RNG.discard(::pow(2,34));
	   sstring_AutoCor(gen_a, NULL, 10, 1000000029, 27,3,1);
   }*/

   if(battery==1) bbattery_Crush(gen_a);
   if(battery==2) bbattery_BigCrush(gen_a);

   fclose(stdout);

   unif01_DeleteExternGenBits(gen_a);

   return 0;

}
