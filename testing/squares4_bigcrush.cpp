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
 * ars_bigcrush.cpp
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
}

//set a global seed
static const uint64_t seed= 0x548c9decbce65297 ;

static hydra::squares4 RNG(seed);

uint32_t squares(void){

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

   unif01_Gen* gen_a = unif01_CreateExternGenBits(const_cast<char*>("squares4"),squares );

   char* battery_name=const_cast<char*>("");

   switch( battery ) {

   case 0:
	   battery_name=const_cast<char*>( "SmallCrush");
	   break;

   case 1:
	   battery_name=const_cast<char*>("Crush");
	   break;

   case 2:
	   battery_name=const_cast<char*>("BigCrush");
	   break;

   }


   std::ostringstream filename;
   filename << "hydra_squares4_TestU01_" << battery_name << "_log.txt" ;

   std::ostringstream message;
   message << "Running TestU01's " << battery_name << " on hydra::squares4." << std::endl
		   << "Find the test's report on the file " << filename.str().c_str() << " in the program's work directory." << std::endl
		   << "It is going to take from seconds (SmallCrush) to hours (BigCrush)."<< std::endl
		   << "Check the result issuing the command: tail -n 25 " << filename.str().c_str() << std::endl;

   std::cout << "------------------- [ Testing hydra::squares4 ] -------------------"  << std::endl;

   std::cout << message.str().c_str()  << std::endl;

   freopen(filename.str().c_str(), "w", stdout);

   if(battery==0) bbattery_SmallCrush(gen_a);
   if(battery==1) bbattery_Crush(gen_a);
   if(battery==2) bbattery_BigCrush(gen_a);

   unif01_TimerGenWr(gen_a, 1000000000, 0);

   fclose(stdout);

   unif01_DeleteExternGenBits(gen_a);

   return 0;

}
