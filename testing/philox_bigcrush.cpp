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
}



//set a global seed
static const uint64_t seed= 0x123abdf3 ;


static hydra::philox RNG(seed);

uint32_t philox_hi(void){

	return uint32_t(RNG()>>32);
}

uint32_t philox_lo(void){

	return uint32_t(RNG());
}

int main(int argv, char** argc)
{
	unsigned battery = 0;
	bool     test_high_bits=0;

	std::vector<unsigned> allowed{0,1,2};
	TCLAP::ValuesConstraint<unsigned> allowedVals( allowed );

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<unsigned> EArg("b", "battery","TestU01's battery: 0 - SmallCrush (default) / 1 - Crush / 2 - BigCrush", false, 0, &allowedVals);
		cmd.add(EArg);

		TCLAP::SwitchArg HighBitArg("H", "high_bits", "Test the 32 higher bits of output", false) ;
		cmd.add(HighBitArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		battery = EArg.getValue();
		test_high_bits= HighBitArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

   unif01_Gen* gen_a;

   char* filename;

   std::string bit_range("");

   if( test_high_bits ) {

	   bit_range =  "_HighBits";
	   gen_a = unif01_CreateExternGenBits(const_cast<char*>("philoxH"), philox_hi );
   }
   else {

	   bit_range =  "_LowBits";
	   gen_a = unif01_CreateExternGenBits(const_cast<char*>("philoxL"), philox_lo );
   }

   std::string battery_name("");

   switch( battery ) {

   case 0:
	   battery_name="_SmallCrush";
	   break;
   case 1:
	   battery_name="_Crush";
	   break;
   case 2:
	   battery_name="_BigCrush";
	   break;

   }

   std::cout << "------------ [ Testing hydra::philox ] --------------"  << std::endl;

   if(battery==0){

	   filename =  const_cast<char*>("hydra_philox_TestU01_SmallCrush_log.txt");

	   std::cout <<
		   "Running TestU01's SmallCrush on hydra::philox.\n"
		   "Find the test report on 'hydra_philox_TestU01_SmallCrush_log.txt'\n"
		   "It is going to take from seconds to minutes.\n"
		   "Check the result issuing the command: tail -n 40 hydra_philox_TestU01_SmallCrush_log.txt"
		   << std::endl;
   }

   if(battery==1){

	   filename =  const_cast<char*>("hydra_philox_TestU01_Crush_log.txt");

	   std::cout<<
		   "Running TestU01's Crush on hydra::philox.\n"
		   "Find the test report on 'hydra_philox_TestU01_Crush_log.txt'\n"
		   "It is going to take from dozens of minutes to hours.\n"
		   "Check the result issuing the command: tail -n 40 hydra_philox_TestU01_Crush_log.txt"
		    << std::endl;
   }
   if(battery==2){

	   filename =  const_cast<char*>("hydra_philox_TestU01_BigCrush_log.txt");

	   std::cout<<
		   "Running TestU01's BigCrush on hydra::philox.\n"
		   "Find the test report on 'hydra_philox_TestU01_BigCrush_log.txt'\n"
		   "It is going to take many hours.\n"
		   "Check the result issuing the command: tail -n 40 hydra_philox_TestU01_BigCrush_log.txt"
		   << std::endl;
   }

   freopen(filename, "w", stdout);

   if(battery==0) bbattery_SmallCrush(gen_a);
   if(battery==1) bbattery_Crush(gen_a);
   if(battery==2) bbattery_BigCrush(gen_a);

   unif01_TimerGenWr(gen_a, 1000000000, 0);

   fclose(stdout);

   unif01_DeleteExternGenBits(gen_a);

   return 0;

}
