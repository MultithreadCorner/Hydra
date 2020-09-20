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
 * bigcrush.cpp
 *
 *  Created on: 18/09/2020
 *      Author: Antonio Augusto Alves Junior
 */


#include <stdio.h>
#include <hydra/Random.h>

extern "C"
{
    #include "unif01.h"
    #include "bbattery.h"
     #include "sstring.h"
    #include "util.h"
}

//set a global seed
static const uint64_t seed= 0x123abdf3 ;

static hydra::ars RNG(seed);

uint32_t ars(void){

	return RNG();
}


int main (void) {


   unif01_Gen* gen_a = unif01_CreateExternGenBits(const_cast<char*>("hydra::ars"), ars );
   bbattery_BigCrush(gen_a);

   unif01_DeleteExternGenBits(gen_a);


   return 0;

}
