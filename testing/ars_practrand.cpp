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
 * ars_practrand.cpp
 *
 *  Created on: 07/10/2020
 *      Author: Antonio Augusto Alves Junior
 */

#include <cstdio>
#include <cstdint>

//hydra
#include <hydra/Random.h>

//command line
#define TCLAP_SETBASE_ZERO 1
#include <tclap/CmdLine.h>

//set a global seed
static const uint64_t default_seed= 0x548c9decbce65295  ;

int main(int argv, char** argc)
{
	uint64_t seed = default_seed;

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<uint64_t> SeedArg("s", "seed","RNG seed.", false, default_seed, "uint64_t");
		cmd.add(SeedArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		seed    = SeedArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

	hydra::ars RNG(seed);

	while (1) {
		uint32_t value = RNG();
		fwrite((void*) &value, sizeof(value), 1, stdout);
	}
}


