/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
 * batch_test_generator.cpp
 *
 *  Created on: Oct 2, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <future>
//command line
#include <tclap/CmdLine.h>

#define MAX_NSEEDS 48
#define NSEEDS 20

static uint64_t seeds[ MAX_NSEEDS ]={
		0xd88f1030046b761c,
		0x6d98e2c649d074a5,
		0x79a119c8698cd4af,
		0xa10c8dfdd11a8d12,
		0x57bda9f700bcf826,
		0x1bb42f88dd2a42ce,
		0x5757acf513608f03,
		0xeb101ec23f252731,
		0xb4535e3196844e3c,
		0xc5002a4c736e3a83,
		0xe0c3f482de50a42a,
		0x9dea95a4da83e73f,
		0x7a183d75888c2731,
		0x3917017e2a09f115,
		0xc692cb05d9ba45fb,
		0x5d9241d7e8169edf,
		0x5143c8bfb7560fde,
		0xe628b62205b36e52,
		0x25317e7d70218baa,
		0xf67f30178a4bf8b0,
		0x82f76584e6410d90,
		0xea13d58fafb366fe,
		0x42f84cbced196d6d,
		0xcded21d45ef1b5d2,
		0x9fe3f9e4ddb50291,
		0x3e0239797ee1a99a,
		0xf42be5e90b20193f,
		0x15867eaa42b268e5,
		0x3c913b79a0f54e22,
		0x6fb7c25f399913db,
		0x1e386fd2aa97eee6,
		0xe66ab3c360c2a456,
		0x03bf4ecad202688e,
		0xed253d377ab806d1,
		0x0112d284c68dd85a,
		0x7b5790b81b5db848,
		0xd343879ab6a37548,
		0x5c75dc21661e23f6,
		0x0bab3104a9e77b13,
		0x12fb46cd8f4b3c16,
		0xcef081a18cd119e7,
		0x170b0f0b74ad3620,
		0x8ed1a7f40a9680eb,
		0x4514ab2bf422a2c4,
		0xc7a7ab2b783a27ac,
		0xaf934397b8ae6e48,
		0x1971badbd8ce49ce,
		0xa1088e6812a31628
};

int main(int argv, char** argc)
{
	std::string battery;
	std::string generator;

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<std::string> GeneratorArg("g", "generator",
				"Generator to be tested."
				" Possible values: [squares3], squares4, ars, philox, threefry, philox_long, threefry_long",false, "squares3", "string");
		cmd.add(GeneratorArg);

		TCLAP::ValueArg<std::string> BatteryArg("b", "battery", "TestU01 battery. Possible values are: smallcrush, crush, bigcrush",false,"smallcrush", "string") ;
		cmd.add(BatteryArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		battery   = BatteryArg.getValue();
		generator = GeneratorArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

	std::hash<std::string> hasher{};

	char* executor_name=const_cast<char*>("");

    bool launch_hilo_bits = true;

	if( hasher(std::string("squares3") )==hasher(generator)){
		launch_hilo_bits = false;
		executor_name= const_cast<char*>("hydra_squares3_bigcrush");
	}

	else if(hasher(std::string("squares4") )==hasher(generator)){
		launch_hilo_bits = false;
		executor_name= const_cast<char*>("hydra_squares4_bigcrush");
	}

	else if(hasher(std::string("ars") )==hasher(generator)){
		launch_hilo_bits = false;
		executor_name= const_cast<char*>("hydra_ars_bigcrush");
	}

	else if(hasher(std::string("philox") )==hasher(generator))
		executor_name= const_cast<char*>("hydra_philox_bigcrush");

	else if(hasher(std::string("threefry") )==hasher(generator))
		executor_name= const_cast<char*>("hydra_threefry_bigcrush");

	else if(hasher(std::string("philox_long") )==hasher(generator))
		executor_name= const_cast<char*>("hydra_philox_long_bigcrush");

	else if(hasher(std::string("threefry_long") )==hasher(generator))
		executor_name= const_cast<char*>("hydra_threefry_long_bigcrush");

	else{
		launch_hilo_bits = false;
		executor_name= const_cast<char*>("hydra_squares3_bigcrush");
	}

	char* battery_code =const_cast<char*>("");

	if( hasher(std::string("smallcrush") )==hasher(battery))
		battery_code = const_cast<char*>("0");

	else if( hasher(std::string("crush") )==hasher(battery))
		battery_code= const_cast<char*>("1");

	else if(hasher(std::string("bigcrush") )==hasher(battery))
		battery_code= const_cast<char*>("2");
	else
	  battery_code= const_cast<char*>("0");



	std::vector<std::string> command_list;

	for(size_t i=0; i<NSEEDS; ++i){
		if(launch_hilo_bits){
			std::ostringstream command_hi;
			command_hi << "./"<< executor_name << " "<< " -i="<< i <<" -b=" << battery_code << " -s=0x" << std::hex << seeds[i] << " -H";

			std::ostringstream command_lo;
			command_lo << "./"<< executor_name << " "<< " -i="<< i <<" -b=" << battery_code << " -s=0x" << std::hex << seeds[i] ;

			command_list.push_back(command_hi.str());
			command_list.push_back(command_lo.str());
		}
		else {
			std::ostringstream command;
			command << "./"<< executor_name << " "<< " -i="<< i <<" -b=" << battery_code << " -s=0x" << std::hex << seeds[i] ;
			command_list.push_back(command.str());
		}
	}

    std::vector<std::future<int>> tasks;

    //submit tasks
	for(int i=0; i<command_list.size(); ++i){

		std::cout << command_list[i].c_str()  << std::endl;

		tasks.push_back(std::async( std::launch::async,
						[i, &command_list](void) {
		  std::system(command_list[i].c_str());
		  return i;
		} ) );

	}

	size_t ntasks = 0;
	//monitor tasks
	while(ntasks < command_list.size()){

		for(auto& task: tasks){

			if( (task.valid()==true) && ( task.wait_for(std::chrono::seconds(0)) == std::future_status::ready)){

				int i = task.get();
				std::cout << ">> Task #" << i << " finished. Active tasks: " << (command_list.size()-ntasks) << std::endl;
				++ntasks;
			}

		}
	}



   return 0;

}
