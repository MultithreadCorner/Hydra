/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * particle_dispersion.C
 *
 *  Created on: 04/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PARTICLE_DISPERSION_C_
#define PARTICLE_DISPERSION_C_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>
//this lib
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/Copy.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/multiarray.h>
#include <hydra/ForEach.h>


struct Evolve
{
	Evolve() = delete;

	Evolve(double time, double step):
		fTime(time),
		fStep(step)
	{}

	Evolve(Evolve const& other):
		fTime(other.fTime),
		fStep(other.fStep)
	{}

	__hydra_dual__
	template<typename Particle>
	inline void operator()(Particle p) {
		double theta =  hydra::get<3>(p);
		hydra::get<0>(p) += delta_x(theta);
		hydra::get<1>(p) += delta_y(theta);
		hydra::get<2>(p) += delta_z(theta);
	}

	__hydra_dual__
	inline double delta_x(double theta){
		return ::exp(-fTime)*::cos(theta)*fStep;
	}

	__hydra_dual__
	inline double delta_y(double theta){
		return ::exp(-fTime)*::sin(theta)*fStep;
	}

	__hydra_dual__
	inline double delta_z(double theta)	{
		return ::exp(-fTime)*fStep;
	}

	double fTime;
	double fStep;
};


typedef hydra::multiarray<double, 4, hydra::device::sys_t> StateDevice_t;
typedef hydra::multiarray<double, 4, hydra::host::sys_t>   StateHost_t;
typedef std::vector<StateHost_t> Ensamble_t;

void particle_dispersion( size_t nparicles, size_t nsteps, double epsilon ){

	Ensamble_t ensamble(nsteps, StateHost_t(nparicles, hydra::make_tuple(0.0, 0.0, 0.0, 0.0) ));


	{
		//initial state
		StateDevice_t initial_state(nparicles, hydra::make_tuple(0.0, 0.0, 0.0, 0.0));
		//distribute particles in the surface
		hydra::Random<> RND( std::chrono::system_clock::now().time_since_epoch().count());
		// set x-coordinate
		RND.SetSeed(159);
		RND.Uniform(-1.0, 1.0, initial_state.begin(0), initial_state.end(0));
		// set y-coordinate
		RND.SetSeed(753);
		RND.Uniform(-1.0, 1.0, initial_state.begin(1), initial_state.end(1));

		size_t i=1;

		for(auto& final_state:ensamble){

			hydra::Random<> RND(std::rand());
			RND.Uniform(0.0, 2.0*PI, initial_state.begin(3), initial_state.end(3));
			hydra::for_each( initial_state.begin(), initial_state.end(), Evolve(i*epsilon, 2.0)  );
			hydra::copy(initial_state.begin(), initial_state.end(), final_state.begin());
			i++;
		}

	}

	for(auto& state:ensamble){
		std::cout<< "=====================================" << std::endl;
		std::cout<< "=====================================" << std::endl;
		std::cout<< std::endl	<< std::endl	<< std::endl;
		for(auto particle:state)
			std::cout<< particle << std::endl;
	}

}



#endif /* PARTICLE_DISPERSION_C_ */
