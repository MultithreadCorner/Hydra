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
#include <hydra/Lambda.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/multiarray.h>
#include <hydra/ForEach.h>

#include "TCanvas.h"
#include "TGraphTime.h"
#include "TROOT.h"
#include "TMarker.h"

struct Evolve
{
	Evolve() = delete;

	Evolve(double ispeed, double time, double slope):
		fInitialSpeed(ispeed),
		fSlope(slope),
		fTime(time)
	{}

	__hydra_dual__
	Evolve(Evolve const& other):
		fInitialSpeed(other.fInitialSpeed),
		fSlope(other.fSlope),
		fTime(other.fTime)
	{}

	template<typename Particle>
	__hydra_dual__
	inline void operator()(Particle p) {

		double theta =  hydra::get<3>(p);
		double phi   =  hydra::get<4>(p);

		hydra::get<0>(p) += delta_x(theta, phi);
		hydra::get<1>(p) += delta_y(theta, phi);
		hydra::get<2>(p) += delta_z(theta, phi);
	}

	__hydra_dual__
	inline double delta_x(double theta, double phi){
		return displacement()*::cos(theta)*sin(phi);
	}

	__hydra_dual__
	inline double delta_y(double theta, double phi){
		return displacement()*::sin(theta)*sin(phi);
	}

	__hydra_dual__
	inline double delta_z(double theta, double phi)	{
		return displacement()*sin(phi);
	}


	__hydra_dual__
	inline double displacement(){
		return ::exp(-fSlope*fTime)*(fSlope*fTime*fTime + fInitialSpeed) ;
	}


	double fInitialSpeed;
	double fSlope;
	double fTime;

};


typedef hydra::multiarray<double, 5, hydra::device::sys_t> StateDevice_t;
typedef hydra::multiarray<double, 5, hydra::host::sys_t>   StateHost_t;
typedef std::vector<StateHost_t> Ensamble_t;

void particle_dispersion( size_t nparicles, size_t nsteps, double initial_speed, double slope ){

	Ensamble_t ensamble(nsteps, StateHost_t(nparicles, hydra::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0) ));


	{
		//initial state
		StateDevice_t initial_state(nparicles, hydra::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0));
		//distribute particles in the surface
		hydra::Random<> RND( std::chrono::system_clock::now().time_since_epoch().count());
		// set x-coordinate
		RND.SetSeed(159);
		RND.Uniform(-0.1, 0.1, initial_state.begin(0), initial_state.end(0));
		// set y-coordinate
		RND.SetSeed(753);
		RND.Uniform(-2.0, 2.0, initial_state.begin(1), initial_state.end(1));

		size_t i=1;

		for(auto& final_state:ensamble){

			RND.SetSeed(std::rand());
			RND.Uniform(0.0, 2.0*PI, initial_state.begin(3), initial_state.end(3));
			RND.SetSeed(std::rand());
			RND.Uniform(0.0, 2.0*PI, initial_state.begin(4), initial_state.end(4));
			hydra::for_each( initial_state.begin(), initial_state.end(), Evolve(initial_speed , double(i)/nsteps, slope)  );
			hydra::copy(initial_state.begin(), initial_state.end(), final_state.begin());
			i++;
		}

	}


	TCanvas canvas("canvas", "", 500, 500);
	TGraphTime *g = new TGraphTime(nsteps,-10.0, -10.0, 10.0, 10.0);

	int step=0;
	for(auto final_state:ensamble){

		for( auto particle:final_state){

			double x = hydra::get<0>(particle);
			double y = hydra::get<1>(particle);

			TMarker *m = new TMarker(x,y,20);
			m->SetMarkerColor(kBlack);
			m->SetMarkerSize(0.5);
			g->Add(m,step);
		}

		step++;
	}


	g->Draw();

}



#endif /* PARTICLE_DISPERSION_C_ */
