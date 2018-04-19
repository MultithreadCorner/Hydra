/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * three_body_phsp.C
 *
 *  Created on: Apr 19, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef THREE_BODY_PHSP_C_
#define THREE_BODY_PHSP_C_

/*---------------------------------
 * std
 * ---------------------------------
 */
#include <iostream>
#include <assert.h>
#include <time.h>
#include <vector>
#include <tuple>
#include <chrono>
/*---------------------------------
 * Include hydra classes and
 * algorithms for
 *--------------------------------
 */
#ifndef HYDRA_HOST_SYSTEM
#define HYDRA_HOST_SYSTEM CPP
#endif

#ifndef HYDRA_DEVICE_SYSTEM
#define HYDRA_DEVICE_SYSTEM TBB
#endif

#include <hydra/PhaseSpace.h>
#include <hydra/Decays.h>
#include <hydra/Vector4R.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#include <TROOT.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <TGenPhaseSpace.h>


void three_body_phsp(size_t nentries=100000)
{

	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass

	// Mother particle
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{Jpsi_mass, K_mass, pi_mass};

	//Device
	//timing
	std::chrono::duration<double, std::milli> elapsed_d;
	{

		hydra::Decays<3, hydra::device::sys_t> Events_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_d.begin(), Events_d.end());

		auto end = std::chrono::high_resolution_clock::now();

		elapsed_d = end - start;

	}

	//Host
	//timing
	std::chrono::duration<double, std::milli> elapsed_h;
	{

		hydra::Decays<3, hydra::host::sys_t> Events_h(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_h.begin(), Events_h.end());

		auto end = std::chrono::high_resolution_clock::now();

		elapsed_h = end - start;

	}

	//TGenPhaseSpace
	//timing
	std::chrono::duration<double, std::milli> elapsed_r;
	{
		TLorentzVector P( 0.0, 0.0, 0.0, B0_mass);

		double masses[3] = {Jpsi_mass, K_mass, pi_mass} ;

		TGenPhaseSpace event;
		event.SetDecay(P, 3, masses);

		std::vector< std::tuple<double, TLorentzVector,TLorentzVector, TLorentzVector> > decays(nentries);

		auto start = std::chrono::high_resolution_clock::now();
		for (size_t n = 0;n<nentries; n++) {

			Double_t weight    = event.Generate();
			TLorentzVector *p0 = event.GetDecay(0);
			TLorentzVector *p1 = event.GetDecay(1);
			TLorentzVector *p2 = event.GetDecay(2);

			decays[n]= std::make_tuple(weight, *p0, *p1, *p2 );

		}
		auto end = std::chrono::high_resolution_clock::now();

		elapsed_r = end - start;
	}


	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------------------------------"   << std::endl;
	std::cout << "| B0 -> J/psi K pi"                         << std::endl;
	std::cout << "| Number of events :"<< nentries             << std::endl;
	std::cout << "| Time in Hydra/device (ms):"<< elapsed_d.count()<< std::endl;
	std::cout << "| Time in Hydra/host (ms) :" << elapsed_h.count()<< std::endl;
	std::cout << "| Time in ROOT/TGenPhaseSpace (ms) :"<< elapsed_r.count()<< std::endl;
	std::cout << "-----------------------------------------"  << std::endl;


}

#endif /* THREE_BODY_PHSP_C_ */
