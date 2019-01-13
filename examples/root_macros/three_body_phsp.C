/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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

#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Decays.h>
#include <hydra/Vector4R.h>
#include <hydra/Tuple.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/DenseHistogram.h>

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

template<typename Backend>
void generate_decays(Backend const& policy, TH2D* histogram, size_t nentries);

void three_body_phsp(const char* backend ="device",  size_t nentries=100000 ){

	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass


	TH2D* Dalitz = new TH2D("Dalitz",
			"Device;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	if (strcmp(backend, "device") == 0)
		generate_decays( hydra::device::sys, Dalitz, nentries);
	else
		generate_decays( hydra::host::sys, Dalitz, nentries);


	Dalitz->Draw();
}


template<typename Backend>
void generate_decays(Backend const& policy, TH2D* histogram, size_t nentries)
{

	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass

	// Mother particle
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{Jpsi_mass, K_mass, pi_mass};

	// Histogram
	hydra::DenseHistogram<double, 2, hydra::device::sys_t> Dalitz({100,100},
			{pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
			{pow(B0_mass - K_mass,2)   , pow(B0_mass - Jpsi_mass,2)});

	// Dalitz
	auto dalitz_calculator = hydra::wrap_lambda(
			[] __hydra_dual__ ( unsigned int np, hydra::Vector4R* particles){

		hydra::Vector4R Jpsi = particles[0];
		hydra::Vector4R K    = particles[1];
		hydra::Vector4R pi   = particles[2];

		double M2_Jpsi_pi = (Jpsi + pi).mass2();
		double M2_Kpi     = (K + pi).mass2();

		return hydra::make_tuple(M2_Jpsi_pi, M2_Kpi);
	});

	//Device timing
	std::chrono::duration<double, std::milli> elapsed_generation;
	std::chrono::duration<double, std::milli> elapsed_histogram;
	{

		auto events = hydra::make_decays<3>(policy, nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, events.begin(), events.end());

		auto end = std::chrono::high_resolution_clock::now();

		elapsed_generation = end - start;


		auto dalitz_variables = hydra::make_range( events.GetUnweightedDecays().begin(),
				events.GetUnweightedDecays().end(), dalitz_calculator);

		auto dalitz_weights   = events.GetWeights();

		start = std::chrono::high_resolution_clock::now();

		Dalitz.Fill(dalitz_variables.begin(), dalitz_variables.end(),
					               dalitz_weights.begin()  );

		end = std::chrono::high_resolution_clock::now();

		elapsed_histogram = end - start;

		for(size_t i=0; i< 100; i++){
			for(size_t j=0; j< 100; j++){

				histogram->SetBinContent(i+1, j+1, Dalitz.GetBinContent({i,j}) );
			}
		}

	}


	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------------------------------"   << std::endl;
	std::cout << "| B0 -> J/psi K pi"                         << std::endl;
	std::cout << "| Number of events :"<< nentries             << std::endl;
	std::cout << "| generation (ms):"<< elapsed_generation.count()<< std::endl;
	std::cout << "|  histogram (ms):"<< elapsed_histogram.count()<< std::endl;
	std::cout << "-----------------------------------------"  << std::endl;


}

#endif /* THREE_BODY_PHSP_C_ */
