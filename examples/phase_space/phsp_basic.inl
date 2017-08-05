/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * phsp_basic.inl
 *
 *  Created on: Jul 7, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_BASIC_INL_
#define PHSP_BASIC_INL_


/**
 * @example phsp_basic.inl
 * This example shows how to use the Hydra's
 * phase space Monte Carlo algorithms to
 * generate a sample of B0 -> J/psi K pi and
 * plot the Dalitz plot.
 */


/*---------------------------------
 * std
 * ---------------------------------
 */
#include <iostream>
#include <assert.h>
#include <time.h>
#include <vector>
#include <array>
#include <chrono>

/*---------------------------------
 * command line arguments
 *---------------------------------
 */
#include <tclap/CmdLine.h>

/*---------------------------------
 * Include hydra classes and
 * algorithms for
 *--------------------------------
 */
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Events.h>
#include <hydra/Chain.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Copy.h>
#include <hydra/Tuple.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Decays.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TF1.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>

#endif //_ROOT_AVAILABLE_



int main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line
	double B0_mass    = 5.27955;   // B0 mass
	double Jpsi_mass  = 3.0969;    // J/psi mass
	double K_mass     = 0.493677;  // K+ mass
	double pi_mass    = 0.13957061;// pi mass


	try {

		TCLAP::CmdLine cmd("Command line arguments for PHSP B0 -> J/psi K pi", '=');

		TCLAP::ValueArg<size_t> NArg("n",
				"nevents",
				"Number of events to generate. Default is [ 10e6 ].",
				true, 10e6, "unsigned long");
		cmd.add(NArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries       = NArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
																<< std::endl;
	}

#ifdef 	_ROOT_AVAILABLE_
	//
	TH2D Dalitz_d("Dalitz_d", "Device;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH2D Dalitz_h("Dalitz_h", "Host;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));
#endif

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
	double masses[3]{Jpsi_mass, K_mass, pi_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp(B0_mass, masses);

	//device
	{
		//allocate memory to hold the final states particles
		hydra::Events<3, hydra::device::sys_t > Events_d(nentries);

		hydra::Decays<3, hydra::device::sys_t > Decays_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_d.begin(), Events_d.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//print
		for( size_t i=0; i<10; i++ )
			std::cout << Events_d[i] << std::endl;



#ifdef 	_ROOT_AVAILABLE_

		//bring events to CPU memory space
		hydra::Events<3, hydra::host::sys_t > Events_h(Events_d);

		for( auto event : Events_h ){

			double weight        = hydra::get<0>(event);
			hydra::Vector4R Jpsi = hydra::get<1>(event);
			hydra::Vector4R K    = hydra::get<2>(event);
			hydra::Vector4R pi   = hydra::get<3>(event);

			double M2_Jpsi_pi = (Jpsi + pi).mass2();
			double M2_Kpi     = (K + pi).mass2();

			Dalitz_d.Fill( M2_Jpsi_pi, M2_Kpi, weight);
		}

#endif

	}

	//host
	{
		//allocate memory to hold the final states particles
		hydra::Events<3, hydra::host::sys_t > Events_h(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_h.begin(), Events_h.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "------------------ Host -----------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi"                   << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//print
		for( size_t i=0; i<10; i++ )
			std::cout << Events_h[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		for( auto event : Events_h ){

			double weight        = hydra::get<0>(event);
			hydra::Vector4R Jpsi = hydra::get<1>(event);
			hydra::Vector4R K    = hydra::get<2>(event);
			hydra::Vector4R pi   = hydra::get<3>(event);

			double M2_Jpsi_pi = (Jpsi + pi).mass2();
			double M2_Kpi     = (K + pi).mass2();

			Dalitz_h.Fill(M2_Jpsi_pi, M2_Kpi, weight );
		}

#endif

	}


#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	TCanvas canvas_h("canvas_h", "Phase-space Host", 500, 500);
	Dalitz_h.Draw("colz");
	canvas_h.Print("plots/phsp_basic_h.png");

	TCanvas canvas_d("canvas_d", "Phase-space Device", 500, 500);
	Dalitz_d.Draw("colz");
	canvas_d.Print("plots/phsp_basic_d.png");

	m_app->Run();

#endif

	return 0;
}


#endif /* PHSP_BASIC_INL_ */
