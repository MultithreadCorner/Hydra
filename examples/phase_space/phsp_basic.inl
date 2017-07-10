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
#include <hydra/host/System.h>
#include <hydra/device/System.h>

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




using namespace std;
using namespace hydra;


GInt_t main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line
	GReal_t B0_mass    = 5.27955;   // B0 mass
	GReal_t Jpsi_mass  = 3.0969;    // J/psi mass
	GReal_t K_mass     = 0.493677;  // K+ mass
	GReal_t pi_mass    = 0.13957061;// pi mass


	try {

		TCLAP::CmdLine cmd("Command line arguments for PHSP B0 -> J/psi K pi", '=');

		TCLAP::ValueArg<GULong_t> NArg("n",
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

#ifdef 	_ROOT_AVAILABLE
	//
	TH2D Dalitz_M12_d("Dalitz_M12_d", ";M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH2D Dalitz_M12_h("Dalitz_M12_h", ";M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
				100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
				100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));
#endif

	Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
	GReal_t masses[3]{Jpsi_mass, K_mass, pi_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	PhaseSpace<3> phsp(B0_mass, masses);

	//device
	{
		//allocate memory to hold the final states particles
		Events<3, hydra::device::sys_t > Events_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_d.begin(), Events_d.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi"                   << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//print
		for( size_t i=0; i<10; i++ )
			std::cout << Events_d[i] << std::endl;



#ifdef 	_ROOT_AVAILABLE_

		for( size_t i=0; i< Events_d.size(); i++ )


#endif

	}

	//host
	{

		Events<3, hydra::device::sys_t > Events_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();
		phsp.Generate(P, Events_d.begin(), Events_d.end());
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = end - start;
		//time
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi"                   << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		for( size_t i=0; i<10; i++ ){
			std::cout << Events_d[i] << std::endl;
		}
	}



	//return 0;
	TApplication *myapp=new TApplication("myapp",0,0);

	TCanvas canvas_PHSP_AC("canvas_PHSP_AC", "Phase-space", 500, 500);
	dalitz_AC.Draw("colz");
	canvas_PHSP_AC.Print("plots/PHSP_AC.png");


myapp->Run();

*/






	return 0;
}


#endif /* PHSP_BASIC_INL_ */
