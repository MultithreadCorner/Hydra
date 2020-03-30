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
 * phsp_basic.inl
 *
 *  Created on: Jul 7, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_REWEIGHTING_INL_
#define PHSP_REWEIGHTING_INL_


/**
 * \example phsp_reweighting.inl
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
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Decays.h>
#include <hydra/DenseHistogram.h>
#include <hydra/Range.h>
#include <hydra/functions/BreitWignerNR.h>

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

// Daughter particles

declarg(A, hydra::Vector4R)
declarg(B, hydra::Vector4R)
declarg(C, hydra::Vector4R)

//---------------------------
using namespace hydra::arguments;

int main(int argv, char** argc)
{

	size_t  nentries   = 0; // number of events to generate, to be get from command line

	double P_mass = 5.27955;
	double A_mass = 3.0969;
	double B_mass = 0.493677;
	double C_mass = 0.13957061;

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
	TH2D Dalitz_Flat("Dalitz_Flat",
			"PHSP;"
			"M^{2}(AB) [GeV^{2}/c^{4}];"
			"M^{2}(BC) [GeV^{2}/c^{4}]",
			100, pow(A_mass + B_mass,2), pow(P_mass - C_mass,2),
			100, pow(B_mass + C_mass,2), pow(P_mass - A_mass,2));


	//
	TH2D Dalitz_Resonant("Dalitz_Resonant",
			"Resonance;"
			"M^{2}(AB) [GeV^{2}/c^{4}];"
			"M^{2}(BC) [GeV^{2}/c^{4}]",
			100, pow(A_mass + B_mass,2), pow(P_mass - C_mass,2),
			100, pow(B_mass + C_mass,2), pow(P_mass - A_mass,2));


#endif

	hydra::Vector4R Parent(P_mass, 0.0, 0.0, 0.0);

	double masses[3]{A_mass, B_mass, C_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{P_mass, masses};

	auto dalitz_calculator = hydra::wrap_lambda(	[] __hydra_dual__ (A a, B b, C c) {

		return hydra::make_tuple( (a + b).mass2(), (b + c).mass2());
	});

	auto resonance = hydra::BreitWignerNR<double>( hydra::Parameter::Create("M0").Value(0.895),
								hydra::Parameter::Create("W").Value(0.055) );

	auto breit_wigner = hydra::wrap_lambda(	[ resonance ] __hydra_dual__ ( B b, C c ){
				return resonance( (b+c).mass());
	       } );

	//device
	{
		//allocate memory to hold the final states particles

		hydra::Decays<hydra::tuple<A,B,C>, hydra::device::sys_t > Events(P_mass, masses, nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(Parent, Events);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| P -> A B C"                             << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//the power of lazyness
		auto dalitz_variables = Events | dalitz_calculator ;

		auto dalitz_flat_weights   = Events | Events.GetEventWeightFunctor();

		auto dalitz_resonant_weights   = Events | Events.GetEventWeightFunctor(breit_wigner);


        auto Hist_Dalitz_Flat = hydra::make_dense_histogram<double,2>( hydra::device::sys,
				{100,100},
				{pow(A_mass + B_mass,2), pow(B_mass + C_mass,2)},
				{pow(P_mass - C_mass,2), pow(P_mass - A_mass,2)},
				dalitz_variables, dalitz_flat_weights);

        auto Hist_Dalitz_Resonant = hydra::make_dense_histogram<double,2>( hydra::device::sys,
				{100,100},
				{pow(A_mass + B_mass,2), pow(B_mass + C_mass,2)},
				{pow(P_mass - C_mass,2), pow(P_mass - A_mass,2)},
				dalitz_variables, dalitz_resonant_weights);



#ifdef 	_ROOT_AVAILABLE_

		//bring events to CPU memory space
		//hydra::Events<3, hydra::host::sys_t > Events_h(Events_d);

        for(size_t i=0; i< 100; i++){
        			for(size_t j=0; j< 100; j++){

        				Dalitz_Flat.SetBinContent(i+1, j+1, Hist_Dalitz_Flat.GetBinContent({i,j}) );
        				Dalitz_Resonant.SetBinContent(i+1, j+1, Hist_Dalitz_Resonant.GetBinContent({i,j}) );

        			}
        		}

#endif

	}




#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);


	TCanvas canvas_d1("canvas_d1", "Phase-space Device", 500, 500);
	Dalitz_Flat.Draw("colz");

	TCanvas canvas_d2("canvas_d2", "Phase-space Device", 500, 500);
	Dalitz_Resonant.Draw("colz");

	m_app->Run();

#endif

	return 0;
}


#endif /* PHSP_REWEIGHTING_INL_ */
