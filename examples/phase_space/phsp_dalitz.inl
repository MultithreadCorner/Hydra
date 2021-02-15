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
 * phsp_dalitz.inl
 *
 *  Created on: 24/12/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_DALITZ_INL_
#define PHSP_DALITZ_INL_

/**
 * \example phsp_dalitz.inl
 * This example shows how to use the Hydra's
 * Dalitz phase space Monte Carlo algorithms to
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
#include <hydra/Tuple.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/DenseHistogram.h>
#include <hydra/DalitzPhaseSpace.h>
#include <hydra/Function.h>
#include <hydra/Range.h>
#include <hydra/Lambda.h>
#include <hydra/multiarray.h>
#include <hydra/multivector.h>

#include <hydra/Placeholders.h>
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
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>

#endif //_ROOT_AVAILABLE_

//---------------------------

using namespace hydra::placeholders;

int main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line

	double P_mass = 0.493677;  //5.27955;
		double A_mass = 0.13957061;//3.0969;
		double B_mass = 0.13957061;//0.493677;
		double C_mass = 0.13957061;// pi mass


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
	TH2D Dalitz_d("Dalitz_d",
			"3-body phase-space (hydra::DalitzPhaseSpace);"
			"M^{2}(A B) [GeV^{2}/c^{4}];"
			"M^{2}(B C) [GeV^{2}/c^{4}]",
			100, pow(A_mass + B_mass,2), pow(P_mass - C_mass,2),
			100, pow(B_mass + C_mass,2), pow(P_mass - A_mass,2));

	TH2D Dalitz_r("Dalitz_r",
				"3-body phase-space (hydra::dalitz_range);"
				"M^{2}(A B) [GeV^{2}/c^{4}];"
				"M^{2}(B C) [GeV^{2}/c^{4}]",
				100, pow(A_mass + B_mass,2), pow(P_mass - C_mass,2),
				100, pow(B_mass + C_mass,2), pow(P_mass - A_mass,2));



#endif

	double masses[3]{A_mass, B_mass, C_mass };

	hydra::DalitzPhaseSpace<> dalitz(P_mass, masses );

	hydra::multiarray<double,4,hydra::device::sys_t> data(nentries);

	hydra::multiarray<double,2,hydra::device::sys_t> cos_theta_data(nentries);

	auto start = std::chrono::high_resolution_clock::now();

	dalitz.Generate(data);

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






     hydra::DenseHistogram<double, 2, hydra::device::sys_t> HistFromData_Dalitz(
    		{100,100},
    		{std::pow(A_mass + B_mass,2), std::pow(B_mass + C_mass,2)},
    		{std::pow(P_mass - C_mass,2), std::pow(P_mass - A_mass,2)} );

    hydra::DenseHistogram<double, 2, hydra::device::sys_t> HistFromRange_Dalitz(HistFromData_Dalitz);

    auto dalitz_range  = hydra::dalitz_range(P_mass, masses, 0x123abc, nentries);

    auto dalitz_events = dalitz_range | hydra::wrap_lambda(
    		[]__hydra_dual__ ( hydra::tuple<double, double, double, double> const& event  ){

    	return hydra::make_tuple( hydra::get<1>(event), hydra::get<3>(event));
    } ) ;

    auto dalitz_weights = dalitz_range | hydra::wrap_lambda(
      		[]__hydra_dual__ ( hydra::tuple<double, double, double, double> const&  event  ){

      	return hydra::get<0>(event);
      } ) ;

    //fill histograms
    start = std::chrono::high_resolution_clock::now();

    HistFromData_Dalitz.Fill(hydra::columns(data, _1, _3), hydra::columns(data, _0) );

	end = std::chrono::high_resolution_clock::now();

	elapsed = end - start;

	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------- Device ----------------"<< std::endl;
	std::cout << "| Histogram (Data)"                       << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	start = std::chrono::high_resolution_clock::now();

    HistFromRange_Dalitz.Fill(dalitz_events, dalitz_weights );

	end = std::chrono::high_resolution_clock::now();

	elapsed = end - start;

	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------- Device ----------------"<< std::endl;
	std::cout << "| Histogram (Range)"                      << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	auto cos_Theta12 = hydra::wrap_lambda(
			[P_mass, A_mass, B_mass, C_mass] __hydra_dual__ ( hydra::tuple<double, double, double> const&  event ){

		double MSq_max = std::pow(P_mass - A_mass,2.0);
		double MSq_min = std::pow(B_mass + C_mass,2.0);

		return (MSq_max + MSq_min - 2.0*hydra::get<2>(event))/(MSq_max - MSq_min );

	  }
	);

	start = std::chrono::high_resolution_clock::now();

	auto cos_Theta12_average = dalitz.AverageOn(hydra::device::sys, cos_Theta12, nentries);

	end = std::chrono::high_resolution_clock::now();

	elapsed = end - start;

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------- Device ----------------"<< std::endl;
	std::cout << "| Average of cos(theta)"                  << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "| Value:           :" << cos_Theta12_average.first << "+/-" <<  cos_Theta12_average.second  << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	start = std::chrono::high_resolution_clock::now();

	dalitz.Evaluate(cos_theta_data, cos_Theta12 );

	end = std::chrono::high_resolution_clock::now();

	elapsed = end - start;

	//output
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "----------------- Device ----------------"<< std::endl;
	std::cout << "| cos( theta | P -> A B C)"               << std::endl;
	std::cout << "| Number of events :"<< nentries          << std::endl;
	std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
	std::cout << "-----------------------------------------"<< std::endl;

	for(size_t i=0; i<10; ++i)
        std::cout << cos_theta_data[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		for(size_t i=0; i< 100; i++){
			for(size_t j=0; j< 100; j++){

				Dalitz_d.SetBinContent(i+1, j+1, HistFromData_Dalitz.GetBinContent({i,j}) );
				Dalitz_r.SetBinContent(i+1, j+1, HistFromRange_Dalitz.GetBinContent({i,j}) );
			}
		}

		TApplication *m_app=new TApplication("myapp",0,0);


		TCanvas canvas_d("canvas_d", "Phase-space Device", 500, 500);
		Dalitz_d.Draw("colz");

		TCanvas canvas_r("canvas_r", "Phase-space Device", 500, 500);
		Dalitz_r.Draw("colz");


		m_app->Run();

#endif

	return 0;
}

#endif /* PHSP_DALITZ_INL_ */
