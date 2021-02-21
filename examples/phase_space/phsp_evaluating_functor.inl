/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * phsp_evaluating_functor.inl
 *
 *  Created on: 14/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_EVALUATING_FUNCTOR_INL_
#define PHSP_EVALUATING_FUNCTOR_INL_


/**
 * \examples phsp_evaluating_functor.inl
 *
 * This example shows how to use the Hydra's
 * phase space Monte Carlo algorithms to
 * generate events of B0 -> J/psi K pi on fly and
 * evaluate a set of functors and plot the distributions.
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
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Lambda.h>
#include <hydra/multiarray.h>
#include <hydra/Algorithm.h>
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

// B0 -> J/psi K+ pi-
//         |-> mu+ mu-

declarg(Jpsi, hydra::Vector4R)
declarg(Kaon, hydra::Vector4R)
declarg(Pion, hydra::Vector4R)
declarg(MuonP, hydra::Vector4R)
declarg(MuonM, hydra::Vector4R)

//phase-space variables
declarg(Weight, double)
declarg(M13Sq, double)
declarg(M23Sq, double)
declarg(CosTheta, double)


using namespace hydra::placeholders;
using namespace hydra::arguments;

int main(int argv, char** argc)
{


	size_t  nentries  = 0; // number of events to generate, to be get from command line
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
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

	//invariant masses
	auto M13_Sq =  hydra::wrap_lambda(
			[] __hydra_dual__ (Jpsi jpsi, Kaon kaon, Pion pion  ){

		     return  ( jpsi + pion ).mass2();
		}
	);

	auto M23_Sq = hydra::wrap_lambda(
			[] __hydra_dual__ (  Jpsi jpsi, Kaon kaon, Pion pion ){

		     return  ( kaon + pion ).mass2();
	   }
	);


	//cosine of helicity angle Kpi
	auto cosTheta =  hydra::wrap_lambda(
			[] __hydra_dual__ ( Jpsi jpsi, Kaon kaon, Pion pion )
	{
		hydra::Vector4R p =  jpsi+kaon+pion;
		hydra::Vector4R q = kaon+pion;

		double pd = p * kaon;
		double pq = p * q;
		double qd = q * kaon;
		double mp2 = p.mass2();
		double mq2 = q.mass2();
		double md2 = kaon().mass2();

		return (pd * mq2 - pq * qd)
				/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));
	    }
	);



#ifdef 	_ROOT_AVAILABLE_

	TH2D DalitzHist("Dalitz", ";M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D CosThetaHist("CosTheta", "; cos(#theta_{K*}); Events", 100, -1.0, 1.0);

#endif

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{B0_mass, {Jpsi_mass, K_mass, pi_mass } };

	//device
	{
		//allocate memory to hold the final dataset in
		//optimal memory layout using a hydra::multivector
		hydra::multivector<hydra::tuple<Weight, M13Sq, M23Sq, CosTheta>, hydra::device::sys_t> dataset(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Evaluate(B0, dataset,  M13_Sq, M23_Sq, cosTheta);

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
			std::cout << "<"<< i <<">: " << dataset[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		//bring events to CPU memory space
		hydra::multivector<hydra::tuple<Weight,M13Sq, M23Sq, CosTheta>, hydra::host::sys_t> data(dataset);

		for( auto event : data ){

			double weight        = hydra::get<Weight&>(event);
			double m13Sq         = hydra::get<M13Sq&>(event);
			double m23Sq         = hydra::get<M23Sq&>(event);
			double cos_theta     = hydra::get<CosTheta&>(event);

			DalitzHist.Fill(m13Sq , m23Sq, weight);
			CosThetaHist.Fill(cos_theta , weight);
		}

#endif

	}



#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	//----------------------------

	TCanvas canvas_d("canvas_d", "Phase-space Device", 500, 500);
	DalitzHist.Draw("colz");

	TCanvas canvas2_d("canvas2_d", "Phase-space Device", 500, 500);
	CosThetaHist.Draw("hist");
	CosThetaHist.SetMinimum(0.0);


	m_app->Run();

#endif

	return 0;
}


#endif /* PHSP_EVALUATING_FUNCTOR_INL_ */
