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
 * phsp_evaluating_functor.inl
 *
 *  Created on: 14/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_EVALUATING_FUNCTOR_INL_
#define PHSP_EVALUATING_FUNCTOR_INL_


/**
 * @example phsp_evaluating_functor.inl
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
		std::cerr << "error: " << e.error() << " for arg " << e.argId()	<< std::endl;
	}

	//C++11 lambda for invariant mass
	auto M12Sq = [] __host__ __device__ (unsigned int n, hydra::Vector4R
			* p )
	{ return  ( p[0] + p[2]).mass2(); };


	//C++11 lambda for invariant mass
	auto M23Sq = [] __host__ __device__ (unsigned int n, hydra::Vector4R* p )
	{ return  ( p[1] + p[2]).mass2(); };


	//C++11 lambda for cosine of helicity angle Kpi
	auto COSHELANG = [] __host__ __device__ (unsigned int n, hydra::Vector4R* P )
	{
		hydra::Vector4R p = P[1] + P[2] + P[0];
		hydra::Vector4R q = P[2] + P[1];

		double pd = p * P[2];
		double pq = p * q;
		double qd = q * P[2];
		double mp2 = p.mass2();
		double mq2 = q.mass2();
		double md2 = P[2].mass2();

		return (pd * mq2 - pq * qd)
				/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));
	};

	//wrap functors
	auto cosTheta = hydra::wrap_lambda(COSHELANG);
	auto m12Sq    = hydra::wrap_lambda(M12Sq);
	auto m23Sq    = hydra::wrap_lambda(M23Sq);

#ifdef 	_ROOT_AVAILABLE_

	TH2D Dalitz_d("Dalitz_d", "Device;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D CosTheta_d("CosTheta_d", "Device; cos(#theta_{K*}), Events", 100, -1.0, 1.0);

	//---------

	TH2D Dalitz_h("Dalitz_h", "Host;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D CosTheta_h("CosTheta_h", "Host; cos(#theta_{K*}), Events", 100, -1.0, 1.0);

#endif

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
	double masses[3]{Jpsi_mass, K_mass, pi_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp(B0_mass, masses);

	//useful typedefs
	//dataset row with three doubles
	typedef hydra::tuple<double, double, double, double> row_t;
	//the dataset layout mimics a vector of tuples
	typedef hydra::device::vector<row_t> table_d;
	//the dataset layout mimics a vector of tuples
	typedef hydra::host::vector<row_t>   table_h;
	//multivector allocation
	typedef hydra::multivector<table_d> dataset_d;
	typedef hydra::multivector<table_h> dataset_h;

	//device
	{
		//allocate memory to hold the final dataset in
		//optimal memory layout using a hydra::multivector
		dataset_d data_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();
		//generate the final state particles
		phsp.Evaluate(B0, data_d.begin(), data_d.end(), m12Sq, m23Sq, cosTheta);
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
			std::cout << "<"<< i <<">: " << data_d[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		//bring events to CPU memory space
		dataset_d data_h(data_d);

		for( auto event : data_h ){

			double weight        = hydra::get<0>(event);
			double m12Sq         = hydra::get<1>(event);
			double m23Sq         = hydra::get<2>(event);
			double cosTheta      = hydra::get<3>(event);

			Dalitz_d.Fill(m12Sq , m23Sq, weight);
			CosTheta_d.Fill(cosTheta , weight);
		}

#endif

	}

	//host
	{
		//allocate memory to hold the final dataset in
		//optimal memory layout using a hydra::multivector
		dataset_h data_h(nentries);

		auto start = std::chrono::high_resolution_clock::now();
		//generate the final state particles
		phsp.Evaluate(B0, data_h.begin(), data_h.end(), m12Sq, m23Sq, cosTheta);
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
			std::cout <<  "<"<< i <<">: " << data_h[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		for( auto event : data_h ){

			double weight        = hydra::get<0>(event);
			double m12Sq         = hydra::get<1>(event);
			double m23Sq         = hydra::get<2>(event);
			double cosTheta      = hydra::get<3>(event);

			Dalitz_h.Fill(m12Sq , m23Sq, weight);
			CosTheta_h.Fill(cosTheta , weight);
		}

#endif

	}


#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	TCanvas canvas_h("canvas_h", "Phase-space Host", 500, 500);
	Dalitz_h.Draw("colz");
	canvas_h.Print("plots/phsp_evaluating_h.png");


	TCanvas canvas2_h("canvas2_h", "Phase-space Host", 500, 500);
	CosTheta_h.Draw("hist");
	CosTheta_h.SetMinimum(0.0);
	canvas2_h.Print("plots/phsp_evaluating_h2.png");

	//----------------------------

	TCanvas canvas_d("canvas_d", "Phase-space Device", 500, 500);
	Dalitz_d.Draw("colz");
	canvas_d.Print("plots/phsp_evaluating_d.png");

	TCanvas canvas2_d("canvas2_d", "Phase-space Device", 500, 500);
	CosTheta_d.Draw("hist");
	CosTheta_d.SetMinimum(0.0);
	canvas2_d.Print("plots/phsp_evaluating_d2.png");


	m_app->Run();

#endif

	return 0;
}


#endif /* PHSP_EVALUATING_FUNCTOR_INL_ */
