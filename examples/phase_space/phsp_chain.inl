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
 * phsp_chain.inl
 *
 *  Created on: Jul 10, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_CHAIN_INL_
#define PHSP_CHAIN_INL_

/**
 * \example phsp_chain.inl
 *
 * This example shows how to use the Hydra's
 * phase space Monte Carlo algorithms to
 * generate a sample of B0 -> J/psi K pi, J/psi-> mu+ mu- and
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

#include <hydra/PhaseSpace.h>
#include <hydra/Chains.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
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
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>

#endif //_ROOT_AVAILABLE_


using namespace hydra::placeholders;

int main(int argv, char** argc)
{


	size_t  nentries   = 0; // number of events to generate, to be get from command line

	double B0_mass    = 5.27955;      // B0 mass
	double Jpsi_mass  = 3.0969;       // J/psi mass
	double K_mass     = 0.13957061;   //0.493677;     // K+ mass
	double pi_mass    = 0.13957061;   // pi mass
	double mu_mass    = 0.1056583745 ;// mu mass


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

	TH2D Dalitz_d("Dalitz_d", "Device;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D CosTheta_d("CosTheta_d", "Device; cos(#theta_{K*}), Events", 100, -1.0, 1.0);

	TH1D    Delta_d("Delta_d", "Device; #delta #phi, Events", 100, 0.0, 3.5);

#endif

	//C++11 lambda for invariant mass
	auto M2 = [] __hydra_dual__ (hydra::Vector4R const& p1, hydra::Vector4R const& p2 )
	{ return  ( p1 + p2).mass2(); };


	//C++11 lambda for cosine of helicity angle Kpi
	auto COSHELANG = [] __hydra_dual__ (hydra::Vector4R const& p1, hydra::Vector4R const& p2, hydra::Vector4R const& p3  )
	{
		hydra::Vector4R p = p1 + p2 + p3;
		hydra::Vector4R q = p2 + p3;

		double pd = p * p2;
		double pq = p * q;
		double qd = q * p2;
		double mp2 = p.mass2();
		double mq2 = q.mass2();
		double md2 = p2.mass2();

		return (pd * mq2 - pq * qd)
				/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));
	};

	//C++11 lambda for angle between the planes [K,pi] and [mu+, mu-]
	auto DELTA = [] __hydra_dual__ (hydra::Vector4R const& d2, hydra::Vector4R const& d3,
			hydra::Vector4R const& h1, hydra::Vector4R const&  )
	{
		hydra::Vector4R D = d2 + d3;

		hydra::Vector4R d1_perp = d2 - (D.dot(d2) / D.dot(D)) * D;
		hydra::Vector4R h1_perp = h1 - (D.dot(h1) / D.dot(D)) * D;

		// orthogonal to both D and d1_perp
		hydra::Vector4R d1_prime = D.cross(d1_perp);

		d1_perp = d1_perp / d1_perp.d3mag();
		d1_prime = d1_prime / d1_prime.d3mag();

		double x, y;

		x = d1_perp.dot(h1_perp);
		y = d1_prime.dot(h1_perp);

		double chi = atan2(y, x);

		if(chi < 0.0) chi += 2.0*PI;

		return chi;
	};


	//B0
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	//K pi J/psi masses
	double masses1[3]{Jpsi_mass, K_mass, pi_mass };

	//mu masses
	double masses2[2]{mu_mass , mu_mass};

	// Create PhaseSpace object for B0 -> K pi J/psi
	hydra::PhaseSpace<3> phsp1( masses1);

	// Create PhaseSpace object for J/psi -> mu+ mu-
	hydra::PhaseSpace<2> phsp2( masses2);

	//device
	{
		//allocate memory to hold the final states particles
		auto Chain_d   = hydra::make_chain<3,2>(hydra::device::sys, nentries);


		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles for B0 -> K pi J/psi
		phsp1.Generate(B0, Chain_d.GetDecays(_0).begin(), Chain_d.GetDecays(_0).end());

		//pass the list of J/psi to generate the final
		//state particles for J/psi -> mu+ mu-
		phsp2.Generate(Chain_d.GetDecays(_0).GetDaughters(0).begin(), Chain_d.GetDecays(_0).GetDaughters(0).end(),
				Chain_d.GetDecays(_1).begin());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi | J/psi -> mu+ mu-"    << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//print
		for( size_t i=0; i<10; i++ )
			std::cout << Chain_d[i] << std::endl;


		//bring events to CPU memory space
		auto Chain_h   = hydra::make_chain<3,2>(hydra::host::sys, nentries);
		Chain_h   = Chain_d;

#ifdef 	_ROOT_AVAILABLE_
		for( auto event : Chain_h ) {

			auto   B0_decay    = hydra::get<1>(event) ;
			auto   Jpsi_decay  = hydra::get<2>(event) ;

			double weight        = hydra::get<0>(B0_decay );
			hydra::Vector4R Jpsi = hydra::get<1>(B0_decay );
			hydra::Vector4R K    = hydra::get<2>(B0_decay );
			hydra::Vector4R pi   = hydra::get<3>(B0_decay );

			hydra::Vector4R mup  = hydra::get<1>(Jpsi_decay );
			hydra::Vector4R mum  = hydra::get<2>(Jpsi_decay );

			double M2_Jpsipi  = M2(Jpsi, pi);
			double M2_Kpi     = M2(K, pi);
			double CosTheta   = COSHELANG(Jpsi, pi, K );
			double DeltaAngle = DELTA(K, pi, mup, mum );

			Dalitz_d.Fill(M2_Jpsipi, M2_Kpi , weight);
			CosTheta_d.Fill(CosTheta , weight);
			Delta_d.Fill(DeltaAngle, weight );
		}
#endif

	}




#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	//--------------------------------------

	TCanvas canvas1_d("canvas1_d", "Phase-space Host", 500, 500);
	Dalitz_d.Draw("colz");
	//canvas1_d.Print("plots/phsp_chain_d1.png");

	TCanvas canvas2_d("canvas2_d", "Phase-space Host", 500, 500);
	CosTheta_d.Draw("hist");
	//canvas2_d.Print("plots/phsp_chain_d2.png");

	TCanvas canvas3_d("canvas3_d", "Phase-space Host", 500, 500);
	Delta_d.Draw("hist");
	//canvas3_d.Print("plots/phsp_chain_d3.png");

	m_app->Run();

#endif

	return 0;
}




#endif /* PHSP_CHAIN_INL_ */
