/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * generate_decay_chain.C
 *
 *  Created on: 25/03/2018
 *      Author: Davide Brundu
 * 
 *  Generate the decay chain D0 -> f0 (-> pi pi) rho (-> mu mu) 
 *  Plot the Dalitz plot andt the Delta Angle between decay planes
 *  both using the lambda and the hydra functor
 *
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
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Lambda.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/DenseHistogram.h>
#include <hydra/Placeholders.h>
#include <hydra/multivector.h>

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
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


// B0 -> [J/psi] K+ pi-
//             \-> mu+ mu-

declarg(Jpsi, hydra::Vector4R)
declarg(Kaon, hydra::Vector4R)
declarg(Pion, hydra::Vector4R)
declarg(MuonP, hydra::Vector4R)
declarg(MuonM, hydra::Vector4R)

//phase-space variables
declarg(M13Sq, double)
declarg(M23Sq, double)
declarg(CosTheta, double)
declarg(DeltaPhi, double)
declarg(Weight, double)


using namespace hydra::placeholders;
using namespace hydra::arguments;



void generate_4body_decay_chain(size_t  nentries =100000)
{


	double B0_mass   = 5.27955;     // B0 mass
	double Jpsi_mass = 3.0969;      // J/psi mass
	double K_mass    = 0.493677;    // K+ mass
	double pi_mass   = 0.13957061;  // pi mass
	double mup_mass  = 0.1056583745;// mu mass
	double mum_mass  = 0.1056583745;// mu mass



	TH2D* DalitzHist = new TH2D("Dalitz_d", ";M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D* CosThetaHist = new TH1D("CosTheta_d", "; cos(#theta_{K*}); Events", 100, -1.0, 1.0);

	TH1D* DeltaPhiHist = new TH1D("Delta_d", "; #Delta #phi; Events", 100, 0.0, 3.5);

	auto Variables  = hydra::wrap_lambda(
			[] __hydra_dual__ (Jpsi jpsi, Kaon kaon, Pion pion , MuonP mup, MuonM mum )
	{

		hydra::Vector4R B0    = jpsi+kaon+pion;
		hydra::Vector4R Kstar = kaon+pion;
		hydra::Vector4R Zplus = jpsi+pion;

		//invariant masses
		M23Sq M2_Kpi    = Kstar.mass2();
		M13Sq M2_jpsipi = Zplus.mass2();


		//cosine of helicity angle

		double pd = B0 * kaon;
		double pq = B0 * Kstar;
		double qd = Kstar * kaon;
		double mp2 = B0.mass2();
		double mq2 = Kstar.mass2();
		double md2 = kaon().mass2();

		CosTheta cos_helangle = (pd * mq2 - pq * qd)	/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));

		//angle between the decay planes

		hydra::Vector4R d1_perp = kaon - (Kstar.dot(kaon) / Kstar.dot(Kstar)) * Kstar;
		hydra::Vector4R h1_perp = mup  - (Kstar.dot(mup)  / Kstar.dot(Kstar)) * Kstar;

		// orthogonal to both D and d1_perp
		hydra::Vector4R d1_prime = Kstar.cross(d1_perp);

		d1_perp  = d1_perp  / d1_perp.d3mag();
		d1_prime = d1_prime / d1_prime.d3mag();

		double x = d1_perp.dot(h1_perp);
		double y = d1_prime.dot(h1_perp);

		DeltaPhi chi = atan2(y, x);

		if(chi < 0.0) chi += 2.0*PI;



		return hydra::make_tuple(M2_jpsipi, M2_Kpi, cos_helangle, chi ) ;
	});


	//B0
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0 -> K pi J/psi
	hydra::PhaseSpace<3> ThreeBodyPHSP(B0_mass, {Jpsi_mass, K_mass, pi_mass });

	// Create PhaseSpace object for J/psi -> mu+ mu-
	hydra::PhaseSpace<2> TwoBodyPHSP(Jpsi_mass, {mup_mass , mum_mass});


	//device
	{
		//allocate memory to hold the final states particles
		auto ThreeBodyDecay = hydra::Decays< hydra::tuple<Jpsi,Kaon,Pion>,
				hydra::device::sys_t >( B0_mass, {Jpsi_mass, K_mass, pi_mass }, nentries);

		auto TwoBodyDecay   = hydra::Decays< hydra::tuple<MuonP,MuonM>,
						hydra::device::sys_t >(Jpsi_mass, { mup_mass , mup_mass}, nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles for B0 -> K pi J/psi
		ThreeBodyPHSP.Generate(B0, ThreeBodyDecay);

		//pass the list of J/psi to generate the final
		//state particles for J/psi -> mu+ mu-
		TwoBodyPHSP.Generate( ThreeBodyDecay.GetDaugtherRange(_0),  TwoBodyDecay);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

	    auto  chain   = ThreeBodyDecay.Meld( TwoBodyDecay ) | Variables;
	    auto  weights = ThreeBodyDecay | ThreeBodyDecay.GetEventWeightFunctor();

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
			std::cout <<"Weight {"<< weights[i]  << "} | Event { " << chain[i]<< " }" << std::endl;




		//bring the data to host memory to fill the histograms faster
		hydra::multivector< hydra::tuple<M13Sq, M23Sq, CosTheta, DeltaPhi>, hydra::host::sys_t> variables(chain.size());
		hydra::copy( chain, variables);

		hydra::host::vector<Weight> Weights(chain.size());
		hydra::copy(weights, Weights );

		auto dataset = variables.meld(Weights);


		for( auto entry: dataset )
		{
			M13Sq M2_13       = hydra::get<M13Sq&>(entry);
			M23Sq M2_23       = hydra::get<M23Sq&>(entry);
			CosTheta cosTheta = hydra::get<CosTheta&>(entry);
			DeltaPhi deltaPhi = hydra::get<DeltaPhi&>(entry);
            Weight weight     = hydra::get<Weight&>(entry);

			DalitzHist->Fill(M2_13, M2_23, weight );
			CosThetaHist->Fill(cosTheta, weight );
			DeltaPhiHist->Fill(deltaPhi, weight );
		}

	}


	//--------------------------------------

	TCanvas* canvas1_d =  new TCanvas("canvas1_d", "Phase-space", 500, 500);
	DalitzHist->Draw("colz");

	TCanvas* canvas2_d =  new TCanvas("canvas2_d", "Phase-space", 500, 500);
	CosThetaHist->Draw("hist");
	CosThetaHist->SetMinimum(0.0);

	TCanvas* canvas3_d =  new TCanvas("canvas3_d", "Phase-space", 500, 500);
	DeltaPhiHist->Draw("hist");
	DeltaPhiHist->SetMinimum(0.0);

}
