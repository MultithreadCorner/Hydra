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
 * HydraPhaseSpaceExample.cu
 *
 *  Created on: Sep 22, 2016
 *      Author: Antonio Augusto Alves Junior
 */


#include <iostream>
#include <assert.h>
#include <time.h>
#include <vector>
#include <array>
#include <chrono>
//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/experimental/Vector4R.h>
#include <hydra/experimental/PhaseSpace.h>
#include <hydra/experimental/Events.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Copy.h>
//root
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

using namespace std;
using namespace hydra;



GInt_t main(int argv, char** argc)
{


	size_t  nentries       = 0;
	GReal_t mother_mass    = 0;
	GReal_t daughter1_mass = 0;
	GReal_t daughter2_mass = 0;
	GReal_t daughter3_mass = 0;

	GReal_t gand_daughter1_mass = 0;
	GReal_t gand_daughter2_mass = 0;


	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<GULong_t> NArg("n", "number-of-events",
				"Number of events",
				true, 5e6, "long");
		cmd.add(NArg);

		TCLAP::ValueArg<GReal_t> MassMotherArg("P", "parent-mass",
				"Mass of mother particle",
				true, 0.0, "double");
		cmd.add(MassMotherArg);

		TCLAP::ValueArg<GReal_t> MassDaughter1Arg("A", "daughter-A-mass",
				"Mass of daughter particle 'A' [P -> A B [C -> a b]]",
				true, 0.0, "double");
		cmd.add(MassDaughter1Arg);

		TCLAP::ValueArg<GReal_t> MassDaughter2Arg("B", "daughter-B-mass",
				"Mass of daughter particle 'B' [P -> A B [C -> a b]]",
				true, 0.0, "double");
		cmd.add(MassDaughter2Arg);

		TCLAP::ValueArg<GReal_t> MassDaughter3Arg("C", "daughter-C-mass",
				"Mass of daughter particle 'C' [P -> A B [C -> a b]]",
				true, 0.0, "double");
		cmd.add(MassDaughter3Arg);

		TCLAP::ValueArg<GReal_t> MassGrandDaughter1Arg("a", "grand-daughter-a-mass",
				"Mass of grand-daughter particle 'a' [P -> A B [C -> a b]]",
				true, 0.0, "double");
		cmd.add(MassGrandDaughter1Arg);

		TCLAP::ValueArg<GReal_t> MassGrandDaughter2Arg("b", "grand-daughter-b-mass",
				"Mass of grand-daughter particle 'b' [P -> A B [C -> a b]]",
				true, 0.0, "double");
		cmd.add(MassGrandDaughter2Arg);



		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries       = NArg.getValue();
		mother_mass    = MassMotherArg.getValue();
		daughter1_mass = MassDaughter1Arg.getValue();
		daughter2_mass = MassDaughter2Arg.getValue();
		daughter3_mass = MassDaughter3Arg.getValue();
		gand_daughter1_mass = MassGrandDaughter1Arg.getValue();
		gand_daughter2_mass = MassGrandDaughter2Arg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
																<< std::endl;
	}


	//----------------
		// P-> A B C
		//----------------
		hydra::experimental::Vector4R P(mother_mass, 0.0, 0.0, 0.0);
		GReal_t massesP[3]{daughter1_mass, daughter2_mass, daughter3_mass };

		// Create PhaseSpace object for B0-> K pi J/psi
		hydra::experimental::PhaseSpace<3> phsp_P(mother_mass, massesP);

		hydra::experimental::Events<3, device> P2ABC_Events_d(nentries);

		auto start1 = std::chrono::high_resolution_clock::now();
		phsp_P.Generate(P, P2ABC_Events_d.begin(), P2ABC_Events_d.end());
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| P -> A B C | Time (ms) ="<< elapsed1.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		for( size_t i=0; i<10; i++ ){
			cout << P2ABC_Events_d[i] << endl;
		}


	   //----------------
	   // C-> a b
	   //----------------
		GReal_t massesC[2]{gand_daughter1_mass, gand_daughter2_mass };
		// Create PhaseSpace object for J/psi->mu+ mu-
		hydra::experimental::PhaseSpace<2> phsp_C(daughter1_mass , massesC);

		hydra::experimental::Events<2, device> C2ab_Events_d(nentries);


		auto start2 = std::chrono::high_resolution_clock::now();
		phsp_C.Generate( P2ABC_Events_d.DaughtersBegin(0), P2ABC_Events_d.DaughtersEnd(0)
				, C2ab_Events_d.begin());
		auto end2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| C -> a b | Time (ms) ="<< elapsed2.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		for( size_t i=0; i<10; i++ ){
			cout << C2ab_Events_d[i] << endl;
		}


		typedef hydra::experimental::Events<3, device>::value_type event3_t;
		typedef hydra::experimental::Events<2, device>::value_type event2_t;
		typedef thrust::tuple< event3_t, event2_t> chain_t;


		auto Weight = [] __host__ __device__ ( chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;
			auto   p_weight = thrust::get<0>(p_decay);
			auto   c_weight = thrust::get<0>(c_decay);

			return 	p_weight * c_weight;

		};

		auto MB0 = [] __host__ __device__ ( chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;

			hydra::experimental::Vector4R p1 = thrust::get<1>(p_decay);
			hydra::experimental::Vector4R p2 = thrust::get<2>(p_decay);
			hydra::experimental::Vector4R p3 = thrust::get<3>(p_decay);

			return ( p1 + p2 + p3 ).mass();
		};

		auto MC = [] __host__ __device__ ( chain_t event )
			{
				auto   p_decay  = thrust::get<0>(event) ;
				auto   c_decay  = thrust::get<1>(event) ;

				hydra::experimental::Vector4R p1 = thrust::get<1>(c_decay);
				hydra::experimental::Vector4R p2 = thrust::get<2>(c_decay);

				return ( p1 + p2  ).mass();
			};

		auto M12 = [] __host__ __device__ (chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;

			hydra::experimental::Vector4R p1 = thrust::get<1>(p_decay);
			hydra::experimental::Vector4R p2 = thrust::get<2>(p_decay);

			return  ( p1 + p2).mass();

		};

		auto M13 = [] __host__ __device__( chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;

			hydra::experimental::Vector4R p1 = thrust::get<1>(p_decay);
			hydra::experimental::Vector4R p3 = thrust::get<3>(p_decay);

			return  ( p1 + p3 ).mass();
		};

		auto M23 = [] __host__ __device__( chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;

			hydra::experimental::Vector4R p2 = thrust::get<2>(p_decay);
			hydra::experimental::Vector4R p3 = thrust::get<3>(p_decay);

			return  ( p2 + p3 ).mass();
		};

		auto COSHELANG23 = [] __host__ __device__ ( chain_t event )
		{
			auto   p_decay  = thrust::get<0>(event) ;
			auto   c_decay  = thrust::get<1>(event) ;

			hydra::experimental::Vector4R p1 = thrust::get<1>(p_decay);
			hydra::experimental::Vector4R p2 = thrust::get<2>(p_decay);
			hydra::experimental::Vector4R p3 = thrust::get<3>(p_decay);

			hydra::experimental::Vector4R p = p1 + p2 + p3;
			hydra::experimental::Vector4R q = p2 + p3;


			GReal_t pd = p * p2;
			GReal_t pq = p * q;
			GReal_t qd = q * p2;
			GReal_t mp2 = p.mass2();
			GReal_t mq2 = q.mass2();
			GReal_t md2 = p2.mass2();

			return (pd * mq2 - pq * qd)
					/ sqrt((pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2));

		};

		auto DELTA = [] __host__ __device__ ( chain_t event )
			{
				auto   p_decay  = thrust::get<0>(event) ;
				auto   c_decay  = thrust::get<1>(event) ;

				hydra::experimental::Vector4R d1 = thrust::get<1>(p_decay);
				hydra::experimental::Vector4R d2 = thrust::get<2>(p_decay);
				hydra::experimental::Vector4R d3 = thrust::get<3>(p_decay);

				hydra::experimental::Vector4R h1 = thrust::get<1>(c_decay);
				hydra::experimental::Vector4R h2 = thrust::get<2>(c_decay);

				hydra::experimental::Vector4R D = d2 + d3;

				hydra::experimental::Vector4R d1_perp = d2 - (D.dot(d2) / D.dot(D)) * D;
				hydra::experimental::Vector4R h1_perp = h1 - (D.dot(h1) / D.dot(D)) * D;

				// orthogonal to both D and d1_perp

				hydra::experimental::Vector4R d1_prime = D.cross(d1_perp);

				d1_perp = d1_perp / d1_perp.d3mag();
				d1_prime = d1_prime / d1_prime.d3mag();

				GReal_t x, y;

				x = d1_perp.dot(h1_perp);
				y = d1_prime.dot(h1_perp);

				GReal_t chi = atan2(y, x);

				if (chi < 0.0)
					chi += 2.0*PI;

				return chi;

			};



		auto Weight_W  = wrap_lambda(Weight);

		auto MB0_W     = wrap_lambda(MB0);
		auto MC_W      = wrap_lambda(MC);
		auto M12_W     = wrap_lambda(M12);
		auto M13_W     = wrap_lambda(M13);
		auto M23_W     = wrap_lambda(M23);
		auto COSHELANG23_W = wrap_lambda( COSHELANG23);
		auto DELTA_W       = wrap_lambda( DELTA);


		auto functors = thrust::make_tuple(Weight_W, MB0_W, MC_W, M12_W, M13_W, M23_W, COSHELANG23_W, DELTA_W);

		auto start3 = std::chrono::high_resolution_clock::now();
		auto result_d = eval( functors,  P2ABC_Events_d.begin(), P2ABC_Events_d.end(), C2ab_Events_d.begin());
		auto end3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| Eval. Functors | Time (ms) ="<< elapsed3.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		for( size_t i=0; i<10; i++ ){
			cout << result_d[i] << endl;
		}

		return 0;


		hydra::experimental::Events<3, host> P2ABC_Events_h(P2ABC_Events_d);

		TH2D dalitz_AC("dalitz_AC", ";M^{2}(BC) [GeV^{2}/c^{4}]; M^{2}(AC) [GeV^{2}/c^{4}]",
				100, pow(daughter2_mass+daughter3_mass,2), pow(mother_mass - daughter1_mass,2),
				100, pow(daughter1_mass+daughter3_mass,2), pow(mother_mass - daughter2_mass,2));

		TH2D dalitz_BC("dalitz_BC", ";M^{2}(BC) [GeV^{2}/c^{4}]; M^{2}(AB) [GeV^{2}/c^{4}]",
				100, pow(daughter2_mass+daughter3_mass,2), pow(mother_mass - daughter1_mass,2),
				100, pow(daughter1_mass+daughter2_mass,2), pow(mother_mass - daughter3_mass,2));

		TH2D dalitz_AB("dalitz_AB", ";M^{2}(AC) [GeV^{2}/c^{4}]; M^{2}(AB) [GeV^{2}/c^{4}]",
				100, pow(daughter1_mass+daughter3_mass,2), pow(mother_mass - daughter2_mass,2),
				100, pow(daughter1_mass+daughter2_mass,2), pow(mother_mass - daughter3_mass,2));

		TH1D  mass_AB("mass_AB", ";M(AB) [GeV^{2}/c^{4}];Events",
				100, daughter2_mass+daughter3_mass, mother_mass - daughter1_mass );

		TH1D  mass_AC("mass_AC", ";M(AC) [GeV^{2}/c^{4}];Events",
					100, daughter1_mass+daughter3_mass, mother_mass - daughter2_mass );

		TH1D  mass_BC("mass_BC", ";M(BC) [GeV^{2}/c^{4}];Events",
					100, daughter2_mass+daughter1_mass, mother_mass - daughter3_mass );

		TH1D  coshelang_BC("coshelang_BC", ";cos #theta_{BC};Events", 100, -1.0, 1.0);

		TH1D  delta_angle("delta_angle", ";#delta #phi;Events", 100, 0.0, 2.0*PI);


		for(auto event: P2ABC_Events_h){

			GReal_t weight = thrust::get<0>(event);

			hydra::experimental::Vector4R A  = thrust::get<1>(event);
			hydra::experimental::Vector4R B    = thrust::get<2>(event);
			hydra::experimental::Vector4R C   = thrust::get<3>(event);

			hydra::experimental::Vector4R AB = A + B;
			hydra::experimental::Vector4R AC = A + C;
			hydra::experimental::Vector4R BC = B + C;

			GReal_t AB_M2 = AB.mass2();
			GReal_t AC_M2 = AC.mass2();
			GReal_t BC_M2 = BC.mass2();


			dalitz_AC.Fill( BC_M2, AC_M2,  weight);
			dalitz_BC.Fill( BC_M2, AB_M2,  weight);
			dalitz_AB.Fill( AC_M2, AB_M2,  weight);

		}

		auto result_h = get_copy<host>(result_d);

		for(auto entry:result_h ){

			GReal_t _weight    = thrust::get<0>(entry);
			GReal_t _mass_AB   = thrust::get<3>(entry);
			GReal_t _mass_AC   = thrust::get<4>(entry);
			GReal_t _mass_BC   = thrust::get<5>(entry);
			GReal_t _coshelang_BC = thrust::get<6>(entry);
			GReal_t _delta_angle  = thrust::get<7>(entry);

			mass_AB.Fill( _mass_BC , _weight);
			mass_AC.Fill( _mass_AC , _weight);
			mass_BC.Fill( _mass_AB , _weight);
			coshelang_BC.Fill(_coshelang_BC, _weight);
			delta_angle.Fill( _delta_angle, _weight);

		}

		//return 0;
		TApplication *myapp=new TApplication("myapp",0,0);

		TCanvas canvas_PHSP_AC("canvas_PHSP_AC", "Phase-space", 500, 500);
		dalitz_AC.Draw("colz");
		canvas_PHSP_AC.Print("plots/PHSP_AC.png");

		TCanvas canvas_PHSP_BC("canvas_PHSP_BC", "Phase-space", 500, 500);
		dalitz_BC.Draw("colz");
		canvas_PHSP_BC.Print("plots/PHSP_BC.png");

		TCanvas canvas_PHSP_AB("canvas_PHSP_AB", "Phase-space", 500, 500);
		dalitz_AB.Draw("colz");
		canvas_PHSP_AB.Print("plots/PHSP_AB.png");

		TCanvas canvas_mass_AB("canvas_masss_AB", "Mass", 500, 500);
		mass_AB.Draw("hist");
		mass_AB.SetMinimum(0.0);
		canvas_mass_AB.Print("plots/mass_AB.png");

		TCanvas canvas_mass_AC("canvas_masss_AC", "Mass", 500, 500);
		mass_AC.Draw("hist");
		mass_AC.SetMinimum(0.0);
		canvas_mass_AC.Print("plots/mass_AC.png");

		TCanvas canvas_mass_BC("canvas_masss_BC", "Mass", 500, 500);
		mass_BC.Draw("hist");
		mass_BC.SetMinimum(0.0);
		canvas_mass_BC.Print("plots/mass_BC.png");

		TCanvas canvas_coshelang("canvas_coshelang", "coshelang", 500, 500);
		coshelang_BC.Draw("hist");
		coshelang_BC.SetMinimum(0.0);
		canvas_coshelang.Print("plots/mass_BC.png");


		TCanvas canvas_delta_angle("canvas_delta_angle", "delta_angle", 500, 500);
		delta_angle.Draw("hist");
		delta_angle.SetMinimum(0.0);
			canvas_delta_angle.Print("plots/delta_angle.png");







		myapp->Run();

		return 0;
}
