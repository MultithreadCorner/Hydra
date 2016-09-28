/*
 * HydraPhaseSpaceExample.cu
 *
 *  Created on: Sep 22, 2016
 *      Author: augalves
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <map>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>
#include <type_traits>
#include <typeinfo>
//command line
#include <tclap/CmdLine.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Containers.h>
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
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooCFunction1Binding.h"
#include "RooTFnBinding.h"
#include "RooPlot.h"

#include <src/Gauss.h>
#include <src/Exp.h>

using namespace std;
using namespace hydra;


GInt_t main(int argv, char** argc)
{


	size_t  nentries       = 0;
	GReal_t mother_mass    = 0;
	GReal_t daughter1_mass = 0;
	GReal_t daughter2_mass = 0;
	GReal_t daughter3_mass = 0;


	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<GULong_t> NArg("n", "number-of-events",
				"Number of events",
				true, 5e6, "long");
		cmd.add(NArg);

		TCLAP::ValueArg<GReal_t> MassMotherArg("m", "mother-mass",
				"Mass of mother particle",
				true, 0.0, "double");
		cmd.add(MassMotherArg);

		TCLAP::ValueArg<GReal_t> MassDaughter1Arg("a", "daughter-a-mass",
				"Mass of daughter particle 'a' (m -> a b c)",
				true, 0.0, "double");
		cmd.add(MassDaughter1Arg);

		TCLAP::ValueArg<GReal_t> MassDaughter2Arg("b", "daughter-b-mass",
				"Mass of daughter particle 'b' (m -> a b c)",
				true, 0.0, "double");
		cmd.add(MassDaughter2Arg);

		TCLAP::ValueArg<GReal_t> MassDaughter3Arg("c", "daughter-c-mass",
				"Mass of daughter particle 'c' (m -> a b c)",
				true, 0.0, "double");
		cmd.add(MassDaughter3Arg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries       = NArg.getValue();
		mother_mass    = MassMotherArg.getValue();
		daughter1_mass = MassDaughter1Arg.getValue();
		daughter2_mass = MassDaughter2Arg.getValue();
		daughter3_mass = MassDaughter3Arg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
														<< std::endl;
	}


	Vector4R B0(mother_mass, 0.0, 0.0, 0.0);
	vector<GReal_t> massesB0{daughter1_mass, daughter2_mass, daughter3_mass };

	/// Create PhaseSpace object for B0-> K pi J/psi
	PhaseSpace<3> phsp(B0.mass(), massesB0);

	Events<3, device> B02JpsiKpi_Events_d(nentries);

	auto start = std::chrono::high_resolution_clock::now();
	phsp.Generate(B0, B02JpsiKpi_Events_d);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	//time
	std::cout << "-----------------------------------------"<<std::endl;
	std::cout << "| Time (s) ="<< GReal_t(elapsed.count())/1000000 <<std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

	for( size_t i=0; i<10; i++ ){
		cout << B02JpsiKpi_Events_d[i] << endl;
	}


	auto Weight = [] __host__ __device__ (Events<3, device>::value_type event )
	{ return thrust::get<0>(event) ; };

	auto MB0 = [] __host__ __device__ (Events<3, device>::value_type event )
	{ return (thrust::get<1>(event) + thrust::get<2>(event) +	thrust::get<3>(event) ).mass();	};

	auto M12 = [] __host__ __device__ ( Events<3, device>::value_type event)
	{ return  (thrust::get<1>(event)+ thrust::get<2>(event)).mass2(); };

	auto M13 = [] __host__ __device__( Events<3, device>::value_type event)
	{ return  (thrust::get<1>(event)+ thrust::get<3>(event)).mass2(); };

	auto M23 = [] __host__ __device__( Events<3, device>::value_type event)
	{ return  (thrust::get<2>(event)+ thrust::get<3>(event)).mass2(); };

	auto Weight_W  = LambdaWrapper<GReal_t( Events<3, device>::value_type),
					decltype(Weight) >(Weight);

	auto MB0_W  = LambdaWrapper<GReal_t( Events<3, device>::value_type),
				decltype(MB0) >(MB0);

	auto M12_W  = LambdaWrapper<GReal_t( Events<3, device>::value_type),
			decltype(M12) >(M12);

	auto M13_W  = LambdaWrapper<GReal_t( Events<3, device>::value_type),
				decltype(M13) >(M13);

	auto M23_W  = LambdaWrapper<GReal_t( Events<3, device>::value_type),
					decltype(M23) >(M23);



	auto functors = thrust::make_tuple(Weight_W, MB0_W, M12_W, M13_W, M23_W);
	auto range_0 =make_range( B02JpsiKpi_Events_d.WeightsBegin(), B02JpsiKpi_Events_d.WeightsEnd());
	auto range_1 =make_range( B02JpsiKpi_Events_d.DaughtersBegin(1), B02JpsiKpi_Events_d.DaughtersBegin(1));
	auto range_2 =make_range( B02JpsiKpi_Events_d.DaughtersBegin(2), B02JpsiKpi_Events_d.DaughtersBegin(2));
	auto range_3 =make_range( B02JpsiKpi_Events_d.DaughtersBegin(3), B02JpsiKpi_Events_d.DaughtersBegin(3));

	auto result_d = Eval( functors, range_0,range_1,range_2,range_3 );

	for( size_t i=0; i<10; i++ ){
			cout << result_d[i] << endl;
	}


	auto result_h = get_copy<host>(result_d);

	for( size_t i=0; i<10; i++ ){
				cout << result_h[i] << endl;
		}


	Events<3, host> B02JpsiKpi_Events_h(B02JpsiKpi_Events_d);

	TH2D dalitz("dalitz", ";M(K#pi); M(J/#Psi#pi)", 100,
			pow(daughter2_mass+daughter3_mass,2), pow(mother_mass - daughter1_mass,2),
			100, 	pow(daughter1_mass+daughter3_mass,2), pow(mother_mass - daughter2_mass,2));



	for(auto event: B02JpsiKpi_Events_h){

		GReal_t weight = thrust::get<0>(event);

		Vector4R Jpsi = thrust::get<1>(event);
		Vector4R K    = thrust::get<2>(event);
		Vector4R pi   = thrust::get<3>(event);

		Vector4R Jpsipi = Jpsi + pi;
		Vector4R Kpi    = K + pi;
		GReal_t mass1 = Kpi.mass();
		GReal_t mass2 = Jpsipi.mass();

		dalitz.Fill(mass1*mass1 , mass2*mass2,  weight);
	}





	TApplication *myapp=new TApplication("myapp",0,0);
	TCanvas canvas_gauss("canvas_gauss", "Gaussian distribution", 500, 500);
	dalitz.Draw("colz");
	canvas_gauss.Print("PHSP.pdf");
	myapp->Run();

return 0;
}
