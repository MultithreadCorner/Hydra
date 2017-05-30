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

//OpenMP
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include <omp.h>
#include <thread>
#endif
//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Events.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Copy.h>
#include <hydra/Parameter.h>
//root
#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>
#include <TLegend.h>
#include <thrust/system/cuda/vector.h>
using namespace std;
using namespace hydra;



GInt_t main(int argv, char** argc)
{


	size_t  nentries_start = 0;
	size_t  nentries_delta = 0;
	GUInt_t  npoints        = 0;
	GReal_t mother_mass    = 0;
	GReal_t daughter1_mass = 0;
	GReal_t daughter2_mass = 0;
	GReal_t daughter3_mass = 0;


	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<GULong_t> NStartArg("n", "number-of-events",
				"Initial number of events",
				true, 1e6, "long");
		cmd.add(NStartArg);

		TCLAP::ValueArg<GULong_t> NDeltaArg("s", "step-size",
				"Step size",
				true, 5e5, "long");
		cmd.add(NDeltaArg);

		TCLAP::ValueArg<GUInt_t> NPointsArg("p", "number-of-points",
				"Number of points",
				true, 20, "unsigned int");
		cmd.add(NPointsArg);


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




		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries_start = NStartArg.getValue();
		nentries_delta = NDeltaArg.getValue();
		npoints        = NPointsArg.getValue();
		mother_mass    = MassMotherArg.getValue();
		daughter1_mass = MassDaughter1Arg.getValue();
		daughter2_mass = MassDaughter2Arg.getValue();
		daughter3_mass = MassDaughter3Arg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error()
				  << " for arg " << e.argId()
				  << std::endl;
	}

	TString backend;

#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
	backend =TString("OpenMP");
#elif THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_TBB
	backend =TString("TBB");
#elif THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_CPP
	backend =TString("CPP");
#else
	backend =TString("?");
#endif

	Vector4R P(mother_mass, 0.0, 0.0, 0.0);
	GReal_t masses[3]{daughter1_mass, daughter2_mass, daughter3_mass };

	// Create PhaseSpace object for P-> A B C
	PhaseSpace<3> P2ABC_PHSP(mother_mass, masses);

	 auto mass = [] __host__ __device__ (size_t npars, const Parameter* pars, Vector4R* particles )
	    {
	    	auto   p0  = particles[0] ;
	    	auto   p1  = particles[1] ;
	    	auto   p2  = particles[2] ;

	    	auto   p = p1+p2+p0;

	    	return p.mass();
	    };
	    auto mass2 = [] __host__ __device__ (Vector4R* particles )
	       {
	       	auto   p0  = particles[0] ;
	       	auto   p1  = particles[1] ;
	       	auto   p2  = particles[2] ;

	       	auto   p = p1+p2+p0;

	       	return p.mass();
	       };


	    std::string Mean1("Mean_1"); 	// mean of gaussian 1
	    std::string Sigma1("Sigma_1"); 	// sigma of gaussian 1
	    Parameter  mean1_p  = Parameter::Create().Name(Mean1).Value(3.0) .Error(0.0001).Limits( 1.0, 4.0);
	    Parameter  sigma1_p = Parameter::Create().Name(Sigma1).Value(0.5).Error(0.0001).Limits(0.1, 1.5);

	   auto Mass = wrap_lambda(mass,  mean1_p, sigma1_p );
	   auto Mass2 = wrap_lambda(mass2);

	   Mass.PrintRegisteredParameters();
	   Mass2.PrintRegisteredParameters();

   auto result = P2ABC_PHSP.AverageOn(P , Mass, 10000);

   return 0;
	//----------------------------
	// Device P-> A B C
	//---------------------------
	TH1D 	*DeviceTiming= new TH1D( "DeviceTiming", ";Number of events;Duration [ms]",
			npoints , nentries_start, nentries_start  + npoints*nentries_delta );
	{ // device section

		Events<3, device> P2ABC_Events_d(0);

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| CUDA P -> A B C                       |"<<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;
		for( GUInt_t i=1; i<npoints+1 ; i++ )
		{
			size_t nentries= nentries_start  + (i-1)*nentries_delta;

			P2ABC_Events_d.resize(nentries);

			auto start = std::chrono::high_resolution_clock::now();
			P2ABC_PHSP.Generate(P, P2ABC_Events_d.begin(), P2ABC_Events_d.end());
			cudaDeviceSynchronize();
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> elapsed = end - start;

			double time = elapsed.count();

			std::cout << std::endl
					  <<"|>> Number of events "<< nentries
					  <<" duration (ms) "  << time
					  << std::endl;

			for( size_t j=0; j<10; j++ ){
				cout << P2ABC_Events_d[j] << endl;
			}

			DeviceTiming->SetBinContent(i, time);

		}

	} //end device section


	//----------------------------
	// Host P-> A B C
	//---------------------------
	TH1D 	*HostTiming = new TH1D( "HostTiming", ";Number of events;Duration [ms]",
			npoints , nentries_start, nentries_start  + npoints*nentries_delta );
	{ // host section

		Events<3, host> P2ABC_Events_h(0);

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| OpenMP - TBB P -> A B C                       |"<<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;
		for( GUInt_t i=1; i<npoints+1 ; i++ )
		{
			size_t nentries= nentries_start  + (i-1)*nentries_delta;

			P2ABC_Events_h.resize(nentries);

			auto start = std::chrono::high_resolution_clock::now();
			P2ABC_PHSP.Generate(P, P2ABC_Events_h.begin(), P2ABC_Events_h.end());
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> elapsed = end - start;

			double time = elapsed.count();

			std::cout << std::endl
					<<"|>> Number of events "<< nentries
					<<" duration (ms) "  << time
					<< std::endl;

			for( size_t j=0; j<10; j++ ){
				cout << P2ABC_Events_h[j] << endl;
			}

			HostTiming->SetBinContent(i, time);

		}

	} //end host section


	TH1D 	*SpeedUp = new TH1D("SpeedUp", ";Number of events;Speed-up GPU vs CPU", npoints , nentries_start, nentries_start  + npoints*nentries_delta );

	for( size_t j=1;  j < npoints+1 ; j++ )
	{
		SpeedUp->SetBinContent(j,
				HostTiming->GetBinContent(j)/DeviceTiming->GetBinContent(j) );
	}

#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
	unsigned int  nthreads =  std::thread::hardware_concurrency() ;

		TH1D 	*HostOpenMPTiming = new TH1D( "HostOpenMPTiming",
				";Number of OpenMP threads;Duration [ms]" ,  nthreads,  1,  nthreads+1);

	{

		omp_set_dynamic(0); //disable dynamic teams

		Events<3, host> P2ABC_Events_h( nentries_start );

		for(unsigned int nt=1; nt<nthreads+1; nt++)
		{
			omp_set_num_threads(nt);

			auto start = std::chrono::high_resolution_clock::now();

			P2ABC_PHSP.Generate(P, P2ABC_Events_h.begin(), P2ABC_Events_h.end());

			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> elapsed = end - start;

			double time = elapsed.count();

			std::cout << std::endl
					<<"|>> Number of threads "<< nt
					<<" duration (ms) "               << time
					<< std::endl;

			for( size_t j=0; j<10; j++ ){
				cout << P2ABC_Events_h[j] << endl;
			}

			HostOpenMPTiming->SetBinContent(nt, time);
		}
	}
#endif

	TApplication* myapp=new TApplication("myapp",0,0);

	TCanvas* Canvas_PHSP_SpeedUp = new TCanvas("Canvas_PHSP_SpeedUp", "", 500, 500);

	//---------------------------------------------------------------
	// Pad 1
	//---------------------------------------------------------------
    TPad *pad1 = new TPad("pad1","",0,0,1,1);
	pad1->Draw();
	pad1->cd();
	pad1->SetLogy();
	 //host
	//HostTiming->SetDrawOption("ACP");
	HostTiming->SetName("HostTiming");
	//HostTiming->SetTitle("CPU (single thread)");
	HostTiming->SetMarkerStyle(20);
	HostTiming->SetMarkerColor(kRed);
	HostTiming->SetLineColor(kRed);
	HostTiming->SetLineWidth(2);
	HostTiming->SetFillStyle(0);
	HostTiming->GetXaxis()->CenterLabels();
	HostTiming->GetYaxis()->SetTitleOffset(1.4);
	HostTiming->SetStats(0);
	HostTiming->SetMinimum( 0.5*DeviceTiming->GetMinimum());
	HostTiming->Draw("LP");

    //device
	//DeviceTiming->SetDrawOption("LP");
	DeviceTiming->SetName("DeviceTiming");
	//DeviceTiming->SetTitle("CUDA");
	DeviceTiming->SetMarkerStyle(20);
	DeviceTiming->SetMarkerColor(kBlue);
	DeviceTiming->SetLineColor(kBlue);
	DeviceTiming->SetLineWidth(2);
	DeviceTiming->SetFillStyle(0);
	DeviceTiming->GetXaxis()->CenterLabels();
	DeviceTiming->GetYaxis()->SetTitleOffset(1.4);
	DeviceTiming->SetStats(0);
	DeviceTiming->Draw("LPsame");

	//pad1->BuildLegend();
	pad1->Update();


	TPad *pad2 = new TPad("pad2","",0,0,1,1);
	pad2->SetFillStyle(4000); //transparent pad
	pad2->SetFillColor(0); //transparent pad
	pad2->SetFrameFillStyle(4000);
	pad2->SetFrameFillColor(0);
	pad2->Draw();
	pad2->cd();
	SpeedUp->SetLineColor(kViolet);
	SpeedUp->SetLineWidth(2);
	SpeedUp->SetFillStyle(4000);
	SpeedUp->SetFillColor(0);
	SpeedUp->GetXaxis()->CenterLabels();
	SpeedUp->GetYaxis()->SetTitleOffset(1.4);
	SpeedUp->SetMinimum(1);
	SpeedUp->SetStats(0);
	SpeedUp->Draw("Y+][sames");
	Canvas_PHSP_SpeedUp->Update();

	TLegend* legend = new TLegend(0.65,0.15,0.85,0.30);
	legend->AddEntry(DeviceTiming,"GPU","lp");
	legend->AddEntry(HostTiming, "CPU","lp");
	legend->AddEntry(SpeedUp,"speed-up","lp");

	legend->Draw();

	Canvas_PHSP_SpeedUp->Print(TString::Format("plots/PHSP_GPU_vs_%s_SpeedUp.png", backend.Data() ));
	Canvas_PHSP_SpeedUp->Print(TString::Format("plots/PHSP_GPU_vs_%s_SpeedUp.pdf", backend.Data() ));
	Canvas_PHSP_SpeedUp->Print(TString::Format("plots/PHSP_GPU_vs_%s_SpeedUp.C", backend.Data() ));


#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP

	TCanvas* Canvas_PHSP_OpenMP = new TCanvas("Canvas_PHSP_OpenMP", "", 500, 500);
	HostOpenMPTiming->SetMarkerStyle(20);
	HostOpenMPTiming->SetMarkerColor(kRed);
	HostOpenMPTiming->SetLineColor(kRed);
	HostOpenMPTiming->SetLineWidth(2);
	HostOpenMPTiming->SetFillStyle(0);
	HostOpenMPTiming->GetXaxis()->CenterLabels();
	HostOpenMPTiming->GetYaxis()->SetTitleOffset(1.4);
	HostOpenMPTiming->SetStats(0);
	//HostOpenMPTiming->SetMinimum( 0.5*DeviceTiming->GetMinimum());
	HostOpenMPTiming->Draw("LP");

	Canvas_PHSP_OpenMP->Print(TString::Format("plots/PHSP_%s_Thread_SpeedUp.png", backend.Data() ));
	Canvas_PHSP_OpenMP->Print(TString::Format("plots/PHSP_%s_Thread_SpeedUp.pdf", backend.Data() ));
	Canvas_PHSP_OpenMP->Print(TString::Format("plots/PHSP_%s_Thread_SpeedUp.C", backend.Data() ));


#endif
myapp->Run();

    return 0;
}
