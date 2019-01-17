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
 * phsp_basic.inl
 *
 *  Created on: Jul 7, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_UNWEIGHTING_FUNCTOR_AND_FIT_INL_
#define PHSP_UNWEIGHTING_FUNCTOR_AND_FIT_INL_


/**
 * \example phsp_basic.inl
 *
 * This example shows how to use the Hydra's
 * phase space Monte Carlo generation and integration algorithms to
 * generate a sample of B0 -> J/psi K pi and
 * unweights to produce a B0-> K*(892) J/psi and fits a Breit-Wigner shape.
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
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Decays.h>
#include <hydra/PhaseSpace.h>
#include <hydra/PhaseSpaceIntegrator.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>

#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/Distance.h>
#include <hydra/functions/BreitWignerNR.h>
#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */


//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/MnPlot.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include "Minuit2/VariableMetricMinimizer.h"

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

using namespace ROOT::Minuit2;

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
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
																<< std::endl;
	}

#ifdef 	_ROOT_AVAILABLE_
    //__________________________________________________
	//device

	TH3D Dalitz_FLAT("Dalitz_Flat", "Flat Dalitz;M^{2}(J/psi K) [GeV^{2}/c^{4}];M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

	TH3D Dalitz_BW("Dalitz_BW", "Breit-Wigner Dalitz;M^{2}(J/psi K) [GeV^{2}/c^{4}];M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

	TH3D Dalitz_FIT("Dalitz_FIT", "Fit result;M^{2}(J/psi K) [GeV^{2}/c^{4}];M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

#endif

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp({Jpsi_mass, K_mass, pi_mass });

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[] __hydra_dual__ (unsigned int n, hydra::Vector4R* p ){

			double   MSq_12 = (p[0]+p[1]).mass2();
			double   MSq_13 = (p[0]+p[2]).mass2();
			double   MSq_23 = (p[1]+p[2]).mass2();

			return hydra::make_tuple(MSq_12, MSq_13, MSq_23);
	});


	// fit model
	hydra::Parameter M0 = hydra::Parameter::Create()
		.Name("mass" ).Value(0.8).Error(0.001).Limits(0.7, 0.9);

	std::string W0_s("width");
	hydra::Parameter W0 = hydra::Parameter::Create()
		.Name("width").Value(0.05).Error(0.001).Limits(0.04, 0.06);


	// fit functor (breit-wigner)
	auto breit_wigner = hydra::wrap_lambda( [] __hydra_dual__ (unsigned int npar, const hydra::Parameter* params,
					unsigned int n, hydra::Vector4R* particles ){

		double mass  = params[0];
		double width = params[1];

		auto   p0  = particles[0] ;
		auto   p1  = particles[1] ;
		auto   p2  = particles[2] ;

		auto   m12 = (p1+p2).mass();

		double dmass2 = (m12-mass)*(m12-mass);
		double width2   = width*width;

		double denominator = dmass2 + width2/4.0;
		return ((width2)/4.0)/denominator;

	}, M0, W0 );


	auto model = hydra::make_pdf( breit_wigner,
			hydra::PhaseSpaceIntegrator<3, hydra::device::sys_t>(B0_mass, {Jpsi_mass, K_mass, pi_mass }, 1000000));

	//scoped calculations to save memory
	{

		//allocate memory to hold the final states particles
		hydra::Decays<3, hydra::device::sys_t > Events_d(nentries);

		//generate events
		auto start = std::chrono::high_resolution_clock::now();
		//generate the final state particles
		phsp.Generate(B0, Events_d.begin(), Events_d.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;
		std::cout << "| B0 -> J/psi K pi"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;


		std::cout << "<======= Flat [Weighted sample] =======>"<< std::endl;
		for( size_t i=0; i<10; i++ )
			std::cout << Events_d[i] << std::endl;


		auto particles        = Events_d.GetUnweightedDecays();
		auto dalitz_variables = hydra::make_range( particles.begin(), particles.end(), dalitz_calculator);
		auto dalitz_weights   = Events_d.GetWeights();

		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "<======= Dataset w : ( MSq_12, MSq_13, MSq_23) =======>"<< std::endl;
		for( size_t i=0; i<10; i++ )
			std::cout << dalitz_weights[i] << " : "<< dalitz_variables[i] << std::endl;

		//flat dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Flat_Dalitz{
				{100,100,100},
				{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
				{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_Flat_Dalitz.Fill( dalitz_variables.begin(),
				dalitz_variables.end(), dalitz_weights.begin()  );

		std::cout << "<======= Flat Dalitz plot=======>"<< std::endl;
				for( size_t i=0; i<10; i++ )
					std::cout << Hist_Flat_Dalitz[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Flat_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{
							size_t bin     = hydra::get<0>(entry);
							double content = hydra::get<1>(entry);
							unsigned int bins[3];
							Hist_Temp.GetIndexes(bin, bins);
							Dalitz_FLAT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_Flat_Dalitz)
					{
						size_t bin     = hydra::get<0>(entry);
						double content = hydra::get<1>(entry);
						unsigned int bins[3];
						Hist_Flat_Dalitz.GetIndexes(bin, bins);
						Dalitz_FLAT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

					}
	#endif

#endif


		//set the mass and width of the breit-wigner to K*(892)0
		breit_wigner.SetParameter(0, 0.89555 );
		breit_wigner.SetParameter(1, 0.04730 );

		//reorder the container match the shape breit-wigner shape
		size_t last = Events_d.Unweight(breit_wigner, 1.0);

		//breit-wigner weighted dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_BW_Dalitz{
			{100,100,100},
			{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
			{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_BW_Dalitz.Fill( dalitz_variables.begin(),  dalitz_variables.begin()+ last);

#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_BW_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{
							size_t bin     = hydra::get<0>(entry);
							double content = hydra::get<1>(entry);
							unsigned int bins[3];
							Hist_Temp.GetIndexes(bin, bins);
							Dalitz_BW.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_BW_Dalitz)
					{
						size_t bin     = hydra::get<0>(entry);
						double content = hydra::get<1>(entry);
						unsigned int bins[3];
						Hist_BW_Dalitz.GetIndexes(bin, bins);
						Dalitz_BW.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

					}
	#endif

#endif

		// Data fit
		//--------------------------------------------


		auto fcn = hydra::make_loglikehood_fcn( model, particles.begin(),
				particles.begin() + last);

		//print level
		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);

		//minimization strategy
		MnStrategy strategy(2);

		//create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		//print parameters before fitting
		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		//Minimize and profile the time
		auto start_d = std::chrono::high_resolution_clock::now();

		FunctionMinimum minimum_d =  FunctionMinimum( migrad_d(50000, 5) );

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//print parameters after fitting
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		//generate an idependent sample for plotting
		//allocate memory to hold the final states particles
		hydra::Decays<3, hydra::device::sys_t > DisplayEvents(nentries);
		phsp.SetSeed( std::chrono::system_clock::now().time_since_epoch().count() );
		phsp.Generate(B0, DisplayEvents.begin(), DisplayEvents.end());
		DisplayEvents.Reweight( fcn.GetPDF().GetFunctor() );

		auto fitted_particles        = DisplayEvents.GetUnweightedDecays();
		auto fitted_dalitz_variables = hydra::make_range( fitted_particles.begin(), fitted_particles.end(), dalitz_calculator);
		auto fitted_dalitz_weights   = DisplayEvents.GetWeights();

		//fitted dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Fit_Dalitz{
			{100,100,100},
			{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
			{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_Fit_Dalitz.Fill( fitted_dalitz_variables.begin(),
				fitted_dalitz_variables.end(), fitted_dalitz_weights.begin()  );

#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Fit_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{
							size_t bin     = hydra::get<0>(entry);
							double content = hydra::get<1>(entry);
							unsigned int bins[3];
							Hist_Temp.GetIndexes(bin, bins);
							Dalitz_FIT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_Fit_Dalitz)
					{
						size_t bin     = hydra::get<0>(entry);
						double content = hydra::get<1>(entry);
						unsigned int bins[3];
						Hist_Fit_Dalitz.GetIndexes(bin, bins);
						Dalitz_FIT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

					}
	#endif

#endif

	}//end device

//normalize all histograms to Dalitz_BW

	Dalitz_FIT.Scale(Dalitz_FIT.Integral()/Dalitz_BW.Integral() );
	Dalitz_FLAT.Scale(Dalitz_FLAT.Integral()/Dalitz_BW.Integral() );


#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	TCanvas canvas_1("canvas_1", "Phase-space FLAT", 500, 500);
	Dalitz_FLAT.Project3D("yz")->Draw("colz");

	TCanvas canvas_2("canvas_2", "Phase-space Breit-Wigner", 500, 500);
	Dalitz_BW.Project3D("yz")->Draw("colz");

	TCanvas canvas_3("canvas_3", "Phase-space Breit-Wigner fit", 500, 500);
	Dalitz_FIT.Project3D("yz")->Draw("colz");

	TH1* proj=0;
	TCanvas canvas_4("canvas_projections", "Phase-space Breit-Wigner", 1500, 500);
	canvas_4.Divide(3,1);
	canvas_4.cd(1);
	proj = Dalitz_FLAT.Project3D("x")->DrawNormalized("CLHIST");
	proj->SetLineColor(kRed);
	proj =  Dalitz_BW.Project3D("x")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FIT.Project3D("x")->DrawNormalized("CLHISTsame");
	proj->SetLineColor(kBlue);

	canvas_4.cd(2);
	proj = Dalitz_FLAT.Project3D("y")->DrawNormalized("CLHIST");
	proj->SetLineColor(kRed);
	proj =  Dalitz_BW.Project3D("y")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FIT.Project3D("y")->DrawNormalized("CLHISTsame");
	proj->SetLineColor(kBlue);

	canvas_4.cd(3);
	proj = Dalitz_FIT.Project3D("z")->DrawNormalized("CLHIST");
	proj->SetLineColor(kBlue);
	proj =  Dalitz_BW.Project3D("z")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FLAT.Project3D("z")->DrawNormalized("CLHISTsame");
	proj->SetLineColor(kRed);

	m_app->Run();

#endif

	return 0;
}


#endif /* UNWEIGHTING_FUNCTOR_AND_FIT_INL_ */
