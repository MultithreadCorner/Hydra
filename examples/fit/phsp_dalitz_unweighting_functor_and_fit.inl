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
 * phsp_dalitz_unweighting_functor_and_fit.inl
 *
 *  Created on: 18/01/2021
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PHSP_DALITZ_UNWEIGHTING_FUNCTOR_AND_FIT_INL_
#define PHSP_DALITZ_UNWEIGHTING_FUNCTOR_AND_FIT_INL_


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
#include <hydra/DalitzPhaseSpace.h>
#include <hydra/DalitzIntegrator.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Lambda.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/functions/BreitWignerNR.h>
#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>
#include <hydra/Placeholders.h>
#include <hydra/multivector.h>
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
using namespace hydra::arguments;
using namespace hydra::placeholders;
//---------------------------
// Daughter particles
declarg( Weight, double)
declarg(  M12Sq, double)
declarg(  M13Sq, double)
declarg(  M23Sq, double)

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
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
																<< std::endl;
	}

#ifdef 	_ROOT_AVAILABLE_
    //__________________________________________________
	//device

	TH3D Dalitz_FLAT("Dalitz_Flat",
			"Flat Dalitz;"
			"M^{2}(J/psi K) [GeV^{2}/c^{4}];"
			"M^{2}(J/psi #pi) [GeV^{2}/c^{4}];"
			"M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

	TH3D Dalitz_BW("Dalitz_BW",
			"Breit-Wigner Dalitz;"
			"M^{2}(J/psi K) [GeV^{2}/c^{4}];"
			"M^{2}(J/psi #pi) [GeV^{2}/c^{4}];"
			" M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

	TH3D Dalitz_FIT("Dalitz_FIT",
			"Fit result;"
			"M^{2}(J/psi K) [GeV^{2}/c^{4}];"
			"M^{2}(J/psi #pi) [GeV^{2}/c^{4}];"
			"M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + K_mass,2) , pow(B0_mass - pi_mass,2),
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2)   , pow(B0_mass - Jpsi_mass,2));

#endif

	typedef hydra::tuple< Weight, M12Sq,M13Sq,M23Sq> event_type;

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::DalitzPhaseSpace<> phsp(B0_mass, {Jpsi_mass, K_mass, pi_mass });

	// fit model
	hydra::Parameter M0 = hydra::Parameter::Create()
		.Name("mass" ).Value(0.8).Error(0.001).Limits(0.7, 0.9);

	hydra::Parameter W0 = hydra::Parameter::Create()
		.Name("width").Value(0.05).Error(0.001).Limits(0.04, 0.06);


	// fit functor (breit-wigner)
	auto breit_wigner = hydra::wrap_lambda(
			[] __hydra_dual__ (unsigned int npar, const hydra::Parameter* params, M12Sq  m12_sq, M13Sq  m13_sq, M23Sq  m23_sq){

		double mass  = params[0];
		double width = params[1];

		auto   m = ::sqrt(m23_sq);

		double dmass2 = (m-mass)*(m-mass);
		double width2   = width*width;

		double denominator = dmass2 + width2/4.0;
		return ((width2)/4.0)/denominator;

	}, M0, W0 );


	auto model = hydra::make_pdf( breit_wigner,
			hydra::DalitzIntegrator<hydra::device::sys_t>(B0_mass, {Jpsi_mass, K_mass, pi_mass }, 1000000));

	//scoped calculations to save memory
	{

		//allocate memory to hold the final states particles
		hydra::multivector<event_type, hydra::device::sys_t> Events(nentries);

		//generate events
		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(Events, breit_wigner);

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
			std::cout << hydra::columns(Events, _1, _2, _3)[i] << std::endl;

		std::cout << std::endl<< std::endl;

	   //flat dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Flat_Dalitz{
				{100,100,100},
				{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
				{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_Flat_Dalitz.Fill( hydra::columns(Events, _1, _2, _3), hydra::columns(Events, _0) );


#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Flat_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{
							double content = hydra::get<1>(entry);
							unsigned int bins[3];

							Hist_Temp.GetIndexes(  hydra::get<0>(entry), bins);

							Dalitz_FLAT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_Flat_Dalitz)
					{
						double content = hydra::get<1>(entry);
						unsigned int bins[3];

						Hist_Flat_Dalitz.GetIndexes(hydra::get<0>(entry) , bins);
						Dalitz_FLAT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

					}
	#endif

#endif


		//set the mass and width of the breit-wigner to K*(892)0
		breit_wigner.SetParameter(0, 0.89555 );
		breit_wigner.SetParameter(1, 0.04730 );

		//reorder the container match the shape breit-wigner shape
		auto breit_wigner_sample   = hydra::unweight(Events, Events.column(_0) , -1.0, 0xabc123);

		std::cout << std::endl<< std::endl;

		std::cout << "<======= Dataset fit : ( MSq_12, MSq_13, MSq_23) =======>"<< std::endl;

		for( size_t i=0; i<10; i++ )
			std::cout << breit_wigner_sample[i] << std::endl;

		//breit-wigner weighted dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_BW_Dalitz{
			{100,100,100},
			{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
			{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_BW_Dalitz.Fill( hydra::columns( breit_wigner_sample, _1,_2, _3) );

#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_BW_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{
							double content = hydra::get<1>(entry);
							unsigned int bins[3];

							Hist_Temp.GetIndexes(hydra::get<0>(entry), bins);
							Dalitz_BW.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_BW_Dalitz)
					{
						double content = hydra::get<1>(entry);
						unsigned int bins[3];

						Hist_BW_Dalitz.GetIndexes(hydra::get<0>(entry) , bins);
						Dalitz_BW.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

					}
	#endif

#endif

		// Data fit
		//--------------------------------------------


		auto fcn = hydra::make_loglikehood_fcn( model,
		                   hydra::columns( breit_wigner_sample, _1,_2, _3) );

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

		fcn.GetParameters().UpdateParameters(minimum_d);

		//generate an independent sample for plotting
		auto event_display_range  = hydra::dalitz_range(B0_mass, {Jpsi_mass, K_mass, pi_mass },
				0x75456, 1000000, fcn.GetPDF().GetFunctor());


		auto breit_wigner_weights = hydra::columns(event_display_range	, _0) ;

		auto breit_wigner_events  =	hydra::columns(event_display_range , _1,_2, _3);

		//for(auto e: breit_wigner_weights)
			//std::cout<< e <<std::endl;

		//fitted dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Fit_Dalitz{
			{100,100,100},
			{pow(Jpsi_mass + K_mass,2), pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
			{pow(B0_mass - pi_mass,2), pow(B0_mass - K_mass ,2), pow(B0_mass - Jpsi_mass,2)}
		};

		Hist_Fit_Dalitz.Fill( breit_wigner_events, breit_wigner_weights );

#ifdef 	_ROOT_AVAILABLE_

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

					//if device is cuda, bring the histogram data to the host
					//to fill the ROOT histogram faster
					{
						hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Fit_Dalitz);
						std::cout << "Filling a ROOT Histogram... " << std::endl;

						for(auto entry : Hist_Temp)
						{

							double content = hydra::get<1>(entry);
							unsigned int bins[3];

							Hist_Temp.GetIndexes( hydra::get<0>(entry) , bins);
							Dalitz_FIT.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

						}
					}
	#else
					std::cout << "Filling a ROOT Histogram... " << std::endl;

					for(auto entry : Hist_Fit_Dalitz)
					{
						double content = hydra::get<1>(entry);
						unsigned int bins[3];

						Hist_Fit_Dalitz.GetIndexes( hydra::get<0>(entry), bins);
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
	proj = Dalitz_FLAT.Project3D("x")->DrawNormalized("CHIST");
	proj->SetLineColor(kRed);
	proj =  Dalitz_BW.Project3D("x")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FIT.Project3D("x")->DrawNormalized("CHISTsame");
	proj->SetLineColor(kBlue);

	canvas_4.cd(2);
	proj = Dalitz_FLAT.Project3D("y")->DrawNormalized("CHIST");
	proj->SetLineColor(kRed);
	proj =  Dalitz_BW.Project3D("y")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FIT.Project3D("y")->DrawNormalized("CHISTsame");
	proj->SetLineColor(kBlue);

	canvas_4.cd(3);
	proj = Dalitz_FIT.Project3D("z")->DrawNormalized("CHIST");
	proj->SetLineColor(kBlue);
	proj =  Dalitz_BW.Project3D("z")->DrawNormalized("E0same");
	proj->SetMarkerSize(1.0);
	proj->SetMarkerStyle(20);
	proj =  Dalitz_FLAT.Project3D("z")->DrawNormalized("CHISTsame");
	proj->SetLineColor(kRed);

	m_app->Run();

#endif

	return 0;
}




#endif /* PHSP_DALITZ_UNWEIGHTING_FUNCTOR_AND_FIT_INL_ */
