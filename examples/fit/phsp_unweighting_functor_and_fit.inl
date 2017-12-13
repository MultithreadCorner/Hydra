/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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

#ifndef PHSP_BASIC_INL_
#define PHSP_BASIC_INL_


/**
 * @example phsp_basic.inl
 * This example shows how to use the Hydra's
 * phase space Monte Carlo algorithms to
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
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Decays.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Tuple.h>
#include <hydra/GenericRange.h>
#include <hydra/device/System.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/Distance.h>
#include <hydra/functions/BreitWignerNR.h>

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

	TH2D Dalitz_d1("Dalitz_d", "Weighted Sample;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH2D Dalitz_d2("Dalitz_d2", "Unweighted Sample;M^{2}(J/psi #pi) [GeV^{2}/c^{4}]; M^{2}(K #pi) [GeV^{2}/c^{4}]",
			100, pow(Jpsi_mass + pi_mass,2), pow(B0_mass - K_mass,2),
			100, pow(K_mass + pi_mass,2), pow(B0_mass - Jpsi_mass,2));

	TH1D M23_d("M23_d", "Unweighted Sample;M(K #pi) [GeV/c^{2}];Events",
			100, K_mass + pi_mass, B0_mass - Jpsi_mass);

	TH1D M23FIT_d("M23FIT_d", "Unweighted Sample;M(K #pi) [GeV/c^{2}];Events",
			100, K_mass + pi_mass, B0_mass - Jpsi_mass);

#endif

	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
	double masses[3]{Jpsi_mass, K_mass, pi_mass };

	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp(masses);

	// functor to calculate 2-body masses
	auto masses = hydra::wrap_lambda( []__host__ __device__(unsigned int n, hydra::Vector4R* p ){

		double   m12 = (p[0]+p[1]).mass();
		double   m13 = (p[0]+p[2]).mass();
		double   m23 = (p[1]+p[2]).mass();

		return hydra::make_tuple(m12, m13, m23);
	});


	std::string M0_s("mass");
	hydra::Parameter M0 = hydra::Parameter::Create()
		.Name(M0_s ).Value(0.8).Error(0.001).Limits(0.7, 0.9);

	std::string W0_s("width");
	hydra::Parameter W0 = hydra::Parameter::Create()
		.Name(W0_s ).Value(0.05).Error(0.001).Limits(0.04, 0.06);

	auto breit_wigner = hydra::BreitWignerNR<2>(M0, W0);

	//make model and fcn
	auto model = hydra::make_pdf(breit_wigner, Integrator_d );
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


#ifdef 	_ROOT_AVAILABLE_

		//bring events to CPU memory space
		hydra::Decays<3, hydra::host::sys_t > Events_h( Events_d);

		for( auto event : Events_h ){

			double weight           = hydra::get<0>(event);
			hydra::Vector4R Jpsi  = hydra::get<1>(event);
			hydra::Vector4R K      = hydra::get<2>(event);
			hydra::Vector4R pi     = hydra::get<3>(event);

			double M2_Jpsi_pi = (Jpsi + pi).mass2();
			double M2_Kpi     = (K + pi).mass2();

			Dalitz_d1.Fill( M2_Jpsi_pi, M2_Kpi, weight);
		}

#endif

		//set the mass and width of the breit-wigner to K*(892)0
		breit_wigner.SetParameter(0, 0.89555 );
		breit_wigner.SetParameter(1, 0.04730 );

		//reorder the container match the shape breit-wigner shape
		size_t last    = Events_d.Unweight(breit_wigner, 1.0);

		//make smart range to calculate the fit variables
		auto variables_range = hydra::make_range(Events_d.GetUnweightedDecays().begin(),
				Events_d.GetUnweightedDecays().begin()+last, breit_wigner);


		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "<======= Dataset [ m12, m13, m23] =======>"<< std::endl;
		for( size_t i=0; i<10; i++ )
			std::cout << variables_range.begin()[i] << std::endl;

		// Data fit
		//--------------------------------------------


		auto fcn = hydra::make_loglikehood_fcn( model,
						variables_range.begin(), variables_range.end());

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

		//time
		std::cout << "-----------------------------------------"<< std::endl;
		std::cout << "| GPU Time (ms) = "<< elapsed_d.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		//generate an independent sample to draw plots
		hydra::Decays<3, hydra::device::sys_t > Events2_d(1000000);

		//generate the final state particles
		phsp.SetSeed();
		phsp.Generate(B0, Events2_d.begin(), Events2_d.end());

		//reweight the sample using the fitted function
		Events2_d.Reweight( fcn.GetPDF().GetFunctor() );

		std::cout << "<======= Fitted shape [Weighted] =======>"<< std::endl;
		for( size_t i=0; i<10; i++ )
			std::cout << Events2_d[i] << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		//Dalitz plot and mass
		{
			//bring events to CPU memory space
			hydra::Decays<3, hydra::host::sys_t > Events_Temp_h(  Events_d.begin(),
					Events_d.begin()+ last) ;

			for( auto event : Events_Temp_h ){

				double weight        = hydra::get<0>(event);// always 1
				hydra::Vector4R Jpsi = hydra::get<1>(event);
				hydra::Vector4R K    = hydra::get<2>(event);
				hydra::Vector4R pi   = hydra::get<3>(event);

				double M2_Jpsi_pi = (Jpsi + pi).mass2();
				double M2_Kpi     = (K + pi).mass2();
				double M_Kpi     = (K + pi).mass();

				Dalitz_d2.Fill( M2_Jpsi_pi, M2_Kpi);

				M23_d.Fill( M_Kpi);
			}
		}

		//fitted shape
		{
			hydra::Decays<3, hydra::host::sys_t > Events_Temp_h(  Events2_d.begin(),
					Events2_d.end()) ;

			for( auto event : Events_Temp_h ){

				double weight        = hydra::get<0>(event);
				hydra::Vector4R Jpsi = hydra::get<1>(event);
				hydra::Vector4R K    = hydra::get<2>(event);
				hydra::Vector4R pi   = hydra::get<3>(event);
				double M_Kpi          = (K + pi).mass();

				M23FIT_d.Fill( M_Kpi, weight);
			}
		}
#endif

	}//end device




#ifdef 	_ROOT_AVAILABLE_

	TApplication *m_app=new TApplication("myapp",0,0);

	TCanvas canvas_h1("canvas_h", "Phase-space Host", 500, 500);
	Dalitz_h1.Draw("colz");

	TCanvas canvas_d1("canvas_d1", "Phase-space Device", 500, 500);
	Dalitz_d1.Draw("colz");

	TCanvas canvas_m23_h("canvas_m23_h", "Phase-space Host", 500, 500);
	M23_h.Draw("e0");
	M23FIT_h.Scale(M23_h.Integral()/M23FIT_h.Integral() );
	M23FIT_h.Draw("Chistsame");

	M23FIT_h.SetLineColor(2);

	m_app->Run();

#endif

	return 0;
}


#endif /* PHSP_BASIC_INL_ */
