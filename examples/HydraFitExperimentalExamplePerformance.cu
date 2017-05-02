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
 * HydraFitExample.cpp
 *
 *  Created on: Jun 21, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>

//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/VegasState.h>
#include <hydra/Vegas.h>

#include <hydra/experimental/LogLikelihoodFCN.h>
#include <hydra/experimental/PointVector.h>

#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>

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

#include <examples/Gauss.h>
#include <examples/Exp.h>

using namespace std;
using namespace ROOT::Minuit2;
using namespace hydra;
using namespace examples;


GInt_t main(int argv, char** argc)
{

	size_t  nentries           = 0;
	size_t  iterations         = 0;
	GReal_t tolerance          = 0;
	GBool_t use_comb_minimizer = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<size_t> NEventsArg("n", "number-of-events",
				"Number of events for each component.",
				true, 1e6, "long int");
		cmd.add(NEventsArg);

		TCLAP::ValueArg<GReal_t> ToleranceArg("t", "tolerance",
				"Tolerance parameter for migrad and minimize call.",
				false, 1.0, "double");
		cmd.add(ToleranceArg);

		TCLAP::ValueArg<size_t> IterationsArg("i", "max-iterations",
				"Maximum number of iterations for migrad and minimize call.",
				false, 50000000000, "long int");
		cmd.add(IterationsArg);

		TCLAP::SwitchArg MinimizeArg("c","combined-minimizer",
				"Use Migrad + Simplex for minimization", false);

		cmd.add(MinimizeArg);
		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries   = NEventsArg.getValue();
		iterations = IterationsArg.getValue();
		tolerance  = ToleranceArg.getValue();
		use_comb_minimizer = MinimizeArg.getValue();

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	//Print::SetLevel(0);
	//ROOT::Minuit2::MnPrint::SetLevel(0);
	//----------------------------------------------

	//Generator with current time count as seed.
	size_t seed = 0;//std::chrono::system_clock::now().time_since_epoch().count();
	Random<thrust::random::default_random_engine> Generator( seed  );


	//-------------------------------------------
	//range of the analysis
	std::array<GReal_t, 1>  min   ={ 0.0};
	std::array<GReal_t, 1>  max   ={ 15.0};

	//------------------------------------
	//parameters names
	std::string Mean1("Mean_1"); 	// mean of gaussian 1
	std::string Sigma1("Sigma_1"); 	// sigma of gaussian 1
	std::string Tau("Tau"); 		//tau of exponential
	std::string na("Yield_1"); 		//yield #1
	std::string nb("Yield_2"); 		//yield #2
		//fit paremeters


	//----------------------------------------------------------------------
	//create parameters
	//	Gaussian #1:
	//	mean = 3.0,	sigma = 0.5, yield = N1_p
	//
	//	Gaussian #2:
	//	mean  = 5.0, sigma = 1.0, yield = N2_p
	//
	//	Exponential:
	//	tau  = 1.0

	// 1) using named parameter idiom
	Parameter  mean1_p  = Parameter::Create().Name(Mean1).Value(3.0) .Error(0.0001).Limits( 1.0, 4.0);
	Parameter  sigma1_p = Parameter::Create().Name(Sigma1).Value(0.5).Error(0.0001).Limits(0.1, 1.5);
	Parameter  tau_p    = Parameter::Create().Name(Tau).Value(0).Error(0.0001).Limits( -1.0, 1.0);

	// 2) using unnamed parameter idiom
	Parameter NA_p(na ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;
	Parameter NB_p(nb ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;

	//----------------------------------------------------------------------
	// registry the parameters
	UserParameters upar;

	upar.AddParameter(&mean1_p);
	upar.AddParameter(&sigma1_p);
	upar.AddParameter(&tau_p);
	upar.AddParameter(&NA_p);
	upar.AddParameter(&NB_p);

	// upar.GetState().SetPrecision( 1.0e-6);
	ROOT::Minuit2::MnPrint::SetLevel(3);

	//check all is fine
	upar.PrintParameters();


	//----------------------------------------------------------------------
	// create functors with different parameters, to be sure fitting is working
	mean1_p  += 0.5;
	sigma1_p += 0.25;
	tau_p    += 0.25;

	Gauss Gaussian1(mean1_p, sigma1_p,0,kFalse);
	Exp   Exponential(tau_p,0);


	typedef hydra::experimental::Point<GReal_t, 1> point_t;

	//--------------------------------------------------------------------
	//Generate data on the device with the original parameters
	hydra::experimental::PointVector<point_t , device> data_d(2*nentries);


	Generator.Gauss(mean1_p , sigma1_p,
			hydra::experimental::GetCoordinateBegin<0>(data_d),
			hydra::experimental::GetCoordinateBegin<0>(data_d) + nentries );

	Generator.Uniform(min[0], max[0],
			hydra::experimental::GetCoordinateBegin<0>(data_d)+ nentries,
			hydra::experimental::GetCoordinateEnd<0>(data_d) );


	//------------------------------------------------------
	//get data from device and fill histogram

	hydra::experimental::PointVector<point_t ,host> data_h(data_d);


	TH1D hist_data("data", ";X;Yield", 100, min[0], max[0]);
	hist_data.Sumw2();
	for(auto point: data_h )
		hist_data.Fill(((point_t)point).GetCoordinate(0));

	//------------------------------------------------------
	//container to sample fit function on the host nentries trials

	hydra::experimental::PointVector<point_t ,host> data_fit_vegas_h(0);

	//------------------------------------------------------
	//sample fit function on the host nentries trials
	PointVector<host, GReal_t, 1> data_fit_analytic_h(0);

	//------------------------------------------------------
	// histogram to plot the fit result
	TH1D hist_fit_analytic_plot("fit_analytic_plot", ";X;Yield", 100,  min[0], max[0]);



	//----------------------------------------
	//fit on device
	{


		GaussAnalyticIntegral GaussIntegral(min[0], max[0]);
		ExpAnalyticIntegral   ExpIntegral(min[0], max[0]);

		auto Gaussian1_PDF   = make_pdf(Gaussian1, GaussIntegral);
		auto Exponential_PDF = make_pdf(Exponential, ExpIntegral);

		//----------------------------------------------------------------------
		//add the pds to make a extended pdf model

		//list of yields
		std::array<Parameter*, 2>  yields{&NA_p, &NB_p};

		auto model = add_pdfs(yields, Gaussian1_PDF, Exponential_PDF );
		model.SetExtended(1);

		//-------------------------------------------------
		//minimization

		//get the FCN
		auto modelFCN_d = hydra::experimental::make_loglikehood_fcn(model, data_d);//.begin(), data_d.end() );
		auto modelFCN_h = hydra::experimental::make_loglikehood_fcn(model, data_h);//.begin(), data_d.end() );

		//print minuit parameters before the fit
		std::cout << upar << endl;

		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_d(modelFCN_d, upar.GetState() ,  strategy);
		MnMigrad migrad_h(modelFCN_h, upar.GetState() ,  strategy);

		FunctionMinimum *minimum_d=0;
		FunctionMinimum *minimum_h=0;

		// ... Minimize and profile the time
		auto start_d = std::chrono::high_resolution_clock::now();
		minimum_d = new FunctionMinimum(migrad_d(std::numeric_limits<unsigned int>::max(), tolerance));
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		// output
		std::cout<<"minimum: "<<*minimum_d<<std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		// ... Minimize and profile the time
		auto start_h = std::chrono::high_resolution_clock::now();
		minimum_h = new FunctionMinimum(migrad_h(std::numeric_limits<unsigned int>::max(), tolerance));
		auto end_h = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_h = end_h - start_h;

		// output
		std::cout<<"minimum: "<<*minimum_h<<std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| CPU Time (ms) ="<< elapsed_h.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;




		//------------------------------------------------------
		//Sampling the fitted model
		//Set the function with the fitted parameters
		model.SetParameters(minimum_d->UserParameters().Params());
		model.PrintRegisteredParameters();



		hist_fit_analytic_plot.Sumw2();
		for (size_t i=0 ; i<=100 ; i++) {
			GReal_t x = hist_fit_analytic_plot.GetBinCenter(i);
			hist_fit_analytic_plot.SetBinContent(i, model.GetFunctor()( &x) );
		}

		//scale
		hist_fit_analytic_plot.Scale(hist_data.Integral()/hist_fit_analytic_plot.Integral() );
        delete minimum_h;
        delete minimum_d;

	}



	TApplication *myapp=new TApplication("myapp",0,0);


	TCanvas canvas_analytic("canvas_analytic", "ANALYTIC", 500, 500);
	hist_data.Draw("e0");
	hist_data.SetMarkerSize(1);
	hist_data.SetMarkerStyle(20);
	hist_data.GetYaxis()->SetTitleOffset(1.5);
	//sampled data after fit
	//hist_fit_analytic.Draw("barSAME");
	//hist_fit_analytic.SetLineColor(4);
	//hist_fit_analytic.SetFillColor(4);
	//hist_fit_analytic.SetFillStyle(3001);

	//original data
	hist_data.Draw("e0SAME");

	//plot
	hist_fit_analytic_plot.Draw("csame");
	hist_fit_analytic_plot.SetLineColor(2);
	hist_fit_analytic_plot.SetLineWidth(2);
	hist_fit_analytic_plot.GetYaxis()->SetTitleOffset(1.5);

	canvas_analytic.SaveAs("./plots/Fit_perfomance.png");
	canvas_analytic.SaveAs("./plots/Fit_perfomance.pdf");


	myapp->Run();



	return 0;


}





