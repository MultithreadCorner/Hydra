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
 * HydraFitExample.cu
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
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/PointVector.h>
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

/**
 * @file
 * @example HydraFitExample.cu
 * @brief HydraFitExample take parameters from the command line, fill a range with random numbers sampled from
 * the model and perform a extended likelihood fit in parallel using the OpenMP backend.
 * @param -c (--combined-minimizer):  Use Migrad + Simplex for minimization
 * @param -i=<double> (--max-iterations=<double>) : Maximum number of iterations for migrad and minimize call.
 * @param -t=<double> (--tolerance=<double>) : Tolerance parameter for migrad and minimize call.
 * @param -n=<long> (--number-of-events=<long>) (required):  Number of events for each component.
 *
 * Usage:
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit  [-c] [-i=<double>]
 *                                      [-t=<double>] -n=<long> [--]
 *                                      [--version] [-h]
 *
 * For example, the command below:
 * ```
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit -n=1000000
 * ```
 * will print some stuff to standard output and produce the plot:
 *
 * @image html Fit_CUDA.png
 */


/**
 * @file
 * @brief HydraFitExample take parameters from the command line, fill a range with random numbers sampled from
 * the model and perform a extended likelihood fit in parallel using the OpenMP backend.
 * @param -c (--combined-minimizer):  Use Migrad + Simplex for minimization
 * @param -i=<double> (--max-iterations=<double>) : Maximum number of iterations for migrad and minimize call.
 * @param -t=<double> (--tolerance=<double>) : Tolerance parameter for migrad and minimize call.
 * @param -n=<long> (--number-of-events=<long>) (required):  Number of events for each component.
 *
 * Usage:
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit  [-c] [-i=<double>]
 *                                      [-t=<double>] -n=<long> [--]
 *                                      [--version] [-h]
 *
 * For example, the command below:
 * ```
 * ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Fit -n=1000000
 * ```
 * will print some stuff to standard output and produce the plot:
 *
 * @image html Fit_CUDA.png
 */
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
				false, 50000, "long int");
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
	ROOT::Minuit2::MnPrint::SetLevel(3);
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
	std::string Mean2("Mean_2"); 	// mean of gaussian 2
	std::string Sigma1("Sigma_1"); 	// sigma of gaussian 1
	std::string Sigma2("Sigma_2"); 	// sigma of gaussian 2
	std::string Tau("Tau"); 		//tau of exponential
	std::string na("Yield_1"); 		//yield #1
	std::string nb("Yield_2"); 		//yield #2
	std::string nc("Yield_3"); 		//yield #3
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
	Parameter  mean1_p  = Parameter::Create().Name(Mean1).Value(3.0) .Error(0.001).Limits( 1.0, 4.0);
	Parameter  sigma1_p = Parameter::Create().Name(Sigma1).Value(0.5).Error(0.001).Limits(0.1, 1.5);
	Parameter  mean2_p  = Parameter::Create().Name(Mean2).Value(5.0).Error(0.001).Limits(4.0, 6.0);
	Parameter  sigma2_p = Parameter::Create().Name(Sigma2).Value(1.0).Error(0.001).Limits(0.5, 1.5);
    Parameter  tau_p    = Parameter::Create().Name(Tau).Value(0).Error(0.001).Limits( -1.0, 1.0);

    // 2) using unnamed parameter idiom
    Parameter NA_p(na ,nentries, sqrt(nentries), nentries - nentries/2, nentries + nentries/2) ;
	Parameter NB_p(nb ,nentries, sqrt(nentries), nentries - nentries/2, nentries + nentries/2) ;
	Parameter NC_p(nc ,nentries, sqrt(nentries), nentries - nentries/2, nentries + nentries/2) ;

	//----------------------------------------------------------------------
    // registry the parameters
	UserParameters upar;

    upar.AddParameter(&mean1_p);
    upar.AddParameter(&sigma1_p);
    upar.AddParameter(&mean2_p);
    upar.AddParameter(&sigma2_p);
    upar.AddParameter(&tau_p);
    upar.AddParameter(&NA_p);
    upar.AddParameter(&NB_p);
    upar.AddParameter(&NC_p);

    //check all is fine
    upar.PrintParameters();

    //----------------------------------------------------------------------
	// create functors
	Gauss Gaussian1(mean1_p, sigma1_p,0);
	Gauss Gaussian2(mean2_p, sigma2_p,0);
	Exp   Exponential(tau_p);

	//----------------------------------------------------------------------
	//get integration
    //Vegas state hold the resources for performing the integration
    VegasState<1> state = VegasState<1>( min, max); // nota bene: the same range of the analisys
	state.SetVerbose(-1);
	state.SetAlpha(1.75);
	state.SetIterations(5);
	state.SetUseRelativeError(1);
	state.SetMaxError(1e-3);

    //5,000 calls (fast convergence and precise result)
	Vegas<1> vegas( state,10000);

	auto Gaussian1_PDF   = make_pdf(Gaussian1, &vegas);
	auto Gaussian2_PDF   = make_pdf(Gaussian2, &vegas);
	auto Exponential_PDF = make_pdf(Exponential, &vegas);

	Gaussian1_PDF.PrintRegisteredParameters();

	//----------------------------------------------------------------------
	//integrate with the current parameters just to test
	vegas.Integrate(Gaussian1_PDF);
	cout << ">>> GaussianA intetgral prior fit "<< endl;
	cout << "Result: " << vegas.GetResult() << " +/- "
		 << vegas.GetAbsError() << " Chi2: "<< vegas.GetState().GetChiSquare() << endl;

	Gaussian2_PDF.PrintRegisteredParameters();

	vegas.Integrate(Gaussian2_PDF);
	cout << ">>> GaussianB intetgral prior fit "<< endl;
	cout << "Result: " << vegas.GetResult() << " +/- "
			<< vegas.GetAbsError() << " Chi2: "<< vegas.GetState().GetChiSquare() << endl;

	Exponential_PDF.PrintRegisteredParameters();

	vegas.Integrate(Exponential_PDF);
	cout << ">>> Exponential intetgral prior fit "<< endl;
	cout << "Result: " << vegas.GetResult() << " +/- "
			<< vegas.GetAbsError() << " Chi2: "<< vegas.GetState().GetChiSquare() << endl;

	//----------------------------------------------------------------------
	//add the pds to make a extended pdf model

	//list of yields
	std::array<Parameter*, 3>  yields{&NA_p, &NB_p, &NC_p};

	auto model = add_pdfs(yields, Gaussian1_PDF, Gaussian2_PDF, Exponential_PDF );

	//--------------------------------------------------------------------
	//Generate data on the device with the original parameters
	//
	PointVector<device, GReal_t, 1> data_d(3*nentries);

	Generator.Gauss(mean1_p , sigma1_p, data_d.begin(), data_d.begin() + nentries );
	Generator.Gauss(mean2_p , sigma2_p, data_d.begin()+ nentries, data_d.begin() + 2*nentries );
	Generator.Uniform(min[0], max[0], data_d.begin()+ 2*nentries, data_d.end() );

	//---------------------------
	//get data from device and fill histogram
	PointVector<host> data_h(data_d);

	TH1D hist_gaussian("gaussian", "", 100, min[0], max[0]);

	for(auto point: data_h )
		hist_gaussian.Fill(point.GetCoordinate(0));

	//-------------------------------------------------
    //minimization

	//get the FCN
	auto modelFCN = make_loglikehood_fcn(model, data_d.begin(), data_d.end() );

    //print minuit parameters before the fit
	std::cout << upar << endl;

    //minimization strategy
	MnStrategy strategy(2);

	// create Migrad minimizer
	MnMigrad migrad(modelFCN, upar.GetState() ,  strategy);

	// create Minimize minimizer
	MnMinimize minimize(modelFCN,upar.GetState() ,  strategy);
	FunctionMinimum *minimum=0;
	// ... Minimize and profile the time
	auto start = std::chrono::high_resolution_clock::now();

	if(use_comb_minimizer){
		 minimum = new FunctionMinimum(minimize(iterations, tolerance));
	}
	else{
		 minimum = new FunctionMinimum(migrad(iterations, tolerance));
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;

	// output
	std::cout<<"minimum: "<<*minimum<<std::endl;

	//time
	std::cout << "-----------------------------------------"<<std::endl;
	std::cout << "| Time (ms) ="<< elapsed.count() <<std::endl;
	std::cout << "-----------------------------------------"<<std::endl;

	//------------------------------------------------------
	//Sampling the fitted model
    //Set the function with the fitted parameters
	model.SetParameters(minimum->UserParameters().Params());
	model.PrintRegisteredParameters();

	//sample fit function on the host nentries trials
	PointVector<host, GReal_t, 1> data2_h(0);
	Generator.SetSeed(std::chrono::system_clock::now().time_since_epoch().count()+1);//+1 because all can run very fast sometimes
	Generator.Sample(model, min, max, data2_h, nentries );

	TH1D hist_gaussian_fit("gaussian_fit", "", 100, min[0], max[0]);

	// histogram it for graphics representation
	for(auto point: data2_h )
			hist_gaussian_fit.Fill(point.GetCoordinate(0));

	TH1D hist_gaussian_plot("gaussian_plot", "", 100,  min[0], max[0]);
	for (size_t i=0 ; i<=100 ; i++) {
		GReal_t x = hist_gaussian_plot.GetBinCenter(i);
		hist_gaussian_plot.SetBinContent(i, model(&x) );
	}

	//scale
	hist_gaussian_fit.Scale(hist_gaussian.Integral()/hist_gaussian_fit.Integral() );
	hist_gaussian_plot.Scale(hist_gaussian.Integral()/hist_gaussian_plot.Integral() );



	/***********************************
	 * RootApp, Drawing...
	 ***********************************/

	TApplication *myapp=new TApplication("myapp",0,0);


	TCanvas canvas_gauss("canvas_gauss", "Gaussian distribution", 500, 500);
	hist_gaussian.Draw("e0");
	hist_gaussian.SetMarkerSize(1);
	hist_gaussian.SetMarkerStyle(20);

	//sampled data after fit
	hist_gaussian_fit.Draw("barSAME");
	hist_gaussian_fit.SetLineColor(4);
	hist_gaussian_fit.SetFillColor(4);
	hist_gaussian_fit.SetFillStyle(3001);

	//original data
	hist_gaussian.Draw("e0SAME");

	//plot
	hist_gaussian_plot.Draw("csame");
	hist_gaussian_plot.SetLineColor(2);
	hist_gaussian_plot.SetLineWidth(2);

	canvas_gauss.SaveAs("./plots/Fit_CUDA.png");

	myapp->Run();

	delete minimum;

	return 0;


}



