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
#include <limits>
//command line arguments
#include <tclap/CmdLine.h>

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/VegasState.h>
#include <hydra/Vegas.h>
#include <hydra/Plain.h>
#include <hydra/Parameter.h>


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

	size_t  calls              = 0;
	size_t  iterations         = 0;
	GReal_t max_error          = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<size_t> NCallsArg("n", "number-of-calls",
				"Number of call.",
				true, 5000, "size_t");
		cmd.add(NCallsArg);

		TCLAP::ValueArg<GReal_t> MaxErrorArg("e", "maximum-error",
				"Maximum error.",
				false, 1.0e-3, "double");
		cmd.add(MaxErrorArg);

		TCLAP::ValueArg<size_t> IterationsArg("i", "max-iterations",
				"Maximum maximum number of iterations.",
				false, 10, "size_t");
		cmd.add(IterationsArg);


		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		calls      = NCallsArg.getValue();
		iterations = IterationsArg.getValue();
		max_error  = MaxErrorArg.getValue();

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}


	constexpr size_t N = 10;

	//------------------------------------
	//parameters
	//------------------------------------

	std::string  Mean_s[N];
	std::string Sigma_s[N];
	GUInt_t  Position_p[N];
	Parameter    Mean_p[N];
	Parameter   Sigma_p[N];

    //-------------------------------------------
	//range of the analysis
	std::array<GReal_t, N>  min;
	std::array<GReal_t, N>  max;

	for(size_t i=0; i< N; i++){

		    min[i] = -6.0;
		    max[i] =  6.0;
	 Position_p[i] = i;
		 Mean_s[i] = "mean_"  ;
		 Mean_s[i] += std::to_string(i);
		Sigma_s[i] = "sigma_" ;
		Sigma_s[i] += std::to_string(i);
		 Mean_p[i].Name(Mean_s[i]).Value(0.0) .Error(0.0001).Limits( -5.0, 5.0);
		Sigma_p[i].Name(Sigma_s[i]).Value(1.0) .Error(0.0001).Limits( 0.5, 1.5);
	}

	//----------------------------------------------------------------------
	// create functor
	//------------------------------------

	GaussN<N> Gaussian(Mean_p, Sigma_p, Position_p);

	//----------------------------------------------------------------------
	//get integration
	//Vegas state hold the resources for performing the integration
	VegasState<N> state(min, max);

	state.SetVerbose(-2);
	state.SetAlpha(1.5);
	state.SetIterations( iterations );
	state.SetUseRelativeError(1);
	state.SetMaxError( max_error );
	state.SetCalls( calls );
	state.SetDiscardIterations(2);
	Vegas<N> vegas(state);

	Gaussian.PrintRegisteredParameters();

	//----------------------------------------------------------------------
	//VEGAS
	//----------------------------------------------------------------------
	auto start_vegas = std::chrono::high_resolution_clock::now();
	vegas.Integrate(Gaussian);
	auto end_vegas = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
	cout << ">>> Gaussian intetgral [Vegas]"<< endl;
	cout << "Result: " << vegas.GetState().GetResult()
		 << " +/- "    << vegas.GetState().GetSigma() <<std::endl
		 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

	TH1D Hist_Iterations_Results("Hist_Iterations_Results", "",
			vegas.GetState().GetIterationResult().size(), 0.0,
			vegas.GetState().GetIterationResult().size());

	TH1D Hist_Cumulative_Results("Hist_Cumulative_Results", "",
				vegas.GetState().GetCumulatedResult().size(), 0.0,
				vegas.GetState().GetCumulatedResult().size());

	for(size_t i=1; i<= Hist_Iterations_Results.GetNbinsX(); i++)
	{
		Hist_Cumulative_Results.SetBinContent(i, vegas.GetState().GetCumulatedResult()[i-1]);
		Hist_Cumulative_Results.SetBinError(i, vegas.GetState().GetCumulatedSigma()[i-1]);
		Hist_Iterations_Results.SetBinContent(i, vegas.GetState().GetIterationResult()[i-1]);
		Hist_Iterations_Results.SetBinError(i, vegas.GetState().GetIterationSigma()[i-1]);

	}

	//----------------------------------------------------------------------
	//PLAIN
	//----------------------------------------------------------------------
	Plain<N> plain( min, max, vegas.GetState().GetIterationResult().size()*calls);
	auto start_plain = std::chrono::high_resolution_clock::now();
	plain.Integrate(Gaussian);
	auto end_plain = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_plain = end_plain- start_plain;

	cout << ">>> Gaussian intetgral [Plain]"<< endl;
		cout << "Result: " << plain.GetResult()
			 << " +/- "    << plain.GetSigma() <<std::endl
			 << "Time (ms): "<< elapsed_plain.count() <<std::endl;

	//----------------------------------------------------------------------
	//ANALYTIC
	//----------------------------------------------------------------------
	GaussNAnalyticIntegral<N> gaussianAnaInt(min, max);
	auto result = gaussianAnaInt.Integrate(Gaussian);

	cout << ">>> Gaussian intetgral [Analytic]"<< endl;
	cout << "Result: " << std::setprecision(9)<<result.first
					   << " +/- "    << result.second <<std::endl;


	TApplication *myapp=new TApplication("myapp",0,0);
		/*
	TH1D hist_uniform("uniform", "Initial grid",vegas.GetState().GetNBins(), 0, 1);
	TH1D hist_adapted("adapted", "Adapted  grid", vegas.GetState().GetNBins(), 0, 1);
	hist_adapted.SetBins( vegas.GetState().GetNBins(),  vegas.GetState().GetXi().data() );

	for(size_t i=0; i<state.GetNBins()+1;i++ ){

		std::cout<< vegas.GetState().GetXi().data()[i]<< std::endl;
		hist_uniform.SetBinContent(i, vegas.GetState().GetCallsPerBox());
		hist_adapted.SetBinContent(i, vegas.GetState().GetCallsPerBox());
	}


	canvas.Divide(2,1);
	canvas.cd(1);
	hist_uniform.Draw("bar");
	hist_uniform.SetFillColor(0);
	hist_uniform.SetFillStyle(0);
	canvas.cd(2);
	hist_adapted.Draw("bar");
	hist_adapted.SetFillColor(0);
	hist_adapted.SetFillStyle(0);
	*/
	TCanvas canvas("canvas", "", 1000, 500);
    Hist_Iterations_Results.Draw("E0");
    Hist_Iterations_Results.SetMarkerSize(1);
    Hist_Iterations_Results.SetMarkerStyle(20);
	Hist_Cumulative_Results.Draw("hist same");
	Hist_Cumulative_Results.SetLineColor(kRed);
	Hist_Cumulative_Results.SetLineWidth(2);
	myapp->Run();

	return 0;


	}


