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

//OpenMP
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include <omp.h>
#include <thread>
#endif

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
#include <hydra/GaussKronrodQuadrature.h>
#include <hydra/GaussKronrodAdaptiveQuadrature.h>
#include <hydra/GenzMalikRule.h>
#include <hydra/GenzMalikQuadrature.h>
//root
#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TString.h>
#include <TRatioPlot.h>
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
	std::array<GReal_t, N>  _min;
	std::array<GReal_t, N>  _max;

	for(size_t i=0; i< N; i++){

		    _min[i] = -6.0;
		    _max[i] =  6.0;
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

	GaussN<N> Gaussian(Mean_p, Sigma_p, Position_p, 1);

	//----------------------------------------------------------------------
	//get integration
	//Vegas state hold the resources for performing the integration
	VegasState<N, device> state(_min, _max);

	state.SetVerbose(-2);
	state.SetAlpha(1.5);
	state.SetIterations( iterations );
	state.SetUseRelativeError(1);
	state.SetMaxError( max_error );
	state.SetCalls( calls );
	state.SetTrainingCalls( calls/10 );
	state.SetTrainingIterations(0);
	Vegas<N, device> vegas(state);

	Gaussian.PrintRegisteredParameters();

	vector<TH1D> Cumulative_Results;
	vector<TH1D> Iterations_Results;
	vector<TH1D> Iterations_Duration;
	vector<TH1D> FunctionCalls_Duration;
	vector<TH1D> Duration_Problem_Size;
	//----------------------------------------------------------------------
	//VEGAS
	//----------------------------------------------------------------------
	unsigned int nthreads =  1;
	TString title;


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
	nthreads =  std::thread::hardware_concurrency();

	TH1D Hist_Duration_Per_Threads("Hist_Duration_Problem_Size",
			"Duration per number of threads",nthreads ,0, nthreads);

	if(nthreads){
		cout<<"------------------------------------"<< nthreads <<endl;
		cout<<"| System support #threads="<< nthreads <<endl;
		cout<<"------------------------------------"<< nthreads <<endl;
		omp_set_dynamic(0); //disable dynamic teams
		for(unsigned int nt=1; nt<nthreads+1; nt++){
			title=TString::Format("%d OpenMP Threads", nt);
			omp_set_num_threads(nt);

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_TBB
			nthreads =  std::thread::hardware_concurrency();
			title=TString::Format("%d TBB Threads", nthreads );
			unsigned int nt=1;

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
			title=TString::Format("CUDA", nthreads );
			unsigned int nt=1;

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CPP
			title=TString::Format("Sequential", nthreads );
			unsigned int nt=1;
#else
			title=TString::Format("?", nthreads );
			unsigned int nt=1;
#endif


	auto start_vegas = std::chrono::high_resolution_clock::now();
	vegas.Integrate(Gaussian);
	auto end_vegas = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
	cout <<endl;
	cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using "<< nt <<" OMP threads" <<endl;
	cout << "Result: " << vegas.GetState().GetResult()
		 << " +/- "    << vegas.GetState().GetSigma() <<std::endl
		 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

	TH1D Hist_Iterations_Results((TString("Hist_Iterations_Results_nthreads")+=nt).Data(), title+=";Iteration;Integral result",
			vegas.GetState().GetIterationResult().size(), 0.0,
			vegas.GetState().GetIterationResult().size());

	TH1D Hist_Cumulative_Results((TString("Hist_Cumulative_Results_nthreads")+=nt).Data(), title+=";Iteration;Integral result",
			vegas.GetState().GetCumulatedResult().size(), 0.0,
			vegas.GetState().GetCumulatedResult().size());

	TH1D Hist_Iterations_Duration((TString("Hist_Iterations_Duration_nthreads")+=nt).Data(), title+=";Iteration;Duration [ms]",
				vegas.GetState().GetIterationDuration().size(), 0.0,
				vegas.GetState().GetIterationDuration().size());

	TH1D Hist_FunctionCalls_Duration((TString("Hist_FunctionCall_Duration_nthreads")+=nt).Data(), title+=";Iteration;Duration [ms]",
					vegas.GetState().GetFunctionCallsDuration().size(), 0.0,
					vegas.GetState().GetFunctionCallsDuration().size());

	Hist_Duration_Per_Threads.SetBinContent(nt, elapsed_vegas.count());

	for(size_t i=1; i< Hist_Iterations_Results.GetNbinsX()+1; i++)
	{
		Hist_Cumulative_Results.SetBinContent(i, vegas.GetState().GetCumulatedResult()[i-1]);
		Hist_Cumulative_Results.SetBinError(i, vegas.GetState().GetCumulatedSigma()[i-1]);
		Hist_Iterations_Results.SetBinContent(i, vegas.GetState().GetIterationResult()[i-1]);
		Hist_Iterations_Results.SetBinError(i, vegas.GetState().GetIterationSigma()[i-1]);
		Hist_Iterations_Duration.SetBinContent(i, vegas.GetState().GetIterationDuration()[i-1]);
		Hist_Iterations_Duration.SetBinError(i, 0.0);
		Hist_FunctionCalls_Duration.SetBinContent(i, vegas.GetState().GetFunctionCallsDuration()[i-1]);
		Hist_FunctionCalls_Duration.SetBinError(i, 0.0);

	}

	Cumulative_Results.push_back(Hist_Cumulative_Results);
	Iterations_Results.push_back(Hist_Iterations_Results);
	Iterations_Duration.push_back(Hist_Iterations_Duration);
	FunctionCalls_Duration.push_back(Hist_FunctionCalls_Duration);


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
		}

	}
	else{
		cout<<"System does support or implement std::thread::hardware_concurrency" <<endl;
	}
#endif

	unsigned int steps =  10;
	size_t delta_ncalls= calls;

	TH1D Hist_Duration_Problem_Size((TString("Hist_Duration_Problem_Size")).Data(),
		    title+=";Iteration;Duration [ms]", steps, calls, steps*calls);

	for(size_t nc=0; nc< steps; nc++ )
	{
		size_t _ncalls= calls+nc*delta_ncalls;
		vegas.GetState().SetCalls( _ncalls );
		auto start = std::chrono::high_resolution_clock::now();
		vegas.Integrate(Gaussian);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_vegas = end - start;
		cout << endl;
		cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using "
				<< std::thread::hardware_concurrency() << " OMP threads"
				<< "Number of calls " << _ncalls<< endl;
	  	cout << "Result: " << vegas.GetState().GetResult()
									 << " +/- "    << vegas.GetState().GetSigma() <<std::endl
									 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

	  	Hist_Duration_Problem_Size.SetBinContent(nc+1, elapsed_vegas.count());
	}



/*
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
	cout << "Result: " << std::setprecision(10)<<result.first
					   << " +/- "    << result.second <<std::endl;

	GaussKronrodQuadrature<61,200> quad(min[0], max[0]);
	//quad.Print();
	auto start_quad = std::chrono::high_resolution_clock::now();
	auto r = quad.Integrate(Gaussian);
	auto end_quad = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_quad= end_quad - start_quad;
	cout << ">>> Gaussian intetgral [ GaussKronrodQuadrature]"<< endl;
	cout << "Result: " <<r.first << " " << r.second <<std::endl
	<< " Time (ms): "<< elapsed_quad.count() <<std::endl;

	GaussKronrodAdaptiveQuadrature<61,10> adaquad(min[0], max[0]);
	//adaquad.Print();
	auto start_adaquad = std::chrono::high_resolution_clock::now();
	auto adar = adaquad.Integrate(Gaussian);
	auto end_adaquad = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_adaquad= end_adaquad - start_adaquad;

	//adaquad.Print();
	cout << ">>> Gaussian intetgral [ GaussKronrodAdaptiveQuadrature]"<< endl;
	cout << "Result: " <<adar.first << "+/- " << adar.second <<std::endl
	<< " Time (ms): "<< elapsed_adaquad.count() <<std::endl;

	std::array<size_t, 3>  _grid{2,2,2};
	std::array<GReal_t, 3>  _min{-6,-6,-6};
	std::array<GReal_t, 3>  _max{6,6,6};
	Parameter  _mean[3]{0.0, 0.0, 0.0};
	Parameter  _sigma[3]{1.0, 1.0, 1.0};
    GUInt_t  _position[3]{0, 1, 2};

	GaussN<3> Gaussian3(_mean, _sigma, _position );

	auto GMIntegrator = GenzMalikQuadrature<3,hydra::device>(_min, _max, _grid);
	GMIntegrator.Print();

*/

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
	for(unsigned int nt=0; nt<Iterations_Results.size() ; nt++ )
	{
		TCanvas* canvas = new TCanvas((TString("canvas_result_")+=nt).Data(), "", 500, 500);
		canvas->SetGrid();
		canvas->SetTicks(1, 1);
		Iterations_Results[nt].Draw("E0");
		Iterations_Results[nt].SetLineWidth(2);
		Iterations_Results[nt].SetLineColor(kBlue);
		Iterations_Results[nt].SetMarkerSize(1);
		Iterations_Results[nt].SetMarkerStyle(20);
		Iterations_Results[nt].SetMarkerColor(kBlue);
		Iterations_Results[nt].GetYaxis()->SetTitleOffset(1.5);
		Iterations_Results[nt].SetStats(0);
		Cumulative_Results[nt].Draw("E3 same");
		Cumulative_Results[nt].SetLineColor(kRed);
		Cumulative_Results[nt].SetLineWidth(2);
		Cumulative_Results[nt].SetFillColor(kRed);
		Cumulative_Results[nt].SetFillStyle(3001);
		Cumulative_Results[nt].GetYaxis()->SetTitleOffset(1.5);
		Cumulative_Results[nt].SetStats(0);
		Cumulative_Results[nt].DrawCopy("hist c same")->SetFillColor(0);
		canvas->Update();

		TCanvas* canvas2 = new TCanvas((TString("canvas_time_")+=nt).Data(), "", 500, 500);
		canvas2->SetGrid();
		canvas2->SetTicks(1, 1);

		Iterations_Duration[nt].Draw("LP");
		Iterations_Duration[nt].SetLineWidth(2);
		Iterations_Duration[nt].SetLineColor(kBlue);
		Iterations_Duration[nt].SetMarkerSize(1);
		Iterations_Duration[nt].SetMarkerColor(kBlue);
		Iterations_Duration[nt].SetMarkerStyle(20);
		Iterations_Duration[nt].SetStats(0);
		Iterations_Duration[nt].GetYaxis()->SetTitleOffset(1.5);

		GReal_t min1= Iterations_Duration[nt].GetMinimum();
		GReal_t min2= FunctionCalls_Duration[nt].GetMinimum();
		Iterations_Duration[nt].SetMinimum(min1<min2?min1:min2 );

		FunctionCalls_Duration[nt].Draw("LPsame");
		FunctionCalls_Duration[nt].SetLineWidth(2);
		FunctionCalls_Duration[nt].SetLineColor(kRed);
		FunctionCalls_Duration[nt].SetMarkerSize(1);
		FunctionCalls_Duration[nt].SetMarkerColor(kRed);
		FunctionCalls_Duration[nt].SetMarkerStyle(20);
		FunctionCalls_Duration[nt].SetStats(0);
		FunctionCalls_Duration[nt].GetYaxis()->SetTitleOffset(1.5);

		canvas2->Update();

	}


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
	TCanvas* canvas3 = new TCanvas(TString("canvas_time_per_number_of_threads"), "", 500, 500);
	canvas3->SetGrid();
	canvas3->SetTicks(1, 1);
	Hist_Duration_Per_Threads.Draw("LP");
	Hist_Duration_Per_Threads.SetLineWidth(2);
	Hist_Duration_Per_Threads.SetLineColor(kBlue);
	Hist_Duration_Per_Threads.SetMarkerSize(1);
	Hist_Duration_Per_Threads.SetMarkerColor(kBlue);
	Hist_Duration_Per_Threads.SetMarkerStyle(20);
	Hist_Duration_Per_Threads.SetStats(0);
	Hist_Duration_Per_Threads.GetYaxis()->SetTitleOffset(1.5);
#endif

	TCanvas* canvas4 = new TCanvas(TString("canvas_time_per_ploblem_size"), "", 500, 500);
	canvas3->SetGrid();
	canvas3->SetTicks(1, 1);
	Hist_Duration_Problem_Size.Draw("LP");
	Hist_Duration_Problem_Size.SetLineWidth(2);
	Hist_Duration_Problem_Size.SetLineColor(kBlue);
	Hist_Duration_Problem_Size.SetMarkerSize(1);
	Hist_Duration_Problem_Size.SetMarkerColor(kBlue);
	Hist_Duration_Problem_Size.SetMarkerStyle(20);
	Hist_Duration_Problem_Size.SetStats(0);
	Hist_Duration_Problem_Size.GetYaxis()->SetTitleOffset(1.5);

	myapp->Run();

	return 0;


	}

