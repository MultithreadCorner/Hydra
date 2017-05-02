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
#include <hydra/experimental/GaussKronrodQuadrature.h>
#include <hydra/experimental/GaussKronrodAdaptiveQuadrature.h>
#include <hydra/experimental/GenzMalikRule.h>
#include <hydra/experimental/GenzMalikQuadrature.h>
//root
#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TString.h>
#include <TRatioPlot.h>
#include <TLegend.h>


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
    Gaussian.PrintRegisteredParameters();

	//----------------------------------------------------------------------
	//Vegas State_d hold the resources for performing the integration
	VegasState<N, device> State_d(_min, _max);
	State_d.SetVerbose(-2);
	State_d.SetAlpha(1.5);
	State_d.SetIterations( iterations );
	State_d.SetUseRelativeError(1);
	State_d.SetMaxError( max_error );
	State_d.SetCalls( calls );
	State_d.SetTrainingCalls( calls/10 );
	State_d.SetTrainingIterations(1);

	Vegas<N, device> Vegas_d(State_d);

	Vegas<N, host> Vegas_h(Vegas_d);

	vector<TH1D> Cumulative_Results;
	vector<TH1D> Iterations_Results;
	vector<TH1D> Iterations_Duration;
	vector<TH1D> FunctionCalls_Duration;
	vector<TH1D> Duration_Problem_Size;

	unsigned int nthreads =  1;
	TString title;
	TString backend;

	//----------------------------
	//CUDA
	//----------------------------
	{
		title=TString::Format("");
		unsigned int nt=0;
		auto start_vegas = std::chrono::high_resolution_clock::now();
		Vegas_d.Integrate(Gaussian);
		auto end_vegas = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
		cout <<endl;
		cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using "<< nt <<" OMP threads" <<endl;
		cout << "Result: " << Vegas_d.GetState().GetResult()
					 << " +/- "    << Vegas_d.GetState().GetSigma() <<std::endl
					 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

		TH1D Hist_Iterations_Results((TString("Hist_Iterations_Results_nthreads")+=nt).Data(), title+=";Iteration;Integral result",
				Vegas_d.GetState().GetIterationResult().size(), 0.0,
				Vegas_d.GetState().GetIterationResult().size());

		TH1D Hist_Cumulative_Results((TString("Hist_Cumulative_Results_nthreads")+=nt).Data(), title+=";Iteration;Integral result",
				Vegas_d.GetState().GetCumulatedResult().size(), 0.0,
				Vegas_d.GetState().GetCumulatedResult().size());

		TH1D Hist_Iterations_Duration((TString("Hist_Iterations_Duration_nthreads")+=nt).Data(), title+=";Iteration;Duration [ms]",
				Vegas_d.GetState().GetIterationDuration().size(), 0.0,
				Vegas_d.GetState().GetIterationDuration().size());

		TH1D Hist_FunctionCalls_Duration((TString("Hist_FunctionCall_Duration_nthreads")+=nt).Data(), title+=";Iteration;Duration [ms]",
				Vegas_d.GetState().GetFunctionCallsDuration().size(), 0.0,
				Vegas_d.GetState().GetFunctionCallsDuration().size());


		for(size_t i=1; i< Hist_Iterations_Results.GetNbinsX()+1; i++)
		{
			Hist_Cumulative_Results.SetBinContent(i, Vegas_d.GetState().GetCumulatedResult()[i-1]);
			Hist_Cumulative_Results.SetBinError(i, Vegas_d.GetState().GetCumulatedSigma()[i-1]);
			Hist_Iterations_Results.SetBinContent(i, Vegas_d.GetState().GetIterationResult()[i-1]);
			Hist_Iterations_Results.SetBinError(i, Vegas_d.GetState().GetIterationSigma()[i-1]);
			Hist_Iterations_Duration.SetBinContent(i, Vegas_d.GetState().GetIterationDuration()[i-1]);
			Hist_Iterations_Duration.SetBinError(i, 0.0);
			Hist_FunctionCalls_Duration.SetBinContent(i, Vegas_d.GetState().GetFunctionCallsDuration()[i-1]);
			Hist_FunctionCalls_Duration.SetBinError(i, 0.0);

		}

		Cumulative_Results.push_back(Hist_Cumulative_Results);
		Iterations_Results.push_back(Hist_Iterations_Results);
		Iterations_Duration.push_back(Hist_Iterations_Duration);
		FunctionCalls_Duration.push_back(Hist_FunctionCalls_Duration);

	}
	//----------------------------
	//HOST (OMP, TBB, CPP)
	//----------------------------
#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP

	backend =TString("OpenMP");

	nthreads =  std::thread::hardware_concurrency();

	TH1D Hist_Duration_Per_Threads("Hist_Duration_Per_Threads",
			" ;Number of OpenMP threads;Duration [ms]",nthreads ,0, nthreads);

	if(nthreads){
		cout<<"------------------------------------"<< nthreads <<endl;
		cout<<"| System support #threads="<< nthreads <<endl;
		cout<<"------------------------------------"<< nthreads <<endl;
		omp_set_dynamic(0); //disable dynamic teams
		for(unsigned int nt=1; nt<nthreads+1; nt++){
			title=TString::Format("%d OpenMP Threads", nt);
			omp_set_num_threads(nt);

#elif THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_TBB

			backend =TString("TBB");
			nthreads = std::thread::hardware_concurrency() ;
			title=TString::Format("%d TBB Threads", nthreads );
			unsigned int nt=1;

#elif THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_CPP

			backend =TString("CPP");
			title=TString::Format("Sequential", nthreads );
			unsigned int nt=1;
#else
			backend =TString("?");
			title=TString::Format("?", nthreads );
			unsigned int nt=1;
#endif


	auto start_vegas = std::chrono::high_resolution_clock::now();
	Vegas_h.Integrate(Gaussian);
	auto end_vegas = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_vegas = end_vegas - start_vegas;
	cout <<endl;
	cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using "<< nt <<" OMP threads" <<endl;
	cout << "Result: " << Vegas_h.GetState().GetResult()
		 << " +/- "    << Vegas_h.GetState().GetSigma() <<std::endl
		 << "Time (ms): "<< elapsed_vegas.count() <<std::endl;

	TH1D Hist_Iterations_Results((TString("Hist_Iterations_Results_nthreads")+=nt).Data()
			, ";Iteration;Integral result",
			Vegas_h.GetState().GetIterationResult().size(), 0.0,
			Vegas_h.GetState().GetIterationResult().size());

	TH1D Hist_Cumulative_Results((TString("Hist_Cumulative_Results_nthreads")+=nt).Data()
			, ";Iteration;Integral result",
			Vegas_h.GetState().GetCumulatedResult().size(), 0.0,
			Vegas_h.GetState().GetCumulatedResult().size());

	TH1D Hist_Iterations_Duration((TString("Hist_Iterations_Duration_nthreads")+=nt).Data()
			, ";Iteration;Duration [ms]",
				Vegas_h.GetState().GetIterationDuration().size(), 0.0,
				Vegas_h.GetState().GetIterationDuration().size());

	TH1D Hist_FunctionCalls_Duration((TString("Hist_FunctionCall_Duration_nthreads")+=nt).Data()
			, ";Iteration;Duration [ms]",
					Vegas_h.GetState().GetFunctionCallsDuration().size(), 0.0,
					Vegas_h.GetState().GetFunctionCallsDuration().size());
#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
	Hist_Duration_Per_Threads.SetBinContent(nt, elapsed_vegas.count());
#endif
	for(size_t i=1; i< Hist_Iterations_Results.GetNbinsX()+1; i++)
	{
		Hist_Cumulative_Results.SetBinContent(i, Vegas_h.GetState().GetCumulatedResult()[i-1]);
		Hist_Cumulative_Results.SetBinError(i, Vegas_h.GetState().GetCumulatedSigma()[i-1]);
		Hist_Iterations_Results.SetBinContent(i, Vegas_h.GetState().GetIterationResult()[i-1]);
		Hist_Iterations_Results.SetBinError(i, Vegas_h.GetState().GetIterationSigma()[i-1]);
		Hist_Iterations_Duration.SetBinContent(i, Vegas_h.GetState().GetIterationDuration()[i-1]);
		Hist_Iterations_Duration.SetBinError(i, 0.0);
		Hist_FunctionCalls_Duration.SetBinContent(i, Vegas_h.GetState().GetFunctionCallsDuration()[i-1]);
		Hist_FunctionCalls_Duration.SetBinError(i, 0.0);

	}

	Cumulative_Results.push_back(Hist_Cumulative_Results);
	Iterations_Results.push_back(Hist_Iterations_Results);
	Iterations_Duration.push_back(Hist_Iterations_Duration);
	FunctionCalls_Duration.push_back(Hist_FunctionCalls_Duration);


#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
		}

	}
	else{
		cout<<"System does support or implement std::thread::hardware_concurrency" <<endl;
	}
#endif

	unsigned int steps =  10;
	size_t delta_ncalls= calls;

#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
	omp_set_num_threads(1);
#endif

	TH1D Hist_Duration_Problem_Size_Device((TString("Hist_Duration_Problem_Size_Device")).Data(),
		    ";Number of samples;Duration [ms]", steps, calls, steps*calls);

	TH1D Hist_Duration_Problem_Size_Host((TString("Hist_Duration_Problem_Size_Host")).Data(),
			";Number of samples;Duration [ms]", steps, calls, steps*calls);

	TH1D 	SpeedUp("SpeedUp", ";Number of samples;Speed-up GPU vs CPU", steps , calls,  steps*calls);

	for(size_t nc=0; nc< steps; nc++ )
	{
		size_t _ncalls= calls+nc*delta_ncalls;

		Vegas_d.GetState().SetCalls( _ncalls );
		auto start_d = std::chrono::high_resolution_clock::now();
		Vegas_d.Integrate(Gaussian);
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		cout << endl;
		cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using CUDA"
			 << "Number of calls " << _ncalls<< endl;
	  	cout << "Result: " << Vegas_d.GetState().GetResult()
									 << " +/- "    << Vegas_d.GetState().GetSigma() <<std::endl
									 << "Time (ms): "<< elapsed_d.count() <<std::endl;

	  	Hist_Duration_Problem_Size_Device.SetBinContent(nc+1, elapsed_d.count());

	  	Vegas_h.GetState().SetCalls( _ncalls );
	  	auto start_h = std::chrono::high_resolution_clock::now();
	  	Vegas_h.Integrate(Gaussian);
	  	auto end_h = std::chrono::high_resolution_clock::now();
	  	std::chrono::duration<double, std::milli> elapsed_h = end_h - start_h;

	  	cout << endl;

	  	cout << ">>> [Vegas]: Gaussian<"<< N << ">" << " using OpenMP"
	  		 << "Number of calls " << _ncalls<< endl;
	  	cout << "Result: " << Vegas_h.GetState().GetResult()
	  		 << " +/- "    << Vegas_h.GetState().GetSigma() <<std::endl
	  		 << "Time (ms): "<< elapsed_h.count() <<std::endl;

	  	Hist_Duration_Problem_Size_Host.SetBinContent(nc+1, elapsed_h.count());

	  	SpeedUp.SetBinContent(nc+1,
	  			Hist_Duration_Problem_Size_Host.GetBinContent(nc+1)/Hist_Duration_Problem_Size_Device.GetBinContent(nc+1) );
	}



	TApplication *myapp=new TApplication("myapp",0,0);

	for(unsigned int nt=0; nt<Iterations_Results.size() ; nt++ )
	{
		TCanvas* canvas = new TCanvas((TString("canvas_result_")+=nt).Data(), "", 500, 500);
		canvas->SetGrid();
		canvas->SetTicks(1, 1);
		GInt_t nbins =Iterations_Results[nt].GetNbinsX();

		Iterations_Results[nt].Draw("E0");
		Iterations_Results[nt].SetLineWidth(2);
		Iterations_Results[nt].SetLineColor(kBlue);
		Iterations_Results[nt].SetMarkerSize(1);
		Iterations_Results[nt].SetMarkerStyle(20);
		Iterations_Results[nt].SetMarkerColor(kBlue);
		Iterations_Results[nt].GetYaxis()->SetTitleOffset(1.5);
		Iterations_Results[nt].GetXaxis()->CenterLabels();
		Iterations_Results[nt].SetStats(0);
		Iterations_Results[nt].GetXaxis()->SetRangeUser(0, nbins);//);

		Cumulative_Results[nt].Draw("E3 same");
		Cumulative_Results[nt].SetLineColor(kRed);
		Cumulative_Results[nt].SetLineWidth(2);
		Cumulative_Results[nt].SetFillColor(kRed);
		Cumulative_Results[nt].SetFillStyle(3001);
		Cumulative_Results[nt].GetYaxis()->SetTitleOffset(1.5);
		Cumulative_Results[nt].GetXaxis()->CenterLabels();
		Cumulative_Results[nt].SetStats(0);
		Cumulative_Results[nt].GetXaxis()->SetRangeUser(0, nbins);//Cumulative_Results[nt].GetNbinsX());
		auto h=Cumulative_Results[nt].DrawCopy("hist l same");
		h->SetFillColor(0);
		h->GetXaxis()->CenterLabels();
        h->GetXaxis()->SetRangeUser(0, nbins);//Cumulative_Results[nt].GetNbinsX());

		TLegend* legend_Results = new TLegend(0.5,0.2,0.85,0.35);
		if(nt==0)legend_Results->SetHeader("GPU","C");
#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
		else legend_Results->SetHeader(backend + TString::Format(" #%d %s", nt ," threads"),"C");
#else
		else legend_Results->SetHeader(backend + TString::Format(" #%d %s", nthreads ," threads"),"C");
#endif
		legend_Results->AddEntry(&Iterations_Results[nt],"Iteration result","lp");
		legend_Results->AddEntry(&Cumulative_Results[nt],"Cumulative result","lp");
		legend_Results->Draw();
		canvas->Update();
		canvas->SaveAs(TString::Format("plots/result_nt_%d_%s.pdf", nt, backend.Data()).Data() );



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
		Iterations_Duration[nt].GetXaxis()->CenterLabels();
		Iterations_Duration[nt].GetXaxis()->SetRangeUser(0, Iterations_Duration[nt].GetNbinsX());
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
		FunctionCalls_Duration[nt].GetXaxis()->CenterLabels();
		FunctionCalls_Duration[nt].GetXaxis()->SetRangeUser(0, FunctionCalls_Duration[nt].GetNbinsX());

		TLegend* legend_FunctionCalls = new TLegend(0.5,0.7,0.85,0.85);
		if(nt==0)legend_FunctionCalls->SetHeader("GPU","C");
		#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
				else legend_FunctionCalls->SetHeader(backend + TString::Format(" #%d %s", nt ," threads"),"C");
		#else
				else legend_FunctionCalls->SetHeader(backend + TString::Format(" #%d %s", nthreads ," threads"),"C");
		#endif
		legend_FunctionCalls->AddEntry(&Iterations_Duration[nt] ,"Complete iteration","lp");
		legend_FunctionCalls->AddEntry(&FunctionCalls_Duration[nt],"Function Sampling","lp");
		legend_FunctionCalls->Draw();

		canvas2->Update();
		canvas2->SaveAs(TString::Format("plots/time_per_iterations_nt_%d_%s.pdf", nt, backend.Data()).Data() );
	}


#if THRUST_HOST_SYSTEM==THRUST_HOST_SYSTEM_OMP
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
	Hist_Duration_Per_Threads.GetXaxis()->CenterLabels();
	Hist_Duration_Per_Threads.GetXaxis()->SetRangeUser(0, Hist_Duration_Per_Threads.GetNbinsX());
	Hist_Duration_Per_Threads.GetYaxis()->SetTitleOffset(1.5);canvas3->Update();
	canvas3->SaveAs(TString::Format("plots/time_per_thread_%d_%s.pdf", nthreads, backend.Data()).Data() );
#endif

	TCanvas* canvas4 = new TCanvas(TString("canvas_time_per_ploblem_size"), "", 500, 500);
	TPad *pad1 = new TPad("pad1","",0,0,1,1);
	pad1->Draw();
	pad1->cd();
	//pad1->SetLogy();

	//canvas4->SetGrid();
	//canvas4->SetTicks(1, 1);
	Hist_Duration_Problem_Size_Device.Draw("LP");
	Hist_Duration_Problem_Size_Device.SetLineWidth(2);
	Hist_Duration_Problem_Size_Device.SetLineColor(kBlue);
	Hist_Duration_Problem_Size_Device.SetMarkerSize(1);
	Hist_Duration_Problem_Size_Device.SetMarkerColor(kBlue);
	Hist_Duration_Problem_Size_Device.SetMarkerStyle(20);
	Hist_Duration_Problem_Size_Device.SetStats(0);
	Hist_Duration_Problem_Size_Device.GetYaxis()->SetTitleOffset(1.5);
	Hist_Duration_Problem_Size_Device.SetMaximum(1.1*Hist_Duration_Problem_Size_Host.GetMaximum());
	Hist_Duration_Problem_Size_Device.GetXaxis()->CenterLabels();
	Hist_Duration_Problem_Size_Device.GetXaxis()->SetRangeUser(0, Hist_Duration_Problem_Size_Device.GetNbinsX());


	Hist_Duration_Problem_Size_Host.Draw("LP same");
	Hist_Duration_Problem_Size_Host.SetLineWidth(2);
	Hist_Duration_Problem_Size_Host.SetLineColor(kRed);
	Hist_Duration_Problem_Size_Host.SetMarkerSize(1);
	Hist_Duration_Problem_Size_Host.SetMarkerColor(kRed);
	Hist_Duration_Problem_Size_Host.SetMarkerStyle(20);
	Hist_Duration_Problem_Size_Host.SetStats(0);
	Hist_Duration_Problem_Size_Host.GetYaxis()->SetTitleOffset(1.5);
	Hist_Duration_Problem_Size_Host.GetXaxis()->CenterLabels();
	Hist_Duration_Problem_Size_Host.GetXaxis()->SetRangeUser(0, Hist_Duration_Problem_Size_Host.GetNbinsX());

	pad1->Update();

	TPad *pad2 = new TPad("pad2","",0,0,1,1);
	pad2->SetFillStyle(4000); //transparent pad
	pad2->SetFillColor(0); //transparent pad
	pad2->SetFrameFillStyle(4000);
	pad2->SetFrameFillColor(0);
	pad2->Draw();
	pad2->cd();

	SpeedUp.SetLineColor(kViolet);
	SpeedUp.SetLineWidth(2);
	SpeedUp.SetFillStyle(4000);
	SpeedUp.SetFillColor(0);
	SpeedUp.GetXaxis()->CenterLabels();
	SpeedUp.GetYaxis()->SetTitleOffset(1.4);
	SpeedUp.SetMinimum(1);
	SpeedUp.SetStats(0);
	SpeedUp.Draw("Y+][sames");

	TLegend* legend4 = new TLegend(0.6,0.15,0.85,0.3);
	legend4->AddEntry(&Hist_Duration_Problem_Size_Device,"GPU","lp");
	legend4->AddEntry(&Hist_Duration_Problem_Size_Host,"CPU","lp");
	legend4->AddEntry(&SpeedUp, "speed-up","lp");
	legend4->Draw();
	canvas4->Update();
	canvas4->SaveAs(TString::Format("plots/time_per_problem_size_%s.pdf", backend.Data()).Data() );


	myapp->Run();

	return 0;


	}

