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
 * crystal_ball_plus_exponential.inl
 *
 *  Created on: 21/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CRYSTAL_BALL_PLUS_EXPONENTIAL_INL_
#define CRYSTAL_BALL_PLUS_EXPONENTIAL_INL_


/**
 * \example crystal_ball_plus_exponential.inl
 *
 */



#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>
#include <hydra/DenseHistogram.h>
#include <hydra/functions/CrystalBallShape.h>
#include <hydra/functions/Exponential.h>
#include <hydra/Placeholders.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace ROOT::Minuit2;
using namespace hydra::placeholders;

int main(int argv, char** argc)
{
	size_t nentries = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<size_t> EArg("n", "number-of-events","Number of events", true, 10e6, "size_t");
		cmd.add(EArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries = EArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
														<< std::endl;
	}

	//-----------------
    // some definitions
    double min   = -20.0;
    double max   =  20.0;
    char const* model_name = "Crystal Ball + Exponential";

	//generator
	hydra::Random<> Generator(154);

	//===============================================================================================
    //fit model Crystal Ball + Exponential

	//Crystal Ball
	//parameters
	auto  mean  = hydra::Parameter::Create().Name("Mean").Value(0.0).Error(0.0001).Limits(-1.5, 1.5);
	auto  sigma = hydra::Parameter::Create().Name("Sigma").Value(1.0).Error(0.0001).Limits(0.1, 2.5);
	auto  alpha = hydra::Parameter::Create().Name("Alpha").Value(-1.5).Error(0.0001).Limits(-1.6, 1.4);
	//fix tail parameters (usually done with mc)
	auto  n     = hydra::Parameter::Create().Name("N").Value(2.5).Error(0.0001).Limits(0.0, 5.0).Fixed();

	//Signal PDF
	auto Signal_PDF = hydra::make_pdf(hydra::CrystalBallShape<>(mean, sigma, alpha, n),
			hydra::AnalyticalIntegral<hydra::CrystalBallShape<>>(min, max) );

    //-------------------------------------------

	//Exponential
    //parameters
    auto  tau  = hydra::Parameter::Create().Name("Tau").Value(-0.1).Error(0.0001).Limits(-1.0, 0.0);

    //Background
    auto Background_PDF = hydra::make_pdf(hydra::Exponential<>(tau),
    		hydra::AnalyticalIntegral<hydra::Exponential<>>(min, max));

    //------------------
    //yields
	hydra::Parameter N_Signal("N_Signal"        ,2000, 1, 100 , nentries) ;
	hydra::Parameter N_Background("N_Background",2000, 1, 100 , nentries) ;

	//make model
	auto model = hydra::add_pdfs( {N_Signal, N_Background}, Signal_PDF, Background_PDF);
	model.SetExtended(1);

	//===========================

#ifdef _ROOT_AVAILABLE_

	TH1D 	hist_data("data"			, model_name, 100, min, max);
	TH1D 	hist_fit("fit"  			, model_name, 100, min, max);
	TH1D 	hist_signal("signal"		, model_name, 100, min, max);
	TH1D 	hist_background("background", model_name, 100, min, max);


#endif //_ROOT_AVAILABLE_


	//scope begin
	{

		//1D device buffer
		hydra::device::vector<double>  data(nentries);

		//-------------------------------------------------------
		// Generate data
		auto range = Generator.Sample(data.begin(),  data.end(), min, max, model.GetFunctor());

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i< 10; i++)
			std::cout << "[" << i << "] :" << range[i] << std::endl;

		//make model and fcn
		auto fcn = hydra::make_loglikehood_fcn( model, range.begin(), range.end() );

		//-------------------------------------------------------
		//fit
		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);
		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		// ... Minimize and profile the time

		auto start_d = std::chrono::high_resolution_clock::now();

		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(5000, 5));

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		// output
		std::cout<< "Minimum: " << minimum_d << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Fit] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;


		//--------------------------------------------
		hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Data(100, min, max);
		Hist_Data.Fill( range.begin(), range.end() );

#ifdef _ROOT_AVAILABLE_

		//data
		for(size_t i=0;  i<100; i++)
			hist_data.SetBinContent(i+1, Hist_Data.GetBinContent(i));

		//fit
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_fit.GetBinCenter(i);
	        hist_fit.SetBinContent(i, fcn.GetPDF()(x) );
		}
		hist_fit.Scale(hist_data.Integral()/hist_fit.Integral() );

		//signal component
		auto   signal          = fcn.GetPDF().PDF(_0);
		double signal_fraction = fcn.GetPDF().Coeficient(0)/fcn.GetPDF().GetCoefSum();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_signal.GetBinCenter(i);
			hist_signal.SetBinContent(i, signal(x) );
		}
		hist_signal.Scale(hist_data.Integral()*signal_fraction/hist_signal.Integral());

		//signal component
		auto   background          = fcn.GetPDF().PDF(_1);
		double background_fraction = fcn.GetPDF().Coeficient(1)/fcn.GetPDF().GetCoefSum();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_background.GetBinCenter(i);
			hist_background.SetBinContent(i, background(x) );
		}
		hist_background.Scale(hist_data.Integral()*background_fraction/hist_background.Integral());


#endif //_ROOT_AVAILABLE_

	}//scope end


#ifdef _ROOT_AVAILABLE_

	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 500, 500);

	hist_data.Draw("e0");
	hist_data.SetStats(0);
	hist_data.SetLineColor(1);
	hist_data.SetLineWidth(2);

	hist_fit.Draw("histsameC");
	hist_fit.SetStats(0);
	hist_fit.SetLineColor(4);

	hist_signal.Draw("histsameC");
	hist_signal.SetStats(0);
	hist_signal.SetLineColor(3);

	hist_background.Draw("histsameC");
	hist_background.SetStats(0);
	hist_background.SetLineColor(2);
	hist_background.SetLineStyle(2);

	hist_fit.Draw("histsameC");
	hist_data.Draw("e0same");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;

}


#endif /* CRYSTAL_BALL_PLUS_EXPONENTIAL_INL_ */
