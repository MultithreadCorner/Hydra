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
 * kaon_mass.inl
 *
 *  Created on: 02/08/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PARTICLE_MASS_INL_
#define PARTICLE_MASS_INL_


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
#include <hydra/FunctorArithmetic.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>
#include <hydra/Filter.h>
#include <hydra/DenseHistogram.h>
#include <hydra/functions/Ipatia.h>
#include <hydra/functions/DeltaDMassBackground.h>
#include <hydra/functions/GeneralizedGamma.h>
#include <hydra/Placeholders.h>
#include <hydra/GaussKronrodQuadrature.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

// * Include classes from ROOT to fill
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
    double min   =  418.71;
    double max   =  550.0 ;

	//generator
	hydra::Random<> Generator(154);

	//===========================
    //fit model gaussian + argus

	//Ipatia
	//core
	auto mu    = hydra::Parameter::Create("mu").Value(493.677).Error(0.0001).Limits(493.5, 493.7);
	auto sigma = hydra::Parameter::Create("sigma").Value(3.5).Error(0.0001).Limits(2.0,5.0);
	//left tail
	auto L1    = hydra::Parameter::Create("L1").Value(1.7).Error(0.0001).Limits(0.5, 2.6);
	auto N1    = hydra::Parameter::Create("N1").Value(1.45).Error(0.0001).Limits(0.25, 2.7);
	//right tail
	auto L2    = hydra::Parameter::Create("L2").Value(2.6).Error(0.0001).Limits(1.0, 3.5);
	auto N2    = hydra::Parameter::Create("N2").Value(2.35).Error(0.0001).Limits(1.0, 3.5);
	//peakness
	auto alfa  = hydra::Parameter::Create("alfa").Value(-1.1).Error(0.0001).Limits(-1.5, -0.5);
	auto beta  = hydra::Parameter::Create("beta").Value(0.1).Error(0.0001).Limits(0.05, 0.5).Fixed();


	//ipatia function evaluating on the first argument
	auto Signal_PDF = hydra::make_pdf(hydra::Ipatia<>(mu, sigma,L1,N1,L2,N2,alfa,beta),
		hydra::AnalyticalIntegral<hydra::Ipatia<>>(min,  max));

    //-------------------------------------------
	//Background

    //threshold
    auto  M0     = hydra::Parameter::Create("M0").Value(418.71).Error(0.0001).Limits(415.0, 421.0).Fixed();

    //combinatorial
    auto  A1 = hydra::Parameter::Create("A1").Value( 0.55).Error(0.0001).Limits(  0.4,  0.6);
    auto  B1 = hydra::Parameter::Create("B1").Value( 0.067).Error(0.0001).Limits(  0.002,  0.1);
    auto  C1 = hydra::Parameter::Create("C1").Value( 15.50).Error(0.0001).Limits( 10.0, 20.0);

    auto Combinatorial_Background_PDF = hydra::make_pdf( hydra::DeltaDMassBackground<>(M0, A1, B1, C1),
    		hydra::AnalyticalIntegral<hydra::DeltaDMassBackground<>>(min,  max));

    //partial reconstructed -1.5, -10. , 15.
    auto  A2 = hydra::Parameter::Create("A2").Value(0.4).Error(0.0001).Limits( 0.3, 0.5);
    auto  B2 = hydra::Parameter::Create("B2").Value(3.8).Error(0.0001).Limits( 1.0, 5.0);
    auto  C2 = hydra::Parameter::Create("C2").Value(0.85).Error(0.0001).Limits(0.5 , 1.0);

    auto PartialRec_Background_PDF = hydra::make_pdf( hydra::GeneralizedGamma<>(M0, A2, B2, C2),
    		hydra::AnalyticalIntegral<hydra::GeneralizedGamma<>>(min,  max));

    //------------------
    //yields
	hydra::Parameter        N_Signal("N_Signal"        ,5000, 100, 100 , nentries) ;
	hydra::Parameter N_Combinatorial("N_Combinatorial" ,6000, 100, 100 , nentries) ;
	hydra::Parameter    N_PartialRec("N_PartialRec"    ,2000, 100, 100 , nentries) ;

	//make model
	auto model = hydra::add_pdfs( {N_Signal, N_Combinatorial, N_PartialRec},
			Signal_PDF, Combinatorial_Background_PDF, PartialRec_Background_PDF);

	model.SetExtended(true);

#ifdef _ROOT_AVAILABLE_

	TH1D 	hist_data("data"	, "", 100, min, max);
	TH1D 	hist_fit("fit"  	, "", 100, min, max);
	TH1D 	hist_signal("signal", "", 100, min, max);
	TH1D 	hist_combinatorial("combinatorial"  , "", 100, min, max);
	TH1D 	hist_partialrec("partial"  , "", 100, min, max);

#endif //_ROOT_AVAILABLE_

	//scope begin
	{

		//1D device buffer
		hydra::device::vector<double>  data(nentries);

		//-------------------------------------------------------
		// Generate data
		auto range = Generator.Sample(data.begin(),  data.end(), min, max, model.GetFunctor());

		std::cout<< std::endl<< "Generated data (10 first elements):"<< std::endl;
		for(size_t i=0; i< 10; i++)
			std::cout << "[" << i << "] :" << range[i] << std::endl;

		std::cout<< std::endl<< "Total dataset size :"<< range.size() << std::endl;

		//make model and fcn
		auto fcn = hydra::make_loglikehood_fcn( model, range );

		//-------------------------------------------------------
		//fit
		ROOT::Minuit2::MnPrint::SetLevel(3);
		//hydra::Print::SetLevel(hydra::INFO);

		MnStrategy strategy(1);

		// create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		//print parameters prior the fit
		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		// ... Minimize and profile the time

		auto start_d = std::chrono::high_resolution_clock::now();

		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(5000, 50));

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
		Hist_Data.Fill( range );




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

		//combinatorial component
		auto   combinatorial       = fcn.GetPDF().PDF(_1);
		double combinatorial_fraction = fcn.GetPDF().Coeficient(1)/fcn.GetPDF().GetCoefSum();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_combinatorial.GetBinCenter(i);
			hist_combinatorial.SetBinContent(i, combinatorial(x) );
		}
		hist_combinatorial.Scale(hist_data.Integral()*combinatorial_fraction/hist_combinatorial.Integral());

		//partial component
		auto   partialrec          = fcn.GetPDF().PDF(_2);
		double partialrec_fraction = fcn.GetPDF().Coeficient(2)/fcn.GetPDF().GetCoefSum();
		for (size_t i=0 ; i<=100 ; i++) {
			double x = hist_partialrec.GetBinCenter(i);
			hist_partialrec.SetBinContent(i, partialrec(x) );
		}
		hist_partialrec.Scale(hist_data.Integral()*partialrec_fraction/hist_partialrec.Integral());




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

	hist_combinatorial.Draw("histsameC");
	hist_combinatorial.SetStats(0);
	hist_combinatorial.SetLineColor(2);
	hist_combinatorial.SetLineStyle(2);

	hist_partialrec.Draw("histsameC");
	hist_partialrec.SetStats(0);
	hist_partialrec.SetLineColor(6);
	hist_partialrec.SetLineStyle(2);

	hist_fit.Draw("histsameC");
	hist_data.Draw("e0same");


	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;

}





#endif /* PARTICLE_MASS_INL_ */
