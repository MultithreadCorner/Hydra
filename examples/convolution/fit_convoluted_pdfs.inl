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
 * fit_convoluted_pdfs.inl
 *
 *  Created on: 06/01/2019
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FIT_CONVOLUTED_PDFS_INL_
#define FIT_CONVOLUTED_PDFS_INL_


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
#include <hydra/Placeholders.h>
//hydra
#if HYDRA_DEVICE_SYSTEM == CUDA
#include <hydra/CuFFT.h>
#endif

#if HYDRA_DEVICE_SYSTEM != CUDA
#include <hydra/FFTW.h>
#endif

//functors
#include <hydra/functions/BreitWignerNR.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Exponential.h>
#include <hydra/functions/ConvolutionFunctor.h>
#include <hydra/GaussKronrodQuadrature.h>
//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

// Include classes from ROOT
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace hydra::placeholders;
using namespace ROOT::Minuit2;

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
    double min   =  1.0;
    double max   =  11.0;
    unsigned nbins = 250;

	//generator
	hydra::Random<> Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//===========================
    //fit model

	//gaussian convolution kernel
	hydra::Parameter  mean  = hydra::Parameter::Create().Name("Mean_Kernel").Value( 0.0).Fixed();
	hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma_Kernel").Value(0.5).Fixed();

	hydra::Gaussian<0> gaussian_kernel(mean, sigma);

	//Breit-Wigner
	hydra::Parameter  mass  = hydra::Parameter::Create().Name("Mass" ).Value(5.0).Error(0.0001).Limits(4.0,6.0);
	hydra::Parameter  width = hydra::Parameter::Create().Name("Width").Value(0.05).Error(0.0001).Limits(0.01,0.1);

	hydra::BreitWignerNR<0> bw_signal(mass, width );

	//convolution, using storage in device side
#if HYDRA_DEVICE_SYSTEM==CUDA
	auto fft_backend = hydra::fft::cufft_f64;
#endif

#if HYDRA_DEVICE_SYSTEM!=CUDA
	auto fft_backend = hydra::fft::fftw_f64;
#endif

	auto convolution_signal = hydra::make_convolution<0>( hydra::device::sys, fft_backend, bw_signal, gaussian_kernel, min, max,10112);

	auto Signal_PDF = hydra::make_pdf( convolution_signal, hydra::GaussKronrodQuadrature<61, 100, hydra::device::sys_t>(min,  max));

    //--------------------------------------------

    //exponential
    //parameters
    hydra::Parameter  tau  = hydra::Parameter::Create().Name("Tau").Value(-0.1) .Error(0.0001).Limits(-0.5, 0.5);

    //gaussian function evaluating on the first argument
    hydra::Exponential<0> exponential(tau);

    auto Background_PDF = hydra::make_pdf(exponential, hydra::AnalyticalIntegral<hydra::Exponential<0>>(min, max));

    //------------------
    //yields
	hydra::Parameter N_Signal("N_Signal" ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;
	hydra::Parameter N_Background("N_Background" ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;

	//make model
	auto model = hydra::add_pdfs( {N_Signal, N_Background }, Signal_PDF, Background_PDF);
	model.SetExtended(1);

	//===========================

#ifdef _ROOT_AVAILABLE_
	TH1D     hist_data(  "data"      , "", nbins, min, max);
	TH1D     hist_raw_signal("raw_signal"    , "", nbins, 4.0, 6.0);
	TH1D     hist_signal("signal"    , "", nbins, min, max);
	TH1D hist_background("background", "", nbins, min, max);
	TH1D      hist_total("total"     , "", nbins, min, max);

#endif //_ROOT_AVAILABLE_

	//begin scope
	{

		//1D device buffer
		hydra::device::vector<double>  data(2*nentries);

		//-------------------------------------------------------
		// Generate data

		// signal
		auto range = Generator.Sample(data.begin(),  data.end(), min, max, model.GetFunctor());

		// exponential
		//Generator.Exp(tau.GetValue(), data.begin() + nentries,  data.end());

		std::cout<< std::endl<< "Generated data...: "<< std::endl;

		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << data[i] << std::endl;

		hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Data(nbins, min, max);
		Hist_Data.Fill( range.begin(), range.end() );

		//make model and fcn
		auto fcn   = hydra::make_loglikehood_fcn( model, Hist_Data);

		//-------------------------------------------------------

		//fit
		ROOT::Minuit2::MnPrint::SetLevel(-1);
		hydra::Print::SetLevel(hydra::WARNING);
		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		// ... Minimize and profile the time

		auto start_d = std::chrono::high_resolution_clock::now();
		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(std::numeric_limits<unsigned int>::max(), 5));
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		// output
		std::cout<<"Minimum: "<< minimum_d << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Fit] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

#ifdef _ROOT_AVAILABLE_
		hist_data.Sumw2();
		for(size_t i=0;  i<nbins; i++)
			hist_data.SetBinContent(i+1, Hist_Data.GetBinContent(i));

		//draw fitted function
		for (size_t i=0 ; i<=nbins ; i++) {
			//
			hist_total.SetBinContent( i,
					fcn.GetPDF()(hist_total.GetBinCenter(i) ) );

			hist_raw_signal.SetBinContent( i,
								fcn.GetPDF().PDF(_0).GetFunctor().GetFunctor(_0)(hist_raw_signal.GetBinCenter(i) ) );

			hist_signal.SetBinContent( i,
					fcn.GetPDF().PDF(_0)(hist_signal.GetBinCenter(i) ) );

			hist_background.SetBinContent( i,
					fcn.GetPDF().PDF(_1)(hist_background.GetBinCenter(i) ) );

		}

		double signal_fraction = fcn.GetPDF().Coeficient(0)/fcn.GetPDF().GetCoefSum();
		hist_signal.Scale(hist_data.Integral()*signal_fraction/hist_signal.Integral() );

		double background_fraction = fcn.GetPDF().Coeficient(1)/fcn.GetPDF().GetCoefSum();
		hist_background.Scale(hist_data.Integral()*background_fraction/hist_background.Integral() );

		hist_total.Scale(hist_data.Integral()/hist_total.Integral() );

#endif //_ROOT_AVAILABLE_

	}//end scope

#ifdef _ROOT_AVAILABLE_

	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas("canvas_d" ,"Distributions - Device", 1000, 500);
	canvas.Divide(2,1);
	canvas.cd(1);
	hist_data.SetMinimum(0);
	hist_data.Draw("E1");
	hist_data.SetLineWidth(2);

	//total
	hist_total.Draw("histsameC");
	hist_total.SetLineColor(4);
	hist_total.SetLineWidth(2);

	//total
	hist_signal.Draw("histsameC");
	hist_signal.SetLineColor(8);
	hist_signal.SetLineWidth(2);
	//total
	hist_background.Draw("histsameC");
	hist_background.SetLineColor(2);
	hist_background.SetLineWidth(2);

	canvas.cd(2);


    //raw_signal
	auto h=hist_raw_signal.DrawNormalized("histC");
	h->SetLineColor(6);
	h->SetLineWidth(2);

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	convolution_signal.Dispose();
	return 0;


}




#endif /* FIT_CONVOLUTED_PDFS_INL_ */
