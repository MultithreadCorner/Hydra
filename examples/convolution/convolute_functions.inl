/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * convolute_functions.inl
 *
 *  Created on: Nov 22, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTE_FUNCTIONS_INL_
#define CONVOLUTE_FUNCTIONS_INL_


/**
 * \example convolute_functions.inl
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <vector>


#include <hydra/Convolution.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/TrapezoidalShape.h>
#include <hydra/device/System.h>
#include <hydra/functions/ConvolutionFunctor.h>
#include <hydra/GaussKronrodQuadrature.h>

//hydra
#if HYDRA_DEVICE_SYSTEM == CUDA
#include <hydra/CuFFT.h>
#endif

#if HYDRA_DEVICE_SYSTEM != CUDA
#include <hydra/FFTW.h>
#endif


//command line
#include <tclap/CmdLine.h>

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


int main(int argv, char** argc)
{
	size_t nentries = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg<size_t> EArg("n", "number-of-samples","Number of samples", true, 10e6, "size_t");
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

	double min=-10.0;
	double max= 10.0;

	auto nsamples = nentries;
	//===========================
	// kernels
	//---------------------------

	//Gaussian
	auto mean   = hydra::Parameter::Create( "mean").Value(1.0).Error(0.0001);//bias
	auto sigma  = hydra::Parameter::Create("sigma").Value(0.5).Error(0.0001);//resolution

	hydra::Gaussian<double> kernel(mean,  sigma);

	//===========================
	// signals
	//---------------------------


	//Trapezoid
	auto A = hydra::Parameter::Create("A").Value(-5.0).Error(0.0001);
	auto B = hydra::Parameter::Create("B").Value(-2.0).Error(0.0001);
	auto C = hydra::Parameter::Create("C").Value( 2.0).Error(0.0001);
	auto D = hydra::Parameter::Create("D").Value( 5.0).Error(0.0001);


	//ipatia function evaluating on the first argument
	auto signal = hydra::TrapezoidalShape<double>(A, B, C, D);


	//===========================
	// samples
	//---------------------------
	hydra::device::vector<double> conv_result(nsamples, 0.0);

	//uniform (x) gaussian kernel

#if HYDRA_DEVICE_SYSTEM==CUDA
	auto fft_backend = hydra::fft::cufft_f64;
#endif

#if HYDRA_DEVICE_SYSTEM!=CUDA
	auto fft_backend = hydra::fft::fftw_f64;
#endif

	/*
	 * using the hydra::convolute
	 */

	auto start_d = std::chrono::high_resolution_clock::now();

	hydra::convolute(hydra::device::sys, fft_backend,
			signal, kernel, min, max,  conv_result, true);

	auto end_d = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;
	//time
	std::cout << "-----------------------------------------"<<std::endl;
	std::cout << "| Time (ms) ="<< elapsed_d.count()    <<std::endl;
	std::cout << "-----------------------------------------"<<std::endl;


	/*
	 * using the hydra::ConvolutionFunctor
	 */
    auto convoluton = hydra::make_convolution<double>(  hydra::device::sys,  fft_backend,
    		signal, kernel, min, max,  conv_result.size() );

    hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61_Kernel(-25.0, 25.0);
    hydra::GaussKronrodQuadrature<61,100, hydra::device::sys_t> GKQ61_Signal(min,  max);

    auto kernel_int = GKQ61_Kernel.Integrate(kernel);
    auto signal_int = GKQ61_Signal.Integrate(signal);
    auto convol_int = GKQ61_Signal.Integrate(convoluton);

    std::cout << "===========================================" << std::endl;
    std::cout << "Kernel Integral: " <<kernel_int.first <<"+/-"<< kernel_int.second << std::endl;
    std::cout << "Signal Integral: " <<signal_int.first <<"+/-"<< signal_int.second << std::endl;
    std::cout << "Convol Integral: " <<convol_int.first <<"+/-"<< convol_int.second << std::endl;
    std::cout << "===========================================" << std::endl;


	//------------------------
	//------------------------
#ifdef _ROOT_AVAILABLE_

	//fill histograms
	TH1D *hist_convol   = new TH1D("convol","Convolution result", conv_result.size(), min, max);
	TH1D *hist_convol_functor   = new TH1D("convol_functor","Convolution functor", conv_result.size(), min, max);
	TH1D *hist_signal   = new TH1D("signal", "Signal", conv_result.size(), min, max);
	TH1D *hist_kernel   = new TH1D("kernel", "Gaussian resolution model: bias 1.0 and width 0.5)", conv_result.size(), -0.5*(max-min),0.5*(max-min) );

	for(int i=1;  i<hist_convol->GetNbinsX()+1; i++){

		hist_convol_functor->SetBinContent(i, convoluton(hist_convol_functor->GetBinCenter(i)) );
		hist_convol->SetBinContent(i, conv_result[i-1] );
		hist_signal->SetBinContent(i, signal(hist_signal->GetBinCenter(i) ) );
		hist_kernel->SetBinContent(i, kernel(hist_kernel->GetBinCenter(i) ) );
	}
#endif //_ROOT_AVAILABLE_






#ifdef _ROOT_AVAILABLE_

	TApplication *myapp=new TApplication("myapp",0,0);

	//----------------------------
	//draw histograms
	TCanvas* canvas = new TCanvas("canvas" ,"canvas", 1500, 1000);
	canvas->Divide(2,2);

	auto c1 = canvas->cd(1);

	hist_convol->SetStats(0);
	hist_convol->SetLineColor(4);
	hist_convol->SetLineWidth(2);
	c1->SetGrid();
	hist_convol->Draw("histl");


	auto c2 = canvas->cd(2);
	hist_convol_functor->SetStats(0);
	hist_convol_functor->SetLineColor(8);
	hist_convol_functor->SetLineWidth(2);
	c2->SetGrid();
	hist_convol_functor->Draw("histl");


	auto c3 = canvas->cd(3);
	hist_signal->SetStats(0);
	hist_signal->SetLineColor(2);
	hist_signal->SetLineWidth(2);
	c3->SetGrid();
	hist_signal->Draw("histl");

    auto c4 =  canvas->cd(4);
	hist_kernel->SetStats(0);
	hist_kernel->SetLineColor(6);
	hist_kernel->SetLineWidth(2);
	c4->SetGrid();
	hist_kernel->Draw("hist");


	myapp->Run();

#endif //_ROOT_AVAILABLE_
convoluton.Dispose();
	return 0;

}


#endif /* CONVOLUTE_FUNCTIONS_INL_ */
