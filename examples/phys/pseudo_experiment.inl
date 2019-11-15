/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * pseudo_experiment.inl
 *
 *  Created on: 14/11/2019
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PSEUDO_EXPERIMENT_INL_
#define PSEUDO_EXPERIMENT_INL_

/**
 * \example pseudo_experiment.inl
 *
 */

/**
 * \brief This example shows how to implement a fast pseudo-experiment chain
 * to estimate the value and uncertainty of observables using SPlots and Booststrapping
 * in a conceptualy consistent way.
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>
#include <hydra/Algorithm.h>
#include <hydra/Filter.h>
#include <hydra/GaussKronrodQuadrature.h>
#include <hydra/SPlot.h>
#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Exponential.h>

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

 // Include classes from ROOT
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


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

	/*
     * Dataset layout: the data points are 2 dimensional. Each data point can be
     * represented by hydra::tuple<double, double> object. The values in the dimension <0>
     * represents the discriminant variable, which is distributed following a gaussian + argus.
     * The dimension <1> is the observable, which is distributed following a non-relativistic breit-wigner + exponential
     *
     */

	//======================================================
	// Pseudo-experiment Model
	//======================================================

	//======================================================
	// 1) Gaussian + Exponential model (dimension <0>)
	// data range
    double data_min   =  0.0;
    double data_max   =  10.0;

	//Gaussian

	//parameters
	hydra::Parameter  mean  = hydra::Parameter::Create().Name("Mean").Value( 5.28).Error(0.0001).Limits(5.25,5.29);
	hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma").Value(0.0026).Error(0.0001).Limits(0.0024,0.0028);

	//gaussian function evaluating on the first argument
	hydra::Gaussian<> signal(mean, sigma);
	auto Gaussian_PDF = hydra::make_pdf( hydra::Gaussian<>(mean, sigma),
			hydra::AnalyticalIntegral<hydra::Gaussian<>>(data_min, data_max));

	//-------------------------------------------
	//Exponential
    //parameters
    auto  tau  = hydra::Parameter::Create().Name("Tau").Value(-0.1).Error(0.0001).Limits(-1.0, 0.0);

    //Background PDF
    auto Exponential_PDF = hydra::make_pdf(hydra::Exponential<>(tau),
    		 hydra::AnalyticalIntegral<hydra::Exponential<>>(data_min, data_max));

	//------------------

	//yields
	hydra::Parameter N_Exponential("N_Exponential", 500, 100, 100 , nentries) ;
	hydra::Parameter N_Gaussian("N_Gaussian", 2000, 100, 100 , nentries) ;

	//make model
	auto discriminant_variable_model = hydra::add_pdfs( {N_Gaussian, N_Exponential}, N_Gaussian, N_Exponential);
	discriminant_variable_model.SetExtended(1);

	//======================================================
	// 2) Breit-Wigner + Chebychev (dimension <1>)
	//-----------------
	// data range
	double obs_min   =  0.0;
	double obs_max   =  15.0;

	//Breit-Wigner

	//parameters
	hydra::Parameter  mass  = hydra::Parameter::Create().Name("Mass" ).Value(6.0).Error(0.0001).Limits(5.0,7.0);
	hydra::Parameter  width = hydra::Parameter::Create().Name("Width").Value(0.5).Error(0.0001).Limits(0.3,1.0);

	//Breit-Wigner function evaluating on the first argument
	auto BreitWigner_PDF = hydra::make_pdf( hydra::BreitWignerNR<>(mass, width ),
			hydra::AnalyticalIntegral<hydra::BreitWignerNR<>>(obs_min, obs_max));

    //-------------------------------------------

	//Chebychev

    //parameters
    auto  c0  = hydra::Parameter::Create("C_0").Value( 1.5).Error(0.0001).Limits( 1.0, 2.0);
    auto  c1  = hydra::Parameter::Create("C_1").Value( 0.2).Error(0.0001).Limits( 0.1, 0.3);
    auto  c2  = hydra::Parameter::Create("C_2").Value( 0.1).Error(0.0001).Limits( 0.01, 0.2);
    auto  c3  = hydra::Parameter::Create("C_3").Value( 0.1).Error(0.0001).Limits( 0.01, 0.2);

    //Polynomial function evaluating on the first argument
    auto Chebychev_PDF = hydra::make_pdf( hydra::Chebychev<3>(obs_min, obs_max, std::array<hydra::Parameter,4>{c0, c1, c2, c3}),
    		hydra::AnalyticalIntegral< hydra::Chebychev<3>>(obs_min, obs_max));

    //------------------
    //yields
	hydra::Parameter N_Signal("N_Signal"        ,500, 100, 100 , nentries) ;
	hydra::Parameter N_Background("N_Background",2000, 100, 100 , nentries) ;

	//make model
	auto model = hydra::add_pdfs( {N_Signal, N_Background}, Signal_PDF, Background_PDF);
	model.SetExtended(1);

	//======================================================
	// Pseudo-experiment data sample generation
	//======================================================
	//
	//generator
	hydra::Random<> Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//dataset
	hydra::multiarray<double,2,  hydra::device::sys_t> data_d(nentries);



	//device
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_data_dicriminating_d("data_discriminating_d", "Discriminating variable", 100, min, max);
	TH1D hist_data_control_d("data_control_d", "Control Variable", 100, min, max);
	TH1D hist_fit_d("fit_d", "Discriminating variable", 100, min, max);
	TH1D hist_control_1_d("control_1_d", "Control Variable: Gaussian PDF",    100, min, max);
	TH1D hist_control_2_d("control_2_d", "Control Variable: Exponential PDF",    100, min, max);

#endif //_ROOT_AVAILABLE_

	{


		//1D data containers
		hydra::multiarray<double,2,  hydra::device::sys_t> data_d(2*nentries);
		hydra::multiarray<double,2,  hydra::host::sys_t>   data_h(2*nentries);

		//-------------------------------------------------------
		// Generate toy data

		//first component: [Gaussian] x [Exponential]
		// gaussian
		Generator.Gauss(mean_p.GetValue()+2.5, sigma_p.GetValue()+0.5, data_d.begin(0), data_d.begin(0)+nentries);

		// exponential

		Generator.Exp(tau_p.GetValue()+1.0, data_d.begin(1),  data_d.begin(1)+nentries);

		//second component: [Exponential] -> [Gaussian]
		// gaussian
		Generator.Gauss(mean_p.GetValue()-1.0, 0.5, data_d.begin(1) + nentries, data_d.begin(1) + nentries + nentries/2);
		Generator.Gauss(mean_p.GetValue()+4.5, 0.5, data_d.begin(1) + nentries + nentries/2, data_d.end(1));

		// exponential
		Generator.Exp(tau_p.GetValue()+5.0, data_d.begin(0)+nentries,  data_d.end(0));

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << data_d[i] << std::endl;

		//-------------------------------------------------------
		// Bring data to host and suffle it to avoid biases

		hydra::copy(data_d,   data_h);

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(data_h.begin(), data_h.end(), g);

		hydra::copy(data_h, data_d);

		std::cout<< std::endl<< "Suffled data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << data_d[i] << std::endl;

		//filtering
		auto FILTER = [=] __hydra_dual__ (unsigned int n, double* x){
			return (x[0] > min) && (x[0] < max );
		};

		auto filter = hydra::wrap_lambda(FILTER);
		auto range  = hydra::apply_filter(data_d,  filter);

		std::cout<< std::endl<< "Filtered data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << range.begin()[i] << std::endl;


		//make model and fcn
		auto fcn   = hydra::make_loglikehood_fcn(model, range.begin(), range.end() );

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
		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(std::numeric_limits<unsigned int>::max(), 5));
		auto end_d = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		// output
		std::cout<<"Minimum: "<< minimum_d << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Fit] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		//--------------------------------------------
		//splot 2 components
		//hold weights
		hydra::multiarray<double,2,  hydra::device::sys_t> sweigts_d(range.size());

		//create splot
		auto splot  = hydra::make_splot(fcn.GetPDF() );

		start_d = std::chrono::high_resolution_clock::now();
		auto covar = splot.Generate( range.begin(), range.end(), sweigts_d.begin());
		end_d = std::chrono::high_resolution_clock::now();
		elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [sPlot] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		std::cout << "Covariance matrix "<< std::endl << covar<< std::endl << std::endl;
		std::cout<< std::endl << "sWeights:" << std::endl;
		for(size_t i = 0; i<10; i++)
			std::cout<<  "[" << i << "] :" <<  sweigts_d[i] << std::endl;
		std::cout<< std::endl << std::endl;

		//bring data to device
		hydra::multiarray< double,2, hydra::device::sys_t> data2_d(range.size());
		hydra::copy( range ,  data2_d );

        //_______________________________
		//histograms
		size_t nbins = 100;

        hydra::DenseHistogram< double, 1, hydra::device::sys_t> Hist_Data(nbins, min, max);

        start_d = std::chrono::high_resolution_clock::now();
        Hist_Data.Fill(data2_d.begin(0), data2_d.end(0));
        end_d = std::chrono::high_resolution_clock::now();
        elapsed_d = end_d - start_d;

        //time
        std::cout << "-----------------------------------------"<<std::endl;
        std::cout << "| [Histograming data] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
        std::cout << "-----------------------------------------"<<std::endl;

        hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Control(nbins, min, max);

        start_d = std::chrono::high_resolution_clock::now();
        Hist_Control.Fill(data2_d.begin(1), data2_d.end(1));
        end_d = std::chrono::high_resolution_clock::now();
        elapsed_d = end_d - start_d;

        //time
        std::cout << "-----------------------------------------"<<std::endl;
        std::cout << "| [Histograming control] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
        std::cout << "-----------------------------------------"<<std::endl;

        hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Control_1(nbins, min, max);

        start_d = std::chrono::high_resolution_clock::now();
        Hist_Control_1.Fill(data2_d.begin(1), data2_d.end(1), sweigts_d.begin(0) );
        end_d = std::chrono::high_resolution_clock::now();
        elapsed_d = end_d - start_d;

        //time
        std::cout << "-----------------------------------------"<<std::endl;
        std::cout << "| [Histograming control 1] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
        std::cout << "-----------------------------------------"<<std::endl;

        hydra::DenseHistogram<double, 1, hydra::device::sys_t> Hist_Control_2(nbins, min, max);

        start_d = std::chrono::high_resolution_clock::now();
        Hist_Control_2.Fill(data2_d.begin(1), data2_d.end(1), sweigts_d.begin(1) );
        end_d = std::chrono::high_resolution_clock::now();
        elapsed_d = end_d - start_d;

        //time
        std::cout << "-----------------------------------------"<<std::endl;
        std::cout << "| [Histograming control 2] GPU Time (ms) ="<< elapsed_d.count() <<std::endl;
        std::cout << "-----------------------------------------"<<std::endl;





#ifdef _ROOT_AVAILABLE_

        for(size_t bin=0; bin < nbins; bin++){

        	hist_data_dicriminating_d.SetBinContent(bin+1,  Hist_Data[bin] );
        	hist_data_control_d.SetBinContent(bin+1,  Hist_Control[bin] );
        	hist_control_1_d.SetBinContent(bin+1,  Hist_Control_1[bin] );
        	hist_control_2_d.SetBinContent(bin+1,  Hist_Control_2[bin] );

        }


		//draw fitted function
		for (size_t i=1 ; i<=100 ; i++) {
			double x = hist_fit_d.GetBinCenter(i);
	        hist_fit_d.SetBinContent(i, fcn.GetPDF()(x) );
		}
		hist_fit_d.Scale(hist_data_dicriminating_d.Integral()/hist_fit_d.Integral() );


#endif //_ROOT_AVAILABLE_

	}//device end



#ifdef _ROOT_AVAILABLE_

	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_1_d("canvas_1_d" ,"Distributions - Device", 500, 500);

	hist_data_dicriminating_d.Draw("hist");
	hist_fit_d.Draw("histsameC");
	hist_fit_d.SetLineColor(2);

	TCanvas canvas_2_d("canvas_2_d" ,"Distributions - Device", 1000, 500);
	canvas_2_d.Divide(2,1);
	canvas_2_d.cd(1);
	hist_control_1_d.Draw("hist");
	canvas_2_d.cd(2);
	hist_control_2_d.Draw("hist");


	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}
#endif /* PSEUDO_EXPERIMENT_INL_ */
