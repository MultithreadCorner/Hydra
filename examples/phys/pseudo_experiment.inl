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
#include <future>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/Algorithm.h>
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
#include <hydra/functions/BreitWignerNR.h>
#include <hydra/functions/Exponential.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Chebychev.h>
#include <hydra/Placeholders.h>

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
	hydra::Parameter  mean  = hydra::Parameter::Create().Name("Mean").Value( 5.0).Error(0.0001).Limits(4.9, 5.1);
	hydra::Parameter  sigma = hydra::Parameter::Create().Name("Sigma").Value(0.5).Error(0.0001).Limits(0.9, 1.1);

	//gaussian function evaluating on the first argument
	auto Gaussian_PDF = hydra::make_pdf( hydra::Gaussian<>(mean, sigma),
			hydra::AnalyticalIntegral<hydra::Gaussian<>>(data_min, data_max));

	//-------------------------------------------
	//Exponential
    //parameters
    auto  tau  = hydra::Parameter::Create().Name("Tau").Value(-0.2).Error(0.0001).Limits(-1.0, 0.0);

    //Background PDF
    auto Exponential_PDF = hydra::make_pdf(hydra::Exponential<>(tau),
    		 hydra::AnalyticalIntegral<hydra::Exponential<>>(data_min, data_max));

	//------------------

	//yields
	hydra::Parameter N_Exponential("N_Exponential", 5000, 100, 100 , nentries) ;
	hydra::Parameter N_Gaussian("N_Gaussian"      , 5000, 100, 100 , nentries) ;

	//make model
	auto discriminant_model = hydra::add_pdfs( {N_Gaussian, N_Exponential},
			Gaussian_PDF, Exponential_PDF);
	discriminant_model.SetExtended(1);

	//======================================================
	// 2) Breit-Wigner + Chebychev (dimension <1>)
	//-----------------
	// data range
	double obs_min   =  0.0;
	double obs_max   =  15.0;

	//Breit-Wigner

	//parameters
	hydra::Parameter  mass  = hydra::Parameter::Create().Name("Mass" ).Value(7.0).Error(0.0001).Limits(5.0,7.0);
	hydra::Parameter  width = hydra::Parameter::Create().Name("Width").Value(1.0).Error(0.0001).Limits(0.3,1.0);

	//Breit-Wigner function evaluating on the first argument
	auto BreitWigner_PDF = hydra::make_pdf( hydra::BreitWignerNR<>(mass, width ),
			hydra::AnalyticalIntegral<hydra::BreitWignerNR<>>(obs_min, obs_max));

    //-------------------------------------------

	//Chebychev

    //parameters
    auto  c0  = hydra::Parameter::Create("C_0").Value( 1.5).Error(0.0001).Limits( 1.0, 2.0);
    auto  c1  = hydra::Parameter::Create("C_1").Value( -0.2).Error(0.0001).Limits( 0.1, 0.3);
    auto  c2  = hydra::Parameter::Create("C_2").Value( 0.1).Error(0.0001).Limits( 0.01, 0.2);
    auto  c3  = hydra::Parameter::Create("C_3").Value( -0.2).Error(0.0001).Limits( 0.01, 0.2);

    //Polynomial function evaluating on the first argument
    auto Chebychev_PDF = hydra::make_pdf( hydra::Chebychev<3>(obs_min, obs_max, std::array<hydra::Parameter,4>{c0, c1, c2, c3}),
    		hydra::AnalyticalIntegral< hydra::Chebychev<3>>(obs_min, obs_max));

    //------------------
    //yields
	hydra::Parameter N_BreitWigner("N_BreitWigner" , 500,  100, 100 , nentries) ;
	hydra::Parameter N_Chebychev("N_Chebychev"     , 2000, 100, 100 , nentries) ;

	//make model
	auto observable_model = hydra::add_pdfs( {N_BreitWigner, N_Chebychev},
			BreitWigner_PDF, Chebychev_PDF);
	observable_model.SetExtended(1);

	//======================================================
	// Pseudo-experiment data sample generation
	//======================================================
	//

	//dataset
	hydra::multiarray<double,2, hydra::host::sys_t> dataset(nentries);

	//this scope will deploy the device backend to allocate memory and
	//generate the primary dataset using std multithread facility
	{
		//generator
		//dataset
		hydra::multiarray<double,2, hydra::device::sys_t> temp_dataset(3*nentries);

		//fill Gaussian component in a separated thread
		auto discrimant_handler = std::async(std::launch::async,
				[data_min, data_max, &discriminant_model, &temp_dataset]{

			    hydra::Random<> Generator;
				Generator.SetSeed(159);
				auto range = Generator.Sample(hydra::columns( temp_dataset, _0) ,
						data_min, data_max,
						discriminant_model.GetFunctor() );
				return range;
		} );

		//fill Exponential component in a separated thread
		auto observable_handler = std::async(std::launch::async,
				[obs_min, obs_max, &observable_model, &temp_dataset]{

				hydra::Random<> Generator;
				Generator.SetSeed(753);
				auto range = Generator.Sample(hydra::columns( temp_dataset, _1),
					obs_min, obs_max,
					observable_model.GetFunctor());
				return range;
		} );


		size_t ngen =0;

		//wait the sampling finishe before using is results
		discrimant_handler.wait();
		observable_handler.wait();

		auto discrimant_range = discrimant_handler.get();
		auto observable_range = observable_handler.get();


		if( (discrimant_range.size() <= nentries) ||
				(observable_range.size() <= nentries)	)
		{

			if( discrimant_range.size() <= observable_range.size())
				ngen = discrimant_range.size();
			else
				ngen = observable_range.size();

			hydra::copy(temp_dataset.begin(), temp_dataset.begin()+ngen, dataset.begin());
			dataset.erase(dataset.begin()+ngen+1, dataset.end());
		}
		else
		{
			ngen =nentries;
			hydra::copy(temp_dataset.begin(), temp_dataset.begin()+nentries, dataset.begin());

		}

		std::cout <<  "Dataset: "<< ngen<< std::endl;
		for(int i=0; i<10; i++)
			std::cout << temp_dataset[i] << std::endl;
	}

	//device
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_data_dicriminating("data_discriminating", "Discriminating variable", 100, data_min, data_max);
	TH1D   hist_data_observables("data_observables",    "Control Variable", 100, obs_min, obs_max);

	for(auto x: dataset){
		hist_data_dicriminating.Fill( hydra::get<0>(x) );
		hist_data_observables.Fill( hydra::get<1>(x) );
	}

#endif //_ROOT_AVAILABLE_


/* this scope will run a loop where:
 * 1- a new sample will be produced at same statistical level
 * 2- perform a splot to obtain a background free sample, which will contain negative weights
 * 3- perform a fit and store the results
 * 4- repeat the loop
 */
	{

		//boost_strapped data (bs-data)
        auto bs_range = hydra::boost_strapped_range(
        		dataset, std::chrono::system_clock::now().time_since_epoch().count());

        //bring the bs-data to the device
        hydra::multiarray<double,2, hydra::device::sys_t> dataset_device( bs_range.begin(), bs_range.end());

        //create fcn for sfit
        auto discriminanting_fcn = hydra::make_loglikehood_fcn(discriminant_model,
        		hydra::columns(dataset_device, _0) );

        //print level
        ROOT::Minuit2::MnPrint::SetLevel(3);
        hydra::Print::SetLevel(hydra::WARNING);

        //minimization strategy
        MnStrategy strategy(1);

		// create Migrad minimizer
		MnMigrad migrad(discriminanting_fcn, discriminanting_fcn.GetParameters().GetMnState()
				, strategy);

		std::cout<< discriminanting_fcn.GetParameters().GetMnState() << std::endl;

		// ... Minimize and profile the time

		auto start = std::chrono::high_resolution_clock::now();
		FunctionMinimum minimum =  FunctionMinimum(migrad(std::numeric_limits<unsigned int>::max(), 5));
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		// output
		std::cout<<"Minimum: "<< minimum << std::endl;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Fit Time] (ms) = " << elapsed.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		//--------------------------------------------
		//perform splot for two components
		//allocate memory to hold weights
		hydra::multiarray<double, 2, hydra::device::sys_t> sweigts_device( dataset_device.size() );

		//create splot
		auto splot  = hydra::make_splot( discriminanting_fcn.GetPDF() );

		start = std::chrono::high_resolution_clock::now();
		auto covar = splot.Generate( hydra::columns(dataset_device, _0), sweigts_device);
		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [sPlot Time] (ms) ="<< elapsed.count()  <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		std::cout << "Covariance matrix: "<< std::endl << covar << std::endl;

		std::cout<< std::endl << "sWeights:" << std::endl;

		for(size_t i = 0; i<10; i++)
			std::cout << "[" << i << "] :"
			          << sweigts_device[i] << std::endl
			          << std::endl;

		//
		auto observable_fcn = hydra::make_loglikehood_fcn(observable_model,
				hydra::columns(dataset_device, _1), hydra::columns(sweigts_device, _0) );


	}


#ifdef _ROOT_AVAILABLE_

	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_1("canvas_1" ,"Dataset", 1000, 500);
	canvas_1.Divide(2);
    canvas_1.cd(1);
	hist_data_dicriminating.Draw("hist");
	canvas_1.cd(2);
	hist_data_observables.Draw("hist");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}
#endif /* PSEUDO_EXPERIMENT_INL_ */
