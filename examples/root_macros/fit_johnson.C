/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * fit_johnson.C
 *
 *  Created on: 17/10/2018
 *      Author: Davide Brundu
 */

/*!
 * \example fit_johnson.C
 */

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>


/**
 *
 */
#ifndef HYDRA_HOST_SYSTEM
#define HYDRA_HOST_SYSTEM CPP
#endif

#ifndef HYDRA_DEVICE_SYSTEM
#define HYDRA_DEVICE_SYSTEM TBB
#endif
//this lib
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/Random.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/Filter.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/JohnsonSUShape.h>
#include <hydra/DenseHistogram.h>
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
#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

using namespace ROOT::Minuit2;
using namespace hydra::arguments;


declarg(xvar, double)


void fit_johnson(size_t nentries=500000 )
{


	//-----------------
	// some definitions
	double min   = -15.0;
	double max   =  5.0;

	//parameters
	auto Gamma  = hydra::Parameter::Create("gamma" ).Value(3.0 ).Error(0.0001).Limits( -1.0, 5.0  );
	auto Delta  = hydra::Parameter::Create("delta" ).Value(2.0 ).Error(0.0001).Limits(0.5, 2.5);
	auto Xi     = hydra::Parameter::Create("xi"    ).Value(1.1).Error(0.0001).Limits( 0.0, 2.0  );
	auto Lambda = hydra::Parameter::Create("lambda").Value(1.5 ).Error(0.0001).Limits(0.5, 2.5);


	//johnson function evaluating on argument one
	hydra::JohnsonSU<xvar> johnson(Gamma, Delta, Xi, Lambda);

	//make model (pdf with analytical integral)
	auto model = hydra::make_pdf(johnson,
			hydra::AnalyticalIntegral< hydra::JohnsonSU<xvar> >(min, max) );


	//------------------------

	TH1D* hist_johnson_d        = new TH1D("johnson_d",       "johnson", 100, min, max);
	TH1D* hist_fitted_johnson_d = new TH1D("fitted_johnson_d","johnson", 100, min, max);


	//begin raii scope
	{
		//1D device buffer
		hydra::device::vector<double>  dataset(nentries);

		//-------------------------------------------------------
		//generate johnson data
		hydra::copy( hydra::random_range( johnson, 159753, dataset.size() ), dataset);

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << dataset[i] << std::endl;

		//filtering
		auto filter = hydra::wrap_lambda(
				[=] __hydra_dual__ (xvar x){
				return (x > min) && (x < max );
		});

		auto range  = hydra::filter(dataset,  filter);

		std::cout<< std::endl<< "Filtered data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << range[i] << std::endl;

		auto fcn   = hydra::make_loglikehood_fcn(model, range );

		//-------------------------------------------------------
		//fit

		ROOT::Minuit2::MnPrint::SetLevel(3);
		hydra::Print::SetLevel(hydra::WARNING);

		//minimization strategy
		MnStrategy strategy(2);

		//create Migrad minimizer
		MnMigrad migrad_d(fcn, fcn.GetParameters().GetMnState() ,  strategy);

		//print parameters before fitting
		std::cout<<fcn.GetParameters().GetMnState()<<std::endl;

		//Minimize and profile the time
		auto start_d = std::chrono::high_resolution_clock::now();

		FunctionMinimum minimum_d =  FunctionMinimum(migrad_d(5000, 1));

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//print minuit result
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		//time
		std::cout << "-----------------------------------------"  << std::endl;
		std::cout << "| Device Time (ms) ="<< elapsed_d.count()   << std::endl;
		std::cout << "-----------------------------------------"  << std::endl;

		hydra::DenseHistogram<double,1,  hydra::device::sys_t> Hist_Data(100, min, max);
		Hist_Data.Fill( range);



		//draw data
		for(size_t i=0;  i<100; i++)
		{
			hist_johnson_d->SetBinContent(i+1, Hist_Data.GetBinContent(i)  );
		}

		//draw fitted function
		hist_fitted_johnson_d->Sumw2();
		for (size_t i=0 ; i<=100 ; i++) 
		{
			double x = hist_fitted_johnson_d->GetBinCenter(i);
			hist_fitted_johnson_d->SetBinContent(i, fcn.GetPDF()(x) );
		}

		hist_fitted_johnson_d->Scale(hist_johnson_d->Integral()/hist_fitted_johnson_d->Integral() );

	}//end device scope


	//draw histograms
	TCanvas* canvas_d = new TCanvas("canvas_d" ,"Distributions - Device", 500, 500);
	hist_johnson_d->Draw("hist");
	hist_fitted_johnson_d->SetLineColor(2);
	hist_fitted_johnson_d->Draw("histsameC");


}
