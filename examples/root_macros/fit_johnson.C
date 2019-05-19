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
#include <hydra/FunctionWrapper.h>
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


void fit_johnson(size_t nentries=500000 )
{


	//-----------------
	// some definitions
	double min   = -5.0;
	double max   =  5.0;
	double mean  =  0.0;
	double sigma =  1.5;
	double width =  0.5;
	double nu    =  0.0;
	double tau   =  0.0;

	//generator
	hydra::Random<> Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//parameters
	hydra::Parameter  mean_p  = hydra::Parameter::Create().Name("gamma").Value(0.0).Error(0.0001).Limits(-1.0, 1.0);
	hydra::Parameter  width_p = hydra::Parameter::Create().Name("delta").Value(1.0).Error(0.0001).Limits(0.001, 100.5);
	hydra::Parameter  nu_p    = hydra::Parameter::Create().Name("xi").Value(0.05).Error(0.0001).Limits(-5.0, 5.0);
	hydra::Parameter  tau_p   = hydra::Parameter::Create().Name("lambda").Value(1.0).Error(0.0001).Limits(0.001, 100.5);


	//johnson function evaluating on argument one
	hydra::JohnsonSUShape<> johnson(mean_p, width_p, nu_p, tau_p);

	//make model (pdf with analytical integral)
	auto model = hydra::make_pdf(johnson, hydra::JohnsonSUShapeAnalyticalIntegral(min, max) );


	//------------------------

	TH1D* hist_johnson_d        = new TH1D("johnson_d",       "johnson", 100, min, max);
	TH1D* hist_fitted_johnson_d = new TH1D("fitted_johnson_d","johnson", 100, min, max);


	//begin raii scope
	{
		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);

		//-------------------------------------------------------
		//generate gaussian data
		Generator.Gauss(mean, sigma, data_d.begin(), data_d.end());

		std::cout<< std::endl<< "Generated data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << data_d[i] << std::endl;

		//filtering
		auto filter = hydra::wrap_lambda(
				[=] __hydra_dual__ (unsigned int n, double* x){
				return (x[0] > min) && (x[0] < max );
		});

		auto range  = hydra::apply_filter(data_d,  filter);

		std::cout<< std::endl<< "Filtered data:"<< std::endl;
		for(size_t i=0; i<10; i++)
			std::cout << "[" << i << "] :" << range.begin()[i] << std::endl;

		auto fcn   = hydra::make_loglikehood_fcn(model, range.begin(), range.end());

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
		Hist_Data.Fill( range.begin(), range.end() );



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
