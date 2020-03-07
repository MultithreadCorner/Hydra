/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * simultaneous_fit.inl
 *
 *  Created on: 05/02/2020
 *      Author: Davide Brundu
 */

#ifndef SIMULTANEOUS_FIT_INL_
#define SIMULTANEOUS_FIT_INL_

/**
 * \example simultaneous_fit.inl
 *
 * This example implements a simultaneous fit using
 * three unidimensional Gaussian functions,
 * with the mean as a common parameter
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
#include <hydra/host/System.h>
#include <hydra/detail/ArgumentTraits.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Distance.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/SimultaneousFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/functions/Gaussian.h>

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

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */

using namespace ROOT::Minuit2;


int main(int argv, char** argc) {
	size_t nentries = 0;
  hydra::Print::SetLevel(hydra::WARNING);

	try {

		TCLAP::CmdLine cmd("Command line arguments for ", '=');

		TCLAP::ValueArg < size_t
				> EArg("n", "number-of-events", "Number of events", true, 10e6,
						"size_t");
		cmd.add(EArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries = EArg.getValue();

	} catch (TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
				<< std::endl;
	}
//-----------------
  size_t x_nentries = nentries;
	size_t y_nentries = nentries/2;
	size_t z_nentries = nentries/2;

	double min = -30.0;
	double max =  30.0;

	//Parameters for X direction
	auto commonmean  = hydra::Parameter::Create("commonmean" ).Value(0.0).Error(0.0001).Limits(-1.0, 1.0);
	auto xsigma      = hydra::Parameter::Create("SigmaX").Value(1.0).Error(0.0001).Limits(0.01, 1.5);
	auto ysigma      = hydra::Parameter::Create("SigmaY").Value(2.0).Error(0.0001).Limits(0.01, 3.5);
	auto zsigma      = hydra::Parameter::Create("SigmaZ").Value(3.0).Error(0.0001).Limits(0.01, 5.5);


		//Gaussian distribution for X direction
  hydra::Gaussian<double> xgauss(commonmean, xsigma);
	auto xmodel  = hydra::make_pdf(xgauss, hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );

	hydra::Gaussian<double> ygauss(commonmean, ysigma);
	auto ymodel  = hydra::make_pdf(ygauss, hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );

	hydra::Gaussian<double> zgauss(commonmean, zsigma);
	auto zmodel  = hydra::make_pdf(zgauss, hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );

 { //device scope

	 //1D device buffer
	 hydra::device::vector<double> xdataset(x_nentries);
	 hydra::device::vector<double> ydataset(y_nentries);
	 hydra::device::vector<double> zdataset(z_nentries);

	 //gaussian ranges
	 auto gauss_x_range = hydra::random_gauss_range(commonmean.GetValue()+0.1, xsigma.GetValue()+0.1, 159753);
	 auto gauss_y_range = hydra::random_gauss_range(commonmean.GetValue()+0.1, ysigma.GetValue()-0.2, 159754);
	 auto gauss_z_range = hydra::random_gauss_range(commonmean.GetValue()+0.1, zsigma.GetValue()+0.5, 159755);

	 std::cout << "ACTUAL PARAMETERS" << std::endl;
	 	std::cout << "SigmaX : " << xsigma.GetValue()+0.1 << std::endl;
		std::cout << "Mean : "   << commonmean.GetValue()+0.1 << std::endl;
	 	std::cout << "SigmaY : " << ysigma.GetValue()-0.2 << std::endl;
	 	std::cout << "SigmaZ : " << zsigma.GetValue()+0.5 << std::endl;

	 hydra::copy(gauss_x_range.begin(), gauss_x_range.begin()+x_nentries, xdataset.begin());
	 hydra::copy(gauss_y_range.begin(), gauss_y_range.begin()+y_nentries, ydataset.begin());
	 hydra::copy(gauss_z_range.begin(), gauss_z_range.begin()+z_nentries, zdataset.begin());


	 auto xfcn    = hydra::make_loglikehood_fcn(xmodel, xdataset);
	 auto yfcn    = hydra::make_loglikehood_fcn(ymodel, ydataset);
	 auto zfcn    = hydra::make_loglikehood_fcn(zmodel, zdataset);

	 auto sim_fcn = hydra::make_simultaneous_fcn(xfcn, yfcn, zfcn);


	 //-------------------------------------------------------
	 //fit

	 ROOT::Minuit2::MnPrint::SetLevel(3);
	 hydra::Print::SetLevel(hydra::WARNING);

	 //minimization strategy
	 MnStrategy strategy(2);

	 //create Migrad minimizer
	 MnMigrad migrad(sim_fcn, sim_fcn.GetParameters().GetMnState() , strategy);

	 //print parameters before fitting
	 std::cout << sim_fcn.GetParameters().GetMnState() << std::endl;

	 //Minimize and profile the time
	 auto start = std::chrono::high_resolution_clock::now();

	 FunctionMinimum minimum = FunctionMinimum( migrad(std::numeric_limits<unsigned int>::max(), 5));

	 auto stop  = std::chrono::high_resolution_clock::now();

	 std::chrono::duration<double, std::milli> elapsed = stop - start;

	 //print minuit result
	 std::cout << " minimum: " << minimum << std::endl;

	 //time
	 std::cout << "-----------------------------------------"<<std::endl;
	 std::cout << "| Time (ms) ="<< elapsed.count()    <<std::endl;
	 std::cout << "-----------------------------------------"<<std::endl;

 } //end device scope

	return 0;

}
#endif /* SIMULTANEOUS_FIT_INL_ */
