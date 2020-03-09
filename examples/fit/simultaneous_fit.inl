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
 *  Created on: 05/03/2020
 *      Author: Davide Brundu
 *      Note: updated by A.A.A.Jr in 07/03/2020.
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
#include <hydra/omp/System.h>
#include <hydra/tbb/System.h>

#include <hydra/Function.h>
#include <hydra/Filter.h>
#include <hydra/Algorithm.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Distance.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/functions/Gaussian.h>

//Minuit2
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

	double min = -6.0;
	double max =  6.0;

	//Parameters
	auto mean   = hydra::Parameter::Create("mean" ).Value(0.0).Error(0.0001).Limits(-1.0, 1.0);

	//Gaussian distribution for X direction
	auto xsigma = hydra::Parameter::Create("sigma-x").Value(1.0).Error(0.0001).Limits(0.01, 1.5);

	auto xmodel  = hydra::make_pdf(hydra::Gaussian<double>(mean, xsigma),
			hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );

	//Gaussian distribution for  Y direction
	auto ysigma = hydra::Parameter::Create("sigma-y").Value(2.0).Error(0.0001).Limits(0.01, 3.5);

	auto ymodel  = hydra::make_pdf(hydra::Gaussian<double>(mean, ysigma),
			hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );


	//Gaussian distribution  Z direction
	auto zsigma = hydra::Parameter::Create("sigma-z").Value(3.0).Error(0.0001).Limits(0.01, 5.5);

	auto zmodel  = hydra::make_pdf(hydra::Gaussian<double>(mean, zsigma),
			hydra::AnalyticalIntegral< hydra::Gaussian<double> >(min, max) );

	{ //device scope

		auto filter_entries = [min, max]__hydra_dual__(double x){
			return bool (x >= min) && (x< max);
		};

		//make datasets
		// x in omp memory space
		auto gauss_x_range = hydra::random_gauss_range(mean.GetValue()+0.1, xsigma.GetValue()+0.1, 0xd73ad43c3);

		auto xdata= hydra::omp::vector<double>(nentries);

        hydra::copy(gauss_x_range.begin(), gauss_x_range.begin()+ xdata.size(), xdata.begin());

        auto xrange = hydra::filter(xdata, filter_entries);

        // y in tbb memory space
        auto gauss_y_range = hydra::random_gauss_range(mean.GetValue()+0.1, ysigma.GetValue()+0.1, 0xff4e48b27);

        auto ydata= hydra::tbb::vector<double>(nentries);

        hydra::copy(gauss_y_range.begin(), gauss_y_range.begin()+ ydata.size(), ydata.begin());

        auto yrange = hydra::filter(ydata, filter_entries);


        // y in tbb memory space
        auto gauss_z_range = hydra::random_gauss_range(mean.GetValue()+0.1, zsigma.GetValue()+0.1, 0xff4e48c31 );

        auto zdata= hydra::device::vector<double>(nentries);

        hydra::copy(gauss_z_range.begin(), gauss_z_range.begin()+ zdata.size(), zdata.begin());

        auto zrange = hydra::filter(zdata, filter_entries);


        //make the single fcns
		auto xfcn    = hydra::make_loglikehood_fcn(xmodel, xrange);
		auto yfcn    = hydra::make_loglikehood_fcn(ymodel, yrange);
		auto zfcn    = hydra::make_loglikehood_fcn(zmodel, zrange);

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
