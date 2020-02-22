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
 * quick_test.inl
 *
 *  Created on: 13/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef QUICK_TEST_INL_
#define QUICK_TEST_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//hydra
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Lambda.h>
#include <hydra/multivector.h>
#include <hydra/Parameter.h>
#include <hydra/functions/Gaussian.h>
#include <hydra/functions/Exponential.h>
#include <hydra/functions/BifurcatedGaussian.h>
#include <hydra/functions/BreitWignerNR.h>
#include <hydra/functions/ChiSquare.h>
#include <hydra/functions/Chebychev.h>
#include <hydra/functions/JohnsonSUShape.h>
#include <hydra/functions/LogNormal.h>

#include <hydra/detail/external/hydra_thrust/random.h>

#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace hydra::arguments;

declarg(xvar, double)

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
		std::cerr << " error: "  << e.error()
				  << " for arg " << e.argId()
				  << std::endl;
	}


    //Gaussian distribution
	//Parameters
	auto mean  = hydra::Parameter::Create("mean" ).Value(0.0);
	auto sigma = hydra::Parameter::Create("sigma").Value(0.25);

	auto gauss     = hydra::Gaussian<xvar>(mean, sigma);
	auto lognormal = hydra::LogNormal<xvar>(mean, sigma);


    //BifurcatedGaussian distribution
	//Parameters
	auto sigma_left = hydra::Parameter::Create("sigma left").Value(2.0);
	auto sigma_rigt = hydra::Parameter::Create("sigma rigt").Value(1.0);

	auto bigauss = hydra::BifurcatedGaussian<xvar>(mean, sigma_left, sigma_rigt);

	//Exponential distribution
	auto tau  = hydra::Parameter::Create("mean" ).Value(1.0);
	auto exp  = hydra::Exponential<xvar>(tau);

	//Breit-Wigner
	auto mass  = hydra::Parameter::Create().Name("mass" ).Value(5.0);
	auto width = hydra::Parameter::Create().Name("width").Value(0.5);
	auto bw = hydra::BreitWignerNR<xvar>(mass, width );

	//ChiSquare
	auto ndof  = hydra::Parameter::Create().Name("ndof" ).Value(2.0);
	auto chi2 = hydra::ChiSquare<xvar>(ndof);

	//JohnsonSU
	auto delta  = hydra::Parameter::Create().Name("delta" ).Value(2.0);
	auto lambda = hydra::Parameter::Create().Name("lambda").Value(1.5);
	auto gamma  = hydra::Parameter::Create().Name("gamma" ).Value(3.0);
	auto xi     = hydra::Parameter::Create().Name("xi").Value(1.1);

	auto johnson_su = hydra::JohnsonSU<xvar>(gamma, delta, xi, lambda);


	hydra_thrust::default_random_engine engine;

#ifdef _ROOT_AVAILABLE_

	TH1D hist_gauss("hist_gauss", "hydra::Gaussian<xvar>"   , 100,-8.0, 8.0);
	TH1D hist_lognormal("hist_lognormal", "hydra::LogNormal<xvar>"   , 100,0.0, 2.5);
    TH1D hist_bigauss("hist_bigauss", "hydra::BifurcatedGaussian<xvar>"   , 100,-8.0, 8.0);
	TH1D   hist_exp("hist_exp" , "hydra::Exponential<xvar>", 100, 0.0, 10.0);
	TH1D    hist_bw("hist_bw"  , "hydra::BreitWignerNR<xvar>", 100, 0.0, 10.0);
	TH1D   hist_chi("hist_chi" , "hydra::ChiSquare<xvar>", 100, 0.0, 10.0);
	TH1D   hist_johnson_su("hist_su"  , "hydra::JohnsonSU<xvar>", 100, -5.0, 1.0);


	for(size_t i=0; i<nentries; i++)
	{
		auto gauss_dist   = hydra::Distribution<hydra::Gaussian<xvar>>();
		auto lognormal_dist = hydra::Distribution<hydra::LogNormal<xvar>>();
		auto bigauss_dist = hydra::Distribution<hydra::BifurcatedGaussian<xvar>>();
		auto   exp_dist   = hydra::Distribution<hydra::Exponential<xvar>>();
		auto   bw_dist    = hydra::Distribution<hydra::BreitWignerNR<xvar>>();
		auto   chi2_dist  = hydra::Distribution<hydra::ChiSquare<xvar>>();
		auto johnson_su_dist = hydra::Distribution<hydra::JohnsonSU<xvar>>();

		hist_gauss.Fill( gauss_dist(engine, {0.0, 1.5} ));
		hist_lognormal.Fill( lognormal_dist(engine, lognormal ));
		hist_bigauss.Fill( bigauss_dist(engine, bigauss));
		hist_exp.Fill( exp_dist(engine, exp));
		hist_bw.Fill( bw_dist(engine, bw));
		hist_chi.Fill( chi2_dist(engine, chi2));
		hist_johnson_su.Fill( johnson_su_dist(engine, johnson_su ));

	}

	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_gauss("canvas_gauss" ,"hydra::Gaussian", 500, 500);
	hist_gauss.Draw("hist");

	TCanvas canvas_lognormal("canvas_lognormal" ,"hydra::LogNormal", 500, 500);
	hist_lognormal.Draw("hist");

	TCanvas canvas_bigauss("canvas_bigauss" ,"hydra::BifurcatedGaussian", 500, 500);
	hist_bigauss.Draw("hist");

	TCanvas canvas_exp("canvas_exp" ,"hydra::Exponential", 500, 500);
	hist_exp.Draw("hist");

	TCanvas canvas_bw("canvas_bw" ,"hydra::BreitWignerNR", 500, 500);
	hist_bw.Draw("hist");

	TCanvas canvas_chi("canvas_chi" ,"hydra::ChiSquare", 500, 500);
	hist_chi.Draw("hist");

	TCanvas canvas_johnson_su("canvas_chi" ,"hydra::JohnsonSU", 500, 500);
	hist_johnson_su.Draw("hist");



	myapp->Run();

#endif //_ROOT_AVAILABLE_




	return 0;
}



#endif /* QUICK_TEST_INL_ */
