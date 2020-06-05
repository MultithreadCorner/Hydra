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
 * basic_distributions.inl
 *
 *  Created on: Jul 18, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASIC_DISTRIBUTIONS_INL_
#define BASIC_DISTRIBUTIONS_INL_

/**
 * \example basic_distributions.inl
 *
 */

/*
#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>

//command line
#include <tclap/CmdLine.h>

//this lib
#include <hydra/device/System.h>
#include <hydra/host/System.h>
#include <hydra/Function.h>
#include <hydra/Lambda.h>
#include <hydra/Random.h>
#include <hydra/Algorithm.h>
*/
/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
/*
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


	//generator
	hydra::Random<>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );


	//device
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_uniform_d("uniform_d",   "Uniform",     100, -6.0, 6.0);
	TH1D hist_gaussian_d("gaussian_d", "Gaussian",    100, -6.0, 6.0);
	TH1D hist_exp_d("exponential_d",   "Exponential", 100,  0.0, 5.0);
	TH1D hist_bw_d("breit_wigner_d",   "Breit-Wigner",100,  0.0, 5.0);

#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::device::vector<double>  data_d(nentries);
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//uniform
		Generator.Uniform(-5.0, 5.0, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Uniform > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_uniform_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_d)
			hist_gaussian_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//exponential
		Generator.Exp(1.0, data_d);
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Exp > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_exp_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//breit-wigner
		Generator.BreitWigner(2.0, 0.2, data_d.begin(), data_d.end());
		hydra::copy(data_d,  data_h);

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::BreitWigner > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_bw_d.Fill( value);
#endif //_ROOT_AVAILABLE_
	}




#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	canvas_d.Divide(2,2);
	canvas_d.cd(1); hist_uniform_d.Draw("hist");
	canvas_d.cd(2); hist_gaussian_d.Draw("hist");
	canvas_d.cd(3); hist_exp_d.Draw("hist");
	canvas_d.cd(4); hist_bw_d.Draw("hist");



	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}

*/
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
#include <hydra/functions/UniformShape.h>
#include <hydra/functions/TriangularShape.h>
#include <hydra/functions/TrapezoidalShape.h>

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


auto data = hydra::device::vector< double>(10, .0);

    //Gaussian distribution
	//Parameters
	auto mean  = hydra::Parameter::Create("mean" ).Value(0.0);
	auto sigma = hydra::Parameter::Create("sigma").Value(0.25);

	auto gauss     = hydra::Gaussian<xvar>(mean, sigma);

    for(auto x: data)
    	//x.dummy;
    	std::cout << gauss(x) << std::endl;

	//LogNormal distribution
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

	//Uniform
	auto A = hydra::Parameter::Create().Name("A").Value(-5.0);
	auto B = hydra::Parameter::Create().Name("B").Value(-1.5);
	auto C = hydra::Parameter::Create().Name("C").Value( 1.5);
	auto D = hydra::Parameter::Create().Name("D").Value( 5.0);

	auto uniform   = hydra::UniformShape<xvar>(A,D);
	auto triangle  = hydra::TriangularShape<xvar>(A,B,D);
	auto trapezoid = hydra::TrapezoidalShape<xvar>(A,B,C,D);

	hydra_thrust::default_random_engine engine;

#ifdef _ROOT_AVAILABLE_

	TH1D hist_gauss("hist_gauss", "hydra::Gaussian<xvar>"   , 100,-8.0, 8.0);
	TH1D hist_lognormal("hist_lognormal", "hydra::LogNormal<xvar>"   , 100,0.0, 2.5);
    TH1D hist_bigauss("hist_bigauss", "hydra::BifurcatedGaussian<xvar>"   , 100,-8.0, 8.0);
	TH1D hist_exp("hist_exp" , "hydra::Exponential<xvar>", 100, 0.0, 10.0);
	TH1D hist_bw("hist_bw"  , "hydra::BreitWignerNR<xvar>", 100, 0.0, 10.0);
	TH1D hist_chi("hist_chi" , "hydra::ChiSquare<xvar>", 100, 0.0, 10.0);
	TH1D hist_johnson_su("hist_su"  , "hydra::JohnsonSU<xvar>", 100, -5.0, 1.0);
	TH1D hist_uniform("hist_uniform"  , "hydra::UniformShape<xvar>", 100, -6.0, 6.0);
	hist_uniform.SetMinimum(0.0);
	TH1D hist_triangle("hist_triangle"  , "hydra::TriangularShape<xvar>", 100, -6.0, 6.0);
	TH1D hist_trapezoid("hist_trapezoid"  , "hydra::TrapezoidalShape<xvar>", 100, -6.0, 6.0);


	for(size_t i=0; i<nentries; i++)
	{
		auto gauss_dist      = hydra::Distribution<hydra::Gaussian<xvar>>();
		auto lognormal_dist  = hydra::Distribution<hydra::LogNormal<xvar>>();
		auto bigauss_dist    = hydra::Distribution<hydra::BifurcatedGaussian<xvar>>();
		auto   exp_dist      = hydra::Distribution<hydra::Exponential<xvar>>();
		auto   bw_dist       = hydra::Distribution<hydra::BreitWignerNR<xvar>>();
		auto   chi2_dist     = hydra::Distribution<hydra::ChiSquare<xvar>>();
		auto johnson_su_dist = hydra::Distribution<hydra::JohnsonSU<xvar>>();
		auto uniform_dist    = hydra::Distribution<hydra::UniformShape<xvar>>();
		auto triangle_dist   = hydra::Distribution<hydra::TriangularShape<xvar>>();
		auto trapezoid_dist  = hydra::Distribution<hydra::TrapezoidalShape<xvar>>();

		hist_gauss.Fill( gauss_dist(engine, {0.0, 1.5} ));
		hist_lognormal.Fill( lognormal_dist(engine, lognormal ));
		hist_bigauss.Fill( bigauss_dist(engine, bigauss));
		hist_exp.Fill( exp_dist(engine, exp));
		hist_bw.Fill( bw_dist(engine, bw));
		hist_chi.Fill( chi2_dist(engine, chi2));
		hist_johnson_su.Fill( johnson_su_dist(engine, johnson_su ));
		hist_uniform.Fill( uniform_dist(engine,  uniform));
		hist_triangle.Fill(triangle_dist(engine,triangle));
		hist_trapezoid.Fill(trapezoid_dist(engine,trapezoid));

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

	TCanvas canvas_uniform("canvas_uniform" ,"hydra::UniformShape", 500, 500);
	hist_uniform.Draw("hist");

	TCanvas canvas_triangle("canvas_triangle" ,"hydra::TriangularShape", 500, 500);
	hist_triangle.Draw("hist");

	TCanvas canvas_trapezoid("canvas_trapezoid" ,"hydra::TrapezoidalShape", 500, 500);
	hist_trapezoid.Draw("hist");



	myapp->Run();

#endif //_ROOT_AVAILABLE_




	return 0;
}



#endif /* BASIC_DISTRIBUTIONS_INL_ */
