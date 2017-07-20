/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * sample_distribution.inl
 *
 *  Created on: 20/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SAMPLE_DISTRIBUTION_INL_
#define SAMPLE_DISTRIBUTION_INL_

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
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/Copy.h>

/*-------------------------------------
 * Include classes from ROOT to fill
 * and draw histograms and plots.
 *-------------------------------------
 */
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH3D.h>
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





	//Gaussian 1
	double mean1   = -2.0;
	double sigma1  =  1.0;

	auto GAUSSIAN1 =  [=] __host__ __device__ (double* x ){

		double g = 1.0;

		for(size_t i=0; i<3; i++){

			double m2 = (x[i] - mean1 )*(x[i] - mean1 );
			double s2 = sigma1*sigma1;
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}

		return g;
	};

	auto gaussian1 = hydra::wrap_lambda( GAUSSIAN1 );

	//Gaussian 2
	double mean2   =  2.0;
	double sigma2  =  1.0;
	auto GAUSSIAN2 =  [=] __host__ __device__ (double* x ){

		double g = 1.0;

		for(size_t i=0; i<3; i++){

			double m2 = (x[i] - mean2 )*(x[i] - mean2 );
			double s2 = sigma2*sigma2;
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}

		return g;
	};

	auto gaussian2 = hydra::wrap_lambda( GAUSSIAN2 );

	//sum of gaussians
	auto gaussian = gaussian1 + gaussian2;

	//generator
	hydra::Random<thrust::random::default_random_engine>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	double max[3]{6.0, 6.0, 6.0};
	double min[3]{-6.0, -6.0, -6.0};

	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH3D hist_d("hist_d",   "3D Double Gaussian - Device",  100, -6.0, 6.0);
	TH3D hist_h("hist_h",   "3D Double Gaussian - Host"  ,  100, -6.0, 6.0);

#endif //_ROOT_AVAILABLE_

	//3D buffer
	typedef hydra::tuple<double, double, double> row_t;
	typedef hydra::device::vector<row_t> table_d;
	typedef hydra::host::vector<row_t> table_h;
	typedef hydra::multivector<table_d> dataset_d;
	typedef hydra::multivector<table_h> dataset_h;

    //device
	{
		dataset_d data_d(nentries);
		//-------------------------------------------------------
		//
		auto Generator.Sample(hydra::device::sys_t, Gaussians, min, max, nentries );

		hydra::copy(data_d.begin(), data_d.end(), data_h.begin());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Uniform > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_uniform_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_d.begin(), data_d.end());
		hydra::copy(data_d.begin(), data_d.end(), data_h.begin());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_d)
			hist_gaussian_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//exponential
		Generator.Exp(1.0, data_d.begin(), data_d.end());
		hydra::copy(data_d.begin(), data_d.end(),data_h.begin());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Exp > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_exp_d.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//breit-wigner
		Generator.BreitWigner(2.0, 0.2, data_d.begin(), data_d.end());
		hydra::copy(data_d.begin(), data_d.end(), data_h.begin());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::BreitWigner > [" << i << "] :" << data_d[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_bw_d.Fill( value);
#endif //_ROOT_AVAILABLE_
	}



	//host
	//------------------------
#ifdef _ROOT_AVAILABLE_

	TH1D hist_uniform_h("uniform_h",   "Uniform",     100, -6.0, 6.0);
	TH1D hist_gaussian_h("gaussian_h", "Gaussian",    100, -6.0, 6.0);
	TH1D hist_exp_h("exponential_h",   "Exponential", 100,  0.0, 5.0);
	TH1D hist_bw_h("breit_wigner_h",   "Breit-Wigner",100,  0.0, 5.0);

#endif //_ROOT_AVAILABLE_

	{
		//1D device buffer
		hydra::host::vector<double>    data_h(nentries);

		//-------------------------------------------------------
		//uniform
		Generator.Uniform(-5.0, 5.0, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Uniform > [" << i << "] :" << data_h[i]<< std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_uniform_h.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//gaussian
		Generator.Gauss(0.0, 1.0, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Gauss > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_gaussian_h.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//exponential
		Generator.Exp(1.0, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::Exp > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_exp_h.Fill( value);
#endif //_ROOT_AVAILABLE_

		//-------------------------------------------------------
		//breit-wigner
		Generator.BreitWigner(2.0, 0.2, data_h.begin(), data_h.end());

		for(size_t i=0; i<10; i++)
			std::cout << "< Random::BreitWigner > [" << i << "] :" << data_h[i] << std::endl;

#ifdef _ROOT_AVAILABLE_
		for(auto value : data_h)
			hist_bw_h.Fill( value);
#endif //_ROOT_AVAILABLE_
	}


	/*
	//-----------------
	// two gaussians hit-and-miss
	//-----------------

    //gaussian one
	std::array<GReal_t, 2>  means1  ={2.0, 2.0 };
	std::array<GReal_t, 2>  sigmas1 ={1.5, 0.5 };
	Gauss<2> Gaussian1(means1, sigmas1 );

	//gaussian two
	std::array<GReal_t, 2>  means2  ={-2.0, -2.0 };
	std::array<GReal_t, 2>  sigmas2 ={0.5, 1.5 };
	Gauss<2> Gaussian2(means2, sigmas2 );

	auto Gaussians = Gaussian1+Gaussian2;

	//2D range
	std::array<GReal_t, 2>  min  ={-5.0, -5.0 };
	std::array<GReal_t, 2>  max  ={ 5.0,  5.0 };


	TH2D hist_gaussians_DEVICE("gaussians_CCP_OMP_DEVICE", "Sum of gaussians (DEVICE)", 100, -6, 6, 100, -6, 6);

		{   auto start = std::chrono::high_resolution_clock::now();
			auto gaussians_data_d = Generator.Sample<hydra::device>(Gaussians,min, max, nentries );
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			//time
			std::cout << "-----------------------------------------"<<std::endl;
			std::cout << "| 2D sampling device Time (s) ="<< GReal_t(elapsed.count())/1000000 <<std::endl;
			std::cout << "-----------------------------------------"<<std::endl;

			hydra::mc_host_vector<thrust::tuple<GReal_t, GReal_t>> gaussians_data_h( gaussians_data_d);

			for(auto t:gaussians_data_h )
			{
				GReal_t x= thrust::get<0>(t);
				GReal_t y= thrust::get<1>(t);
				hist_gaussians_DEVICE.Fill(x,y);
			}

		}

		TH2D hist_gaussians_HOST("gaussians_CCP_OMP_HOST", "Sum of gaussians (HOST)", 100, -6, 6, 100, -6, 6);
		{
			auto start = std::chrono::high_resolution_clock::now();
			auto gaussians_data_h = Generator.Sample<hydra::host>(Gaussians,min, max, nentries );
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			//time
			std::cout << "-----------------------------------------"<<std::endl;
			std::cout << "| 2D sampling host Time (s) ="<< GReal_t(elapsed.count())/1000000 <<std::endl;
			std::cout << "-----------------------------------------"<<std::endl;

			for(auto t:gaussians_data_h )
			{
				GReal_t x= thrust::get<0>(t);
				GReal_t y= thrust::get<1>(t);
				hist_gaussians_HOST.Fill(x,y);
			}

		}


	//-----------------
	// 3D lambda!
	//-----------------

	//-----------------------------------------------------------------------------------
	auto lambda = [] __host__ __device__ (GReal_t* x)
	{

		GReal_t t=0;
		for(int i=0; i<3; i++)
			t+= x[i]*x[i];

		return exp(-t/sqrt(2.0*PI));

	};

	auto lambaW  = LambdaWrapper<GReal_t(GReal_t* x), decltype(lambda) >(lambda);

	//3D range
	std::array<GReal_t, 3>  min3  ={-5.0, -5.0, -5.0 };
	std::array<GReal_t, 3>  max3  ={ 5.0,  5.0,  5.0 };

	TH3D hist_lambda_DEVICE("lambda_CCP_OMP_DEVICE", "Lambda 3D Gaussian (DEVICE)", 20, -6, 6, 20, -6, 6, 20, -6, 6);

	{
		auto lambda_data_d = Generator.Sample<hydra::device>(lambaW,min3, max3, nentries );
		hydra::mc_host_vector<thrust::tuple<GReal_t, GReal_t, GReal_t>> lambda_data_h(lambda_data_d);
		for(auto t : lambda_data_h){

			GReal_t x = thrust::get<0>(t);
			GReal_t y = thrust::get<1>(t);
			GReal_t z = thrust::get<2>(t);

			hist_lambda_DEVICE.Fill(x,y,z);

		}
	}

	TH3D hist_lambda_HOST("lambda_CCP_OMP_HOST", "Lambda 3D Gaussian (HOST)", 20, -6, 6, 20, -6, 6, 20, -6, 6);

	{
		auto lambda_data_h = Generator.Sample<hydra::host>(lambaW,min3, max3, nentries );

		for(auto t : lambda_data_h){

			GReal_t x = thrust::get<0>(t);
			GReal_t y = thrust::get<1>(t);
			GReal_t z = thrust::get<2>(t);

			hist_lambda_HOST.Fill(x,y,z);

		}
	}

*/


#ifdef _ROOT_AVAILABLE_
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_d("canvas_d" ,"Distributions - Device", 1000, 1000);
	canvas_d.Divide(2,2);
	canvas_d.cd(1); hist_uniform_d.Draw("hist");
	canvas_d.cd(2); hist_gaussian_d.Draw("hist");
	canvas_d.cd(3); hist_exp_d.Draw("hist");
	canvas_d.cd(4); hist_bw_d.Draw("hist");

	//draw histograms
	TCanvas canvas_h("canvas_h" ,"Distributions - Host", 1000, 1000);
	canvas_h.Divide(2,2);
	canvas_h.cd(1); hist_uniform_h.Draw("hist");
	canvas_h.cd(2); hist_gaussian_h.Draw("hist");
	canvas_h.cd(3); hist_exp_h.Draw("hist");
	canvas_h.cd(4); hist_bw_h.Draw("hist");

	myapp->Run();

#endif //_ROOT_AVAILABLE_

	return 0;



}


#endif /* SAMPLE_DISTRIBUTION_INL_ */
