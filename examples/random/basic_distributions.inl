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
 * basic_distributions.inl
 *
 *  Created on: Jul 18, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASIC_DISTRIBUTIONS_INL_
#define BASIC_DISTRIBUTIONS_INL_

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

//root
#include <TROOT.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>


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
	hydra::Random<thrust::random::default_random_engine>
	Generator( std::chrono::system_clock::now().time_since_epoch().count() );

	//------------------------
	TH1D hist_uniform_d("uniform_d",   "Uniform distribution - device",     100, -6.0, 6.0);
	TH1D hist_gaussian_d("gaussian_d", "Gaussian distribution - device",    100, -6, 6);
    TH1D hist_exp_d("exponential_d",   "Exponential distribution - device", 100, 0, 5);
    TH1D hist_bw_d("breit_wigner_d",   "Breit-Wigner distribution - device",100, 0, 5);

    //device
    {
    	//1D device buffer
    	hydra::device::vector<double>  data_d(nentries);
    	hydra::host::vector<double>    data_h(nentries);

    	//uniform
    	Generator.Uniform(-5.0, 5.0, data_d.begin(), data_d.end());
    	hydra::copy(gen_data_d.begin(), gen_data_d.end(), gen_data_h.begin());
    	for(auto value: gen_data_h) hist_uniform_DEVICE.Fill( value);

    	//gaussian
    	Generator.Gauss(0.0, 1.0, gen_data_d.begin(), gen_data_d.end());
    	thrust::copy(gen_data_d.begin(), gen_data_d.end(), gen_data_h.begin());
    	for(auto value: hydra::mc_host_vector<GReal_t>(gen_data_d)) hist_gaussian_DEVICE.Fill( value);

    	//exponential
    	Generator.Exp(1.0, gen_data_d.begin(), gen_data_d.end());
    	thrust::copy(gen_data_d.begin(), gen_data_d.end(), gen_data_h.begin());
    	for(auto value: gen_data_h) hist_exp_DEVICE.Fill( value);

    	//breit-wigner
    	Generator.BreitWigner(2.0, 0.2, gen_data_d.begin(), gen_data_d.end());
    	thrust::copy(gen_data_d.begin(), gen_data_d.end(), gen_data_h.begin());
    	for(auto value: gen_data_h) hist_bw_DEVICE.Fill( value);

      }

    TList List_1D_Histo;
    List_1D_Histo.Add(&hist_uniform_DEVICE );
    List_1D_Histo.Add(&hist_gaussian_DEVICE );
    List_1D_Histo.Add(&hist_bw_DEVICE );
    List_1D_Histo.Add(&hist_exp_DEVICE );
    List_1D_Histo.Add(&hist_ucdf_DEVICE );


    TH1D hist_uniform_HOST("uniform_CCP_OMP_HOST", "Uniform distribution (HOST)", 100, -5.5, 5.5);
    TH1D hist_gaussian_HOST("gaussian_CCP_OMP_HOST", "Gaussian distribution (HOST)", 100, -5, 5);
    TH1D hist_exp_HOST("exponential_CCP_OMP_HOST", "Exponential distribution (HOST)", 100, 0, 5);
    TH1D hist_bw_HOST("breit_wigner_CCP_OMP_HOST", "Breit-Wigner distribution (HOST)", 100, 0, 5);
    TH1D hist_ucdf_HOST("ucdf", "Custom inverse CDF (HOST)", 500, 0, 1);


    {
    	//1D host buffer
    	hydra::mc_host_vector<GReal_t>   gen_data_h(nentries);

    	//uniform
    	Generator.Uniform(-5.0, 5.0, gen_data_h.begin(), gen_data_h.end());
    	for(auto value: gen_data_h) hist_uniform_HOST.Fill( value);

    	//gaussian
    	Generator.Gauss(0.0, 1.0, gen_data_h.begin(), gen_data_h.end());
    	for(auto value: gen_data_h) hist_gaussian_HOST.Fill( value);

    	//exponential
    	Generator.Exp(1.0, gen_data_h.begin(), gen_data_h.end());
    	for(auto value: gen_data_h) hist_exp_HOST.Fill( value);

    	//breit-wigner
    	Generator.BreitWigner(2.0, 0.2, gen_data_h.begin(), gen_data_h.end());
    	for(auto value: gen_data_h) hist_bw_HOST.Fill( value);

    }

    List_1D_Histo.Add(&hist_uniform_HOST );
    List_1D_Histo.Add(&hist_gaussian_HOST );
    List_1D_Histo.Add(&hist_bw_HOST );
    List_1D_Histo.Add(&hist_exp_HOST );
    List_1D_Histo.Add(&hist_ucdf_HOST );

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


	/***********************************
	 * RootApp, Drawing...
	 ***********************************/
	TApplication *myapp=new TApplication("myapp",0,0);

	//draw histograms
	TCanvas canvas_gaussians_DEVICE("canvas_gaussians_DEVICE" ,"", 500, 500);
	hist_gaussians_DEVICE.Draw("colz");
	canvas_gaussians_DEVICE.Print(TString::Format("plots/HIST_2D_%s.png", hist_gaussians_DEVICE.GetName()  ));

	//draw histograms
	TCanvas canvas_lambda_DEVICE("canvas_lambda_DEVICE" ,"", 500, 500);
	hist_lambda_DEVICE.Draw("box");
	canvas_lambda_DEVICE.Print(TString::Format("plots/HIST_2D_%s.png", hist_lambda_DEVICE.GetName()  ));


	TCanvas canvas_gaussians_HOST("canvas_gaussians_HOST" ,"", 500, 500);
	hist_gaussians_HOST.Draw("colz");
	canvas_gaussians_HOST.Print(TString::Format("plots/HIST_2D_%s.png", hist_gaussians_HOST.GetName()  ));


	//draw histograms
	TCanvas canvas_lambda_HOST("canvas_lambda_HOST" ,"", 500, 500);
	hist_lambda_HOST.Draw("box");
	canvas_lambda_HOST.Print(TString::Format("plots/HIST_2D_%s.png", hist_lambda_HOST.GetName()  ));



	TCanvas* Canvas;
	TIter next( &List_1D_Histo );
	TObject *obj;
	while((obj = next()))
	{
		Canvas = new TCanvas(
				TString::Format("hist_1D_%s"   , obj->GetName() )
		, TString::Format("hist_1D_%s"   , obj->GetName() )
		, 550, 500 );

		obj->Draw("HIST");
		((TH1* ) obj)->SetMinimum(0);

			Canvas->Print(TString::Format("plots/HIST_1D_%s.png", obj->GetName()  ));

	}
	myapp->Run();

	return 0;



}



#endif /* BASIC_DISTRIBUTIONS_INL_ */
