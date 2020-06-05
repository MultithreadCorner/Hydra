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
 * fill_basic_distributions.inl
 *
 *  Created on: Jun 05, 2020
 *      Author: Davide Brundu, 
 *              Antonio Augusto Alves Junior
 */

#ifndef FILL_BASIC_DISTRIBUTIONS_INL_
#define FILL_BASIC_DISTRIBUTIONS_INL_

/**
 * \example fill_basic_distributions.inl
 *
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
#include <hydra/Parameter.h>
#include <hydra/RandomFill.h>

//hydra functions
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




#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


declarg(xvar, double)

using namespace hydra::arguments;

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


    auto data = hydra::device::vector<double>(nentries);



    //Gaussian distribution
    //Parameters
    auto mean  = hydra::Parameter::Create("mean" ).Value(0.0);
    auto sigma = hydra::Parameter::Create("sigma").Value(0.25);

    auto gauss     = hydra::Gaussian<xvar>(mean, sigma);


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
    
    
    
    hydra::fill_random(data , gauss);
    
    for(size_t i=0; i<10; ++i) std::cout << data[i] << std::endl;
    


#ifdef _ROOT_AVAILABLE_

    TH1D       hist_gauss("hist_gauss",     "hydra::Gaussian<xvar>"   ,        100,-8.0, 8.0);
    TH1D   hist_lognormal("hist_lognormal", "hydra::LogNormal<xvar>"   ,       100,0.0, 2.5);
    TH1D     hist_bigauss("hist_bigauss",   "hydra::BifurcatedGaussian<xvar>", 100,-8.0, 8.0);
    TH1D         hist_exp("hist_exp" ,      "hydra::Exponential<xvar>",        100, 0.0, 10.0);
    TH1D          hist_bw("hist_bw"  ,      "hydra::BreitWignerNR<xvar>",      100, 0.0, 10.0);
    TH1D         hist_chi("hist_chi" ,      "hydra::ChiSquare<xvar>",          100, 0.0, 10.0);
    TH1D  hist_johnson_su("hist_su"  ,      "hydra::JohnsonSU<xvar>",          100, -5.0, 1.0);
    TH1D     hist_uniform("hist_uniform",   "hydra::UniformShape<xvar>",       100, -6.0, 6.0);
    TH1D    hist_triangle("hist_triangle",  "hydra::TriangularShape<xvar>",    100, -6.0, 6.0);
    TH1D   hist_trapezoid("hist_trapezoid", "hydra::TrapezoidalShape<xvar>",   100, -6.0, 6.0);
    hist_uniform.SetMinimum(0.0);


    for(auto x : data) hist_gauss.Fill( x );
    
    /*
    hydra::fill_random(data , lognormal);
    for(auto x : data) hist_lognormal.Fill( x );
    
    hydra::fill_random(data , bigauss);
    for(auto x : data) hist_bigauss.Fill( x );
    
    hydra::fill_random(data , exp);
    for(auto x : data) hist_exp.Fill( x );
    
    hydra::fill_random(data , bw);
    for(auto x : data) hist_bw.Fill( x );
    
    hydra::fill_random(data , chi2);
    for(auto x : data) hist_chi.Fill( x );
    
    hydra::fill_random(data , johnson_su);
    for(auto x : data) hist_johnson_su.Fill( x );
    
    hydra::fill_random(data , uniform);
    for(auto x : data) hist_uniform.Fill( x );
    
    hydra::fill_random(data , triangle);
    for(auto x : data) hist_triangle.Fill( x );
    
    hydra::fill_random(data , trapezoid);
    for(auto x : data) hist_trapezoid.Fill( x );
    */
    

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



#endif /* FILL_BASIC_DISTRIBUTIONS_INL_ */
