/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * dalitz_plot.inl
 *
 *  Created on: 29/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZ_PLOT_INL_
#define DALITZ_PLOT_INL_

#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>

//command line
#include <tclap/CmdLine.h>

//hydra
#include <hydra/host/System.h>
#include <hydra/device/System.h>
#include <hydra/Function.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/Placeholders.h>
#include <hydra/Complex.h>
#include <hydra/Tuple.h>
#include <hydra/GenericRange.h>
#include <hydra/Distance.h>

#include <hydra/LogLikelihoodFCN.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>

#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/PhaseSpaceIntegrator.h>
#include <hydra/Decays.h>

#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>

#include <hydra/functions/BreitWignerLineShape.h>
#include <hydra/functions/WignerDFunctions.h>
#include <hydra/functions/CosHelicityAngle.h>
#include <hydra/functions/ZemachFunctions.h>

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
#ifdef _ROOT_AVAILABLE_

#include <TROOT.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>

#endif //_ROOT_AVAILABLE_


using namespace ROOT::Minuit2;
using namespace hydra::placeholders;

template<unsigned int CHANNEL, hydra::Wave L>
class Resonance: public hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>
{
	using hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>::_par;

	constexpr static unsigned int _I1 = CHANNEL-1;
	constexpr static unsigned int _I2 = (CHANNEL!=3)*CHANNEL;
	constexpr static unsigned int _I3 = 3-( (CHANNEL-1) + (CHANNEL!=3)*CHANNEL );


public:

	Resonance() = delete;

	Resonance(hydra::Parameter const& c_re, hydra::Parameter const& c_im,
			hydra::Parameter const& mass, hydra::Parameter const& width,
			double mother_mass,	double daugther1_mass,
			double daugther2_mass, double daugther3_mass,
			double radi):
			hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>{c_re, c_im, mass, width},
			fLineShape(mass, width, mother_mass, daugther1_mass, daugther2_mass, daugther3_mass, radi)
	{}


	__host__ __device__
	Resonance( Resonance< CHANNEL,L> const& other):
	hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>(other),
	fLineShape(other.GetLineShape())
	{}

	__host__ __device__
	Resonance< CHANNEL ,L>&
	operator=( Resonance< CHANNEL ,L> const& other)
	{
		if(this==&other) return *this;

		hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>::operator=(other);
		fLineShape=other.GetLineShape();

		return *this;
	}

	__host__ __device__
	hydra::BreitWignerLineShape<L> const& GetLineShape() const {	return fLineShape; }

	__host__ __device__ inline
	hydra::complex<double> Evaluate(unsigned int n, hydra::Vector4R* p)  const {


		hydra::Vector4R p1 = p[_I1];
		hydra::Vector4R p2 = p[_I2];
		hydra::Vector4R p3 = p[_I3];

		hydra::CosTheta fCosDecayAngle;
		hydra::ZemachFunction<L> fAngularDist;

		fLineShape.SetParameter(0, _par[2]);
		fLineShape.SetParameter(1, _par[3]);

		double theta = fCosDecayAngle( (p1+p2+p3), (p1+p2), p1 );
		double angular = fAngularDist(theta);
		auto r = hydra::complex<double>(_par[0], _par[1])*fLineShape((p1+p2).mass())*angular;

		return r;

	}

private:

	mutable hydra::BreitWignerLineShape<L> fLineShape;


};

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

	//-----------------
    //magnitudes and phases from Cleo-c model
    //https://arxiv.org/pdf/0802.4214.pdf

	double NR_MAG         = 7.4;
	double NR_PHI         = (-18.4+180.0)*0.01745329;
	double NR_CRe		  = NR_MAG*cos(NR_PHI);
	double NR_CIm		  = NR_MAG*sin(NR_PHI);

	double K800_MASS  	  = 0.809 ;
	double K800_WIDTH     = 0.470;
	double K800_MAG       = 5.01;
	double K800_PHI       = (-163.7+180.0)*0.01745329;
	double K800_CRe		  = K800_MAG*cos(K800_PHI);
	double K800_CIm		  = K800_MAG*sin(K800_PHI);

	double KST_892_MASS   = 0.89555;
	double KST_892_WIDTH  = 0.0473;
	double KST_892_MAG    = 1.0;
	double KST_892_PHI    = 0.0;
	double KST_892_CRe	  = KST_892_MAG*cos(KST_892_PHI);
	double KST_892_CIm	  = KST_892_MAG*sin(KST_892_PHI);

	double KST0_1430_MASS  = 1.425;
	double KST0_1430_WIDTH = 0.270;
	double KST0_1430_MAG   = 3.0;
	double KST0_1430_PHI   = (49.7-180.0)*0.01745329;
	double KST0_1430_CRe   = KST0_1430_MAG*cos(KST0_1430_PHI);
	double KST0_1430_CIm   = KST0_1430_MAG*sin(KST0_1430_PHI);

	double KST2_1430_MASS  = 1.4324;
	double KST2_1430_WIDTH = 0.109;
	double KST2_1430_MAG   = 0.962;
	double KST2_1430_PHI   = (-29.9+180.0)*0.01745329;
	double KST2_1430_CRe   = KST2_1430_MAG*cos(KST2_1430_PHI);
	double KST2_1430_CIm   = KST2_1430_MAG*sin(KST2_1430_PHI);

	double KST_1680_MASS  = 1.718;
	double KST_1680_WIDTH = 0.322;
	double KST_1680_MAG   = 6.5;
	double KST_1680_PHI   = (29.0)*0.01745329;
	double KST_1680_CRe	  = KST_1680_MAG*cos(KST_1680_PHI);
	double KST_1680_CIm	  = KST_1680_MAG*sin(KST_1680_PHI);

    double D_MASS         = 1.86959;
    double K_MASS         = 0.493677;  // K+ mass
    double PI_MASS        = 0.13957061;// pi mass


    //======================================================
	//K(800)
	auto mass    = hydra::Parameter::Create().Name("MASS_K800" ).Value(K800_MASS ).Error(0.0001).Limits(K800_MASS*0.95,  K800_MASS*1.05 );
	auto width   = hydra::Parameter::Create().Name("WIDTH_K800").Value(K800_WIDTH).Error(0.0001).Limits(K800_WIDTH*0.95, K800_WIDTH*1.05);

	auto coef_re = hydra::Parameter::Create().Name("A_RE_K800" ).Value(K800_CRe).Error(0.001).Limits(K800_CRe*0.95,K800_CRe*1.05);
	auto coef_im = hydra::Parameter::Create().Name("A_IM_K800" ).Value(K800_CIm).Error(0.001).Limits(K800_CIm*0.95,K800_CIm*1.05);

	Resonance<1, hydra::SWave> K800_Resonance_12(coef_re, coef_im, mass, width,
		    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	Resonance<3, hydra::SWave> K800_Resonance_13(coef_re, coef_im, mass, width,
			    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	auto K800_Resonance = (K800_Resonance_12 + K800_Resonance_13);

	//======================================================
	//K*(892)
	mass    = hydra::Parameter::Create().Name("MASS_KST_892" ).Value(KST_892_MASS ).Error(0.0001).Limits(KST_892_MASS*0.95,  KST_892_MASS*1.05 );
	width   = hydra::Parameter::Create().Name("WIDTH_KST_892").Value(KST_892_WIDTH).Error(0.0001).Limits(KST_892_WIDTH*0.95, KST_892_WIDTH*1.05);

	coef_re = hydra::Parameter::Create().Name("A_RE_KST_892" ).Value(KST_892_CRe).Error(0.001).Limits(KST_892_CRe*0.95,KST_892_CRe*1.05).Fixed();
	coef_im = hydra::Parameter::Create().Name("A_IM_KST_892" ).Value(KST_892_CIm).Error(0.001).Limits(KST_892_CIm*0.95,KST_892_CIm*1.05).Fixed();

	Resonance<1, hydra::PWave> KST_892_Resonance_12(coef_re, coef_im, mass, width,
		    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	Resonance<3, hydra::PWave> KST_892_Resonance_13(coef_re, coef_im, mass, width,
			    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	auto KST_892_Resonance = (KST_892_Resonance_12 - KST_892_Resonance_13);

	//======================================================
	//K*0(1430)
	mass    = hydra::Parameter::Create().Name("MASS_KST0_1430" ).Value(KST0_1430_MASS ).Error(0.0001).Limits(KST0_1430_MASS*0.95,  KST0_1430_MASS*1.05 );
	width   = hydra::Parameter::Create().Name("WIDTH_KST0_1430").Value(KST0_1430_WIDTH).Error(0.0001).Limits(KST0_1430_WIDTH*0.95, KST0_1430_WIDTH*1.05);

	coef_re = hydra::Parameter::Create().Name("A_RE_KST0_1430" ).Value(KST0_1430_CRe).Error(0.001).Limits(KST0_1430_CRe*0.95,KST0_1430_CRe*1.05);
	coef_im = hydra::Parameter::Create().Name("A_IM_KST0_1430" ).Value(KST0_1430_CIm).Error(0.001).Limits(KST0_1430_CIm*0.95,KST0_1430_CIm*1.05);

	Resonance<1, hydra::SWave> KST0_1430_Resonance_12(coef_re, coef_im, mass, width,
		    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	Resonance<3, hydra::SWave> KST0_1430_Resonance_13(coef_re, coef_im, mass, width,
			    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	auto KST0_1430_Resonance = (KST0_1430_Resonance_12 + KST0_1430_Resonance_13);

	//======================================================
	//K*2(1430)
	mass    = hydra::Parameter::Create().Name("MASS_KST2_1430" ).Value(KST2_1430_MASS ).Error(0.0001).Limits(KST2_1430_MASS*0.95,  KST2_1430_MASS*1.05 );
	width   = hydra::Parameter::Create().Name("WIDTH_KST2_1430").Value(KST2_1430_WIDTH).Error(0.0001).Limits(KST2_1430_WIDTH*0.95, KST2_1430_WIDTH*1.05);

	coef_re = hydra::Parameter::Create().Name("A_RE_KST2_1430" ).Value(KST2_1430_CRe).Error(0.001).Limits(KST2_1430_CRe*0.95,KST2_1430_CRe*1.05);
	coef_im = hydra::Parameter::Create().Name("A_IM_KST2_1430" ).Value(KST2_1430_CIm).Error(0.001).Limits(KST2_1430_CIm*0.95,KST2_1430_CIm*1.05);

	Resonance<1, hydra::DWave> KST2_1430_Resonance_12(coef_re, coef_im, mass, width,
		    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	Resonance<3, hydra::DWave> KST2_1430_Resonance_13(coef_re, coef_im, mass, width,
			    	D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	auto KST2_1430_Resonance = (KST2_1430_Resonance_12 + KST2_1430_Resonance_13);

	//======================================================
	//K*(1680)
	mass    = hydra::Parameter::Create().Name("MASS_KST_1680" ).Value(KST_1680_MASS ).Error(0.0001).Limits(KST_1680_MASS*0.95,  KST_1680_MASS*1.05 );
	width   = hydra::Parameter::Create().Name("WIDTH_KST_1680").Value(KST_1680_WIDTH).Error(0.0001).Limits(KST_1680_WIDTH*0.95, KST_1680_WIDTH*1.05);

	coef_re = hydra::Parameter::Create().Name("A_RE_KST_1680" ).Value(KST_1680_CRe).Error(0.001).Limits(KST_1680_CRe*0.95,KST_1680_CRe*1.05);
	coef_im = hydra::Parameter::Create().Name("A_IM_KST_1680" ).Value(KST_1680_CIm).Error(0.001).Limits(KST_1680_CIm*0.95,KST_1680_CIm*1.05);

	Resonance<1, hydra::PWave> KST_1680_Resonance_12(coef_re, coef_im, mass, width,
			D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);

	Resonance<3, hydra::PWave> KST_1680_Resonance_13(coef_re, coef_im, mass, width,
			D_MASS,	K_MASS, PI_MASS, PI_MASS , 5.0);


	auto KST_1680_Resonance = (KST_1680_Resonance_12 - KST_1680_Resonance_13);

	//======================================================
	//NR
	coef_re = hydra::Parameter::Create().Name("A_RE_NR" ).Value(NR_CRe).Error(0.001).Limits(NR_CRe*0.95,NR_CRe*1.05);
	coef_im = hydra::Parameter::Create().Name("A_IM_NR" ).Value(NR_CIm).Error(0.001).Limits(NR_CIm*0.95,NR_CIm*1.05);


	//======================================================
	//Total: Model |N.R + \sum{ Resonaces }|^2

	//parametric lambda
	auto Norm = hydra::wrap_lambda(
					[]__host__  __device__ (unsigned int np, const hydra::Parameter* p, unsigned int n, hydra::complex<double>* x){

				hydra::complex<double> r(p[0],p[1]);

				for(unsigned int i=0; i< n;i++)	r += x[i];

				return hydra::norm(r);
	}, coef_re, coef_im);

	//model-functor
	auto Model = hydra::compose(Norm,
			K800_Resonance,
			KST_892_Resonance,
			KST0_1430_Resonance,
			KST2_1430_Resonance,
			KST_1680_Resonance
			);

	//--------------------
	//generator
	hydra::Vector4R B0(D_MASS, 0.0, 0.0, 0.0);
	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{K_MASS, PI_MASS, PI_MASS};

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[]__host__ __device__(unsigned int n, hydra::Vector4R* p ){

		double   M2_12 = (p[0]+p[1]).mass2();
		double   M2_13 = (p[0]+p[2]).mass2();
		double   M2_23 = (p[1]+p[2]).mass2();

		return hydra::make_tuple(M2_12, M2_13, M2_23);
	});


#ifdef 	_ROOT_AVAILABLE_
	//
	TH3D Dalitz_Flat("Dalitz_Flat",
			"Flat Dalitz;"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi^{+} #pi^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));

	TH3D Dalitz_Resonances("Dalitz_Resonances",
			"Dalitz - Toy Data -;"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi^{+} #pi^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));


	TH3D Dalitz_Fit("Dalitz_Fit",
			"Dalitz - Fit -;"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi^{+} #pi^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));


	//control plots
	TH2D Normalization("normalization" ,
			"Model PDF Normalization;Norm;Error",
			100, 302.0, 304.0,
			100, 0.4, 1.2);

#endif

	hydra::Decays<3, hydra::host::sys_t > toy_data;

	//toy data production on device
	{
		std::cout << std::endl;
        std::cout << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout << "======= 1 - GENERATE TOY-DATA ========" << std::endl;
		std::cout << "======================================" << std::endl;

		//allocate memory to hold the final states particles
		hydra::Decays<3, hydra::device::sys_t > Events_d(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, Events_d.begin(), Events_d.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| D+ -> K- pi+ pi+"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

		auto particles        = Events_d.GetUnweightedDecays();
		auto dalitz_variables = hydra::make_range( particles.begin(), particles.end(), dalitz_calculator);
		auto dalitz_weights   = Events_d.GetWeights();

		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "<======= [Daliz variables]  weight : ( MSq_12, MSq_13, MSq_23)  =======>"<< std::endl;

		for( size_t i=0; i<10; i++ )
			std::cout << dalitz_weights[i] << " : "<< dalitz_variables[i] << std::endl;

		//flat dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Dalitz{
				{100,100,100},
				{pow(K_MASS + PI_MASS,2), pow(K_MASS + PI_MASS,2),  pow(PI_MASS + PI_MASS,2)},
				{pow(D_MASS - PI_MASS,2), pow(D_MASS - PI_MASS ,2), pow(D_MASS - K_MASS,2)}
		};

		start = std::chrono::high_resolution_clock::now();

		Hist_Dalitz.Fill( dalitz_variables.begin(),
				dalitz_variables.end(), dalitz_weights.begin()  );

		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Sparse histogram fill"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

		#ifdef 	_ROOT_AVAILABLE_

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

		//if device is cuda, bring the histogram data to the host
		//to fill the ROOT histogram faster
		{
			hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Dalitz);
			std::cout << "Filling a ROOT Histogram... " << std::endl;

			for(auto entry : Hist_Temp)
			{
				size_t bin     = hydra::get<0>(entry);
				double content = hydra::get<1>(entry);
				unsigned int bins[3];
				Hist_Temp.GetIndexes(bin, bins);
				Dalitz_Flat.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

			}
		}
#else
		std::cout << "Filling a ROOT Histogram... " << std::endl;

		for(auto entry : Hist_Dalitz)
		{
			size_t bin     = hydra::get<0>(entry);
			double content = hydra::get<1>(entry);
			unsigned int bins[3];
			Hist_Dalitz.GetIndexes(bin, bins);
			Dalitz_Flat.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

		}
#endif

#endif

		auto last = Events_d.Unweight(Model, 1.0);

		std::cout <<std::endl;
		std::cout << "<======= Toy data =======>"<< std::endl;
		for( size_t i=0; i<10; i++ )
			std::cout << Events_d[i] << std::endl;


		std::cout << std::endl <<"Toy Dataset size: "<< last << std::endl;

		toy_data.resize(last);
		hydra::copy(Events_d.begin(), Events_d.begin()+last, toy_data.begin());

	}//toy data production on device


	//plot toy-data
	{
		std::cout << std::endl;
	    std::cout << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout << "========= 2 - PLOT TOY-DATA ==========" << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout <<  std::endl << std::endl;

		hydra::Decays<3, hydra::device::sys_t > toy_data_temp(toy_data);

		auto particles        = toy_data_temp.GetUnweightedDecays();
		auto dalitz_variables = hydra::make_range( particles.begin(), particles.end(), dalitz_calculator);

		std::cout << "<======= [Daliz variables] { ( MSq_12, MSq_13, MSq_23) } =======>"<< std::endl;

		for( size_t i=0; i<10; i++ )
			std::cout << dalitz_variables[i] << std::endl;

		//flat dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Dalitz{
			{100,100,100},
			{pow(K_MASS + PI_MASS,2), pow(K_MASS + PI_MASS,2),  pow(PI_MASS + PI_MASS,2)},
			{pow(D_MASS - PI_MASS,2), pow(D_MASS - PI_MASS ,2), pow(D_MASS - K_MASS,2)}
		};

		auto start = std::chrono::high_resolution_clock::now();

		Hist_Dalitz.Fill( dalitz_variables.begin(), dalitz_variables.end() );

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Sparse histogram fill"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

#ifdef 	_ROOT_AVAILABLE_

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

		//if device is cuda, bring the histogram data to the host
		//to fill the ROOT histogram faster
		{
			hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Dalitz);
			std::cout << "Filling a ROOT Histogram... " << std::endl;

			for(auto entry : Hist_Temp)
			{
				size_t bin     = hydra::get<0>(entry);
				double content = hydra::get<1>(entry);
				unsigned int bins[3];
				Hist_Temp.GetIndexes(bin, bins);
				Dalitz_Resonances.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

			}
		}
#else
		std::cout << "Filling a ROOT Histogram... " << std::endl;

		for(auto entry : Hist_Dalitz)
		{
			size_t bin     = hydra::get<0>(entry);
			double content = hydra::get<1>(entry);
			unsigned int bins[3];
			Hist_Dalitz.GetIndexes(bin, bins);
			Dalitz_Resonances.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

		}
#endif

#endif

	}


	// fit
	{
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout << "=============== 3 - FIT ==============" << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout <<  std::endl << std::endl;
		//pdf
		auto Model_PDF = hydra::make_pdf( Model,
				hydra::PhaseSpaceIntegrator<3, hydra::device::sys_t>(D_MASS, {K_MASS, PI_MASS, PI_MASS}, 500000));

		std::cout << "-----------------------------------------"<<std::endl;
		std::cout <<"| Initial PDF Norm: "<< Model_PDF.GetNorm() << "̣ +/- " <<   Model_PDF.GetNormError() << std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		hydra::Decays<3, hydra::device::sys_t > toy_data_temp(toy_data);
		auto particles        = toy_data_temp.GetUnweightedDecays();

		auto fcn = hydra::make_loglikehood_fcn(Model_PDF, particles.begin(),
				particles.end());

		//print level
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

		FunctionMinimum minimum_d =  FunctionMinimum( migrad_d(5000, 5) );

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Migrad] Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		//print parameters after fitting
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		//----------
		//plot fit
		//allocate memory to hold the final states particles
		hydra::Decays<3, hydra::device::sys_t > fit_data(nentries);

		auto start = std::chrono::high_resolution_clock::now();

		//generate the final state particles
		phsp.Generate(B0, fit_data.begin(), fit_data.end());

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------- Device (fit data) -----------"<< std::endl;
		std::cout << "| D+ -> K- pi+ pi+"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;


		fit_data.Reweight(fcn.GetPDF().GetFunctor());

		auto particles_fit        = fit_data.GetUnweightedDecays();
		auto dalitz_variables_fit = hydra::make_range( particles_fit.begin(), particles_fit.end(), dalitz_calculator);
		auto dalitz_weights_fit   = fit_data.GetWeights();

		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "<======= [Daliz variables - fit] { weight : ( MSq_12, MSq_13, MSq_23) } =======>"<< std::endl;

		for( size_t i=0; i<10; i++ )
			std::cout << dalitz_weights_fit[i] << " : "<< dalitz_variables_fit[i] << std::endl;

		//flat dalitz histogram
		hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Dalitz{
			{100,100,100},
			{pow(K_MASS + PI_MASS,2), pow(K_MASS + PI_MASS,2),  pow(PI_MASS + PI_MASS,2)},
			{pow(D_MASS - PI_MASS,2), pow(D_MASS - PI_MASS ,2), pow(D_MASS - K_MASS,2)}
		};

		start = std::chrono::high_resolution_clock::now();

		Hist_Dalitz.Fill( dalitz_variables_fit.begin(),
				dalitz_variables_fit.end(), dalitz_weights_fit.begin()  );

		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| Sparse histogram fill"                       << std::endl;
		std::cout << "| Number of events :"<< nentries          << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;

#ifdef 	_ROOT_AVAILABLE_

		for(auto x: fcn.GetPDF().GetNormCache() ){

			std::cout << "Key : "    << x.first
					  << " Value : " << x.second.first
					  << " Error : " << x.second.second
					  << std::endl;
			Normalization.Fill(x.second.first, x.second.second );
		}



#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

		//if device is cuda, bring the histogram data to the host
		//to fill the ROOT histogram faster
		{
			hydra::SparseHistogram<double, 3,  hydra::host::sys_t> Hist_Temp(Hist_Dalitz);
			std::cout << "Filling a ROOT Histogram... " << std::endl;

			for(auto entry : Hist_Temp)
			{
				size_t bin     = hydra::get<0>(entry);
				double content = hydra::get<1>(entry);
				unsigned int bins[3];
				Hist_Temp.GetIndexes(bin, bins);
				Dalitz_Fit.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

			}

		}
#else
		std::cout << "Filling a ROOT Histogram... " << std::endl;

		for(auto entry : Hist_Dalitz)
		{
			size_t bin     = hydra::get<0>(entry);
			double content = hydra::get<1>(entry);
			unsigned int bins[3];
			Hist_Dalitz.GetIndexes(bin, bins);
			Dalitz_Fit.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

		}
#endif

#endif

	}



#ifdef 	_ROOT_AVAILABLE_






	TApplication *m_app=new TApplication("myapp",0,0);

	Dalitz_Fit.Scale(Dalitz_Resonances.Integral()/Dalitz_Fit.Integral() );

	TCanvas canvas_1("canvas_1", "Phase-space FLAT", 500, 500);
	Dalitz_Flat.Project3D("yz")->Draw("colz");

	TCanvas canvas_2("canvas_2", "Phase-space FLAT", 500, 500);
	Dalitz_Flat.Project3D("xy")->Draw("colz");

	TCanvas canvas_3("canvas_3", "Phase-space FLAT", 500, 500);
	Dalitz_Resonances.Project3D("yz")->Draw("colz");

	TCanvas canvas_4("canvas_4", "Phase-space FLAT", 500, 500);
	Dalitz_Resonances.Project3D("xy")->Draw("colz");

	TCanvas canvas_5("canvas_3", "Phase-space FLAT", 500, 500);
	Dalitz_Fit.Project3D("yz")->Draw("colz");

	TCanvas canvas_6("canvas_4", "Phase-space FLAT", 500, 500);
	Dalitz_Fit.Project3D("xy")->Draw("colz");

	//projections
	TCanvas canvas_x("canvas_x", "", 500, 500);
	Dalitz_Fit.Project3D("x")->Draw("hist");
	Dalitz_Fit.Project3D("x")->SetLineColor(2);
	Dalitz_Resonances.Project3D("x")->Draw("e0same");

	TCanvas canvas_y("canvas_y", "", 500, 500);
	Dalitz_Fit.Project3D("y")->Draw("hist");
	Dalitz_Fit.Project3D("y")->SetLineColor(2);
	Dalitz_Resonances.Project3D("y")->Draw("e0same");

	TCanvas canvas_z("canvas_z", "", 500, 500);
	Dalitz_Fit.Project3D("z")->Draw("hist");
	Dalitz_Fit.Project3D("z")->SetLineColor(2);
	Dalitz_Resonances.Project3D("z")->Draw("e0same");

	TCanvas canvas_7("canvas_7", "Normalization", 500, 500);
	Normalization.Draw("colz");


	m_app->Run();

#endif

	return 0;
}




#endif /* DALITZ_PLOT_INL_ */
