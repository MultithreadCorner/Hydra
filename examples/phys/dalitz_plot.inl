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
 * dalitz_plot.inl
 *
 *  Created on: 29/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZ_PLOT_INL_
#define DALITZ_PLOT_INL_


/**
 * \example dalitz_plot.inl
 *
 */



#include <iostream>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ctime>

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
#include <hydra/Range.h>
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
#include <TLegend.h>
#include <TLegendEntry.h>

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


    __hydra_dual__
	Resonance( Resonance< CHANNEL,L> const& other):
	hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>(other),
	fLineShape(other.GetLineShape())
	{}

    __hydra_dual__  inline
	Resonance< CHANNEL ,L>&
	operator=( Resonance< CHANNEL ,L> const& other)
	{
		if(this==&other) return *this;

		hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>::operator=(other);
		fLineShape=other.GetLineShape();

		return *this;
	}

    __hydra_dual__  inline
	hydra::BreitWignerLineShape<L> const& GetLineShape() const {	return fLineShape; }

    __hydra_dual__  inline
	hydra::complex<double> Evaluate(unsigned int n, hydra::Vector4R* p)  const {


		hydra::Vector4R p1 = p[_I1];
		hydra::Vector4R p2 = p[_I2];
		hydra::Vector4R p3 = p[_I3];


		fLineShape.SetParameter(0, _par[2]);
		fLineShape.SetParameter(1, _par[3]);

		double theta = fCosDecayAngle( (p1+p2+p3), (p1+p2), p1 );
		double angular = fAngularDist(theta);
		auto r = hydra::complex<double>(_par[0], _par[1])*fLineShape((p1+p2).mass())*angular;

		return r;

	}

private:

	mutable hydra::BreitWignerLineShape<L> fLineShape;
	hydra::CosHelicityAngle fCosDecayAngle;
	hydra::ZemachFunction<L> fAngularDist;


};


class NonResonant: public hydra::BaseFunctor<NonResonant, hydra::complex<double>, 2>
{

	using hydra::BaseFunctor<NonResonant, hydra::complex<double>, 2>::_par;

public:

	NonResonant() = delete;

	NonResonant(hydra::Parameter const& c_re, hydra::Parameter const& c_im):
			hydra::BaseFunctor<NonResonant, hydra::complex<double>, 2>{c_re, c_im}
	{}


	 __hydra_dual__
	NonResonant( NonResonant const& other):
	hydra::BaseFunctor<NonResonant, hydra::complex<double>, 2>(other)
	{}

	 __hydra_dual__
	NonResonant& operator=( NonResonant const& other)
	{
		if(this==&other) return *this;

		hydra::BaseFunctor<NonResonant, hydra::complex<double>, 2>::operator=(other);

		return *this;
	}

	 __hydra_dual__  inline
	hydra::complex<double> Evaluate(unsigned int n, hydra::Vector4R* p)  const {

		return hydra::complex<double>(_par[0], _par[1]);
	}

};

template<typename Amplitude>
TH3D histogram_component( Amplitude const& amp, std::array<double, 3> const& masses, const char* name, size_t nentries);

template<typename Amplitude, typename Model>
double fit_fraction( Amplitude const& amp, Model const& model, std::array<double, 3> const& masses, size_t nentries);

template<typename Backend, typename Model, typename Container >
size_t generate_dataset(Backend const& system, Model const& model, std::array<double, 3> const& masses, Container& decays, size_t nevents, size_t bunch_size);

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

	double NR_MAG         = 4.8;
	double NR_PHI         = (-18.4+180.0)*0.01745329;
	double NR_CRe		  = NR_MAG*cos(NR_PHI);
	double NR_CIm		  = NR_MAG*sin(NR_PHI);

	double K800_MASS  	  = 0.809 ;
	double K800_WIDTH     = 0.470;
	double K800_MAG       = 2.25;
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
	double KST0_1430_MAG   = 1.50;
	double KST0_1430_PHI   = (45.7-180.0)*0.01745329;
	double KST0_1430_CRe   = KST0_1430_MAG*cos(KST0_1430_PHI);
	double KST0_1430_CIm   = KST0_1430_MAG*sin(KST0_1430_PHI);

	double KST2_1430_MASS  = 1.4324;
	double KST2_1430_WIDTH = 0.109;
	double KST2_1430_MAG   = 0.962;
	double KST2_1430_PHI   = (-33.9+180.0)*0.01745329;
	double KST2_1430_CRe   = KST2_1430_MAG*cos(KST2_1430_PHI);
	double KST2_1430_CIm   = KST2_1430_MAG*sin(KST2_1430_PHI);

	double KST_1680_MASS  = 1.718;
	double KST_1680_WIDTH = 0.322;
	double KST_1680_MAG   = 2.5;
	double KST_1680_PHI   = (26.0)*0.01745329;
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

	auto NR = NonResonant(coef_re, coef_im);
	//======================================================
	//Total: Model |N.R + \sum{ Resonaces }|^2

	//parametric lambda
	auto Norm = hydra::wrap_lambda( [] __hydra_dual__ ( unsigned int n, hydra::complex<double>* x){

				hydra::complex<double> r(0,0);

				for(unsigned int i=0; i< n;i++)	r += x[i];

				return hydra::norm(r);
	});

	//model-functor
	auto Model = hydra::compose(Norm,
		    K800_Resonance,
			KST_892_Resonance,
			KST0_1430_Resonance,
			KST2_1430_Resonance,
			KST_1680_Resonance,
			NR );


	//--------------------
	//generator
	hydra::Vector4R B0(D_MASS, 0.0, 0.0, 0.0);
	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{K_MASS, PI_MASS, PI_MASS};

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[] __hydra_dual__ (unsigned int n, hydra::Vector4R* p ){

		double   M2_12 = (p[0]+p[1]).mass2();
		double   M2_13 = (p[0]+p[2]).mass2();
		double   M2_23 = (p[1]+p[2]).mass2();

		return hydra::make_tuple(M2_12, M2_13, M2_23);
	});


#ifdef 	_ROOT_AVAILABLE_
	//
	/*
	TH3D Dalitz_Flat("Dalitz_Flat",
			"Flat Dalitz;"
			"M^{2}(K^{-} #pi_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi_{1}^{+} #pi_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));
*/
	TH3D Dalitz_Resonances("Dalitz_Resonances",
			"Dalitz - Toy Data -;"
			"M^{2}(K^{-} #pi_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi_{1}^{+} #pi_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));


	TH3D Dalitz_Fit("Dalitz_Fit",
			"Dalitz - Fit -;"
			"M^{2}(K^{-} #pi_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi_{1}^{+} #pi_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));


	//control plots
	TH2D Normalization("normalization",
			"Model PDF Normalization;Norm;Error",
			200, 275.0, 305.0,
			200, 0.58, 0.64);


	TH3D  KST800_12_HIST , KST800_13_HIST,  KST892_12_HIST,  KST892_13_HIST,
	      KST1425_12_HIST, KST1425_13_HIST, KST1430_12_HIST, KST1430_13_HIST,
	      KST1680_12_HIST, KST1680_13_HIST, NR_HIST ;

	double  KST800_12_FF,  KST800_13_FF,  KST892_12_FF,  KST892_13_FF,
		    KST1425_12_FF, KST1425_13_FF, KST1430_12_FF, KST1430_13_FF,
		    KST1680_12_FF, KST1680_13_FF, NR_FF;

#endif

	hydra::Decays<3, hydra::host::sys_t > toy_data;

	//toy data production on device
	{
		std::cout << std::endl;
        std::cout << std::endl;
		std::cout << "======================================" << std::endl;
		std::cout << "======= 1 - GENERATE TOY-DATA ========" << std::endl;
		std::cout << "======================================" << std::endl;


		auto start = std::chrono::high_resolution_clock::now();

		generate_dataset(hydra::device::sys, Model,  {D_MASS, K_MASS, PI_MASS}, toy_data, nentries, 2*nentries);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| D+ -> K- pi+ pi+"                       << std::endl;
		std::cout << "| Number of events :"<< toy_data.size()         << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;


		std::cout << std::endl <<"Toy Dataset size: "<< toy_data.size() << std::endl;

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

		FunctionMinimum minimum_d =  FunctionMinimum( migrad_d(5000,250) );

		auto end_d = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed_d = end_d - start_d;

		//time
		std::cout << "-----------------------------------------"<<std::endl;
		std::cout << "| [Migrad] Time (ms) ="<< elapsed_d.count() <<std::endl;
		std::cout << "-----------------------------------------"<<std::endl;

		//print parameters after fitting
		std::cout<<"minimum: "<<minimum_d<<std::endl;

		nentries = 2000000;
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

		//model dalitz histogram
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


		//==================================
		// Optimized components
		//==================================
		auto Opt_Model = fcn.GetPDF().GetFunctor();

		auto KST800_12 = fcn.GetPDF().GetFunctor().GetFunctor(_1).GetFunctor(_0);
		auto KST800_13 = fcn.GetPDF().GetFunctor().GetFunctor(_1).GetFunctor(_1);

		auto KST892_12 = fcn.GetPDF().GetFunctor().GetFunctor(_2).GetFunctor(_0);
		auto KST892_13 = fcn.GetPDF().GetFunctor().GetFunctor(_2).GetFunctor(_1);

		auto KST1425_12 = fcn.GetPDF().GetFunctor().GetFunctor(_3).GetFunctor(_0);
		auto KST1425_13 = fcn.GetPDF().GetFunctor().GetFunctor(_3).GetFunctor(_1);

		auto KST1430_12 = fcn.GetPDF().GetFunctor().GetFunctor(_4).GetFunctor(_0);
		auto KST1430_13 = fcn.GetPDF().GetFunctor().GetFunctor(_4).GetFunctor(_1);

		auto KST1680_12 = fcn.GetPDF().GetFunctor().GetFunctor(_5).GetFunctor(_0);
		auto KST1680_13 = fcn.GetPDF().GetFunctor().GetFunctor(_5).GetFunctor(_1);

		auto NR         = fcn.GetPDF().GetFunctor().GetFunctor(_6);

		//==================================
		// Draw components
		//==================================
		KST800_12_HIST  =	histogram_component(KST800_12 , {D_MASS, K_MASS, PI_MASS}, "KST800_12_HIST", nentries);
		KST800_13_HIST  =	histogram_component(KST800_13 , {D_MASS, K_MASS, PI_MASS}, "KST800_13_HIST", nentries);
		KST892_12_HIST  =	histogram_component(KST892_12 , {D_MASS, K_MASS, PI_MASS}, "KST892_12_HIST", nentries);
		KST892_13_HIST  =	histogram_component(KST892_13 , {D_MASS, K_MASS, PI_MASS}, "KST892_13_HIST", nentries);
		KST1425_12_HIST =	histogram_component(KST1425_12, {D_MASS, K_MASS, PI_MASS}, "KST1425_12_HIST", nentries);
		KST1425_13_HIST =	histogram_component(KST1425_13, {D_MASS, K_MASS, PI_MASS}, "KST1425_13_HIST", nentries);
		KST1430_12_HIST =	histogram_component(KST1430_12, {D_MASS, K_MASS, PI_MASS}, "KST1430_12_HIST", nentries);
		KST1430_13_HIST =	histogram_component(KST1430_13, {D_MASS, K_MASS, PI_MASS}, "KST1430_13_HIST", nentries);
		KST1680_12_HIST =	histogram_component(KST1680_12, {D_MASS, K_MASS, PI_MASS}, "KST1680_12_HIST", nentries);
		KST1680_13_HIST =	histogram_component(KST1680_13, {D_MASS, K_MASS, PI_MASS}, "KST1680_13_HIST", nentries);
		NR_HIST         =	histogram_component(NR, {D_MASS, K_MASS, PI_MASS}, "NR_HIST", nentries);

		//==================================
		// Fit fractions
		//==================================
		KST800_12_FF  =	fit_fraction(KST800_12 , Opt_Model, {D_MASS, K_MASS, PI_MASS},  nentries);
		KST800_13_FF  =	fit_fraction(KST800_13 , Opt_Model, {D_MASS, K_MASS, PI_MASS},  nentries);
		KST892_12_FF  =	fit_fraction(KST892_12 , Opt_Model, {D_MASS, K_MASS, PI_MASS},  nentries);
		KST892_13_FF  =	fit_fraction(KST892_13 , Opt_Model, {D_MASS, K_MASS, PI_MASS},  nentries);
		KST1425_12_FF =	fit_fraction(KST1425_12 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		KST1425_13_FF =	fit_fraction(KST1425_13 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		KST1430_12_FF =	fit_fraction(KST1430_12 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		KST1430_13_FF =	fit_fraction(KST1430_13 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		KST1680_12_FF =	fit_fraction(KST1680_12 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		KST1680_13_FF =	fit_fraction(KST1680_13 , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);
		NR_FF         =	fit_fraction(NR , Opt_Model, {D_MASS, K_MASS, PI_MASS}, nentries);


		std::cout << "KST800_12_FF :" << KST800_12_FF << std::endl;
		std::cout << "KST800_13_FF :" << KST800_13_FF << std::endl;
		std::cout << "KST892_12_FF :" << KST892_12_FF << std::endl;
		std::cout << "KST892_13_FF :" << KST892_13_FF << std::endl;
		std::cout << "KST1425_12_FF :" << KST1425_12_FF << std::endl;
		std::cout << "KST1425_13_FF :" << KST1425_13_FF << std::endl;
		std::cout << "KST1430_12_FF :" << KST1430_12_FF << std::endl;
		std::cout << "KST1430_13_FF :" << KST1430_13_FF << std::endl;
		std::cout << "KST1680_12_FF :" << KST1680_12_FF << std::endl;
		std::cout << "KST1680_13_FF :" << KST1680_13_FF << std::endl;
		std::cout << "NR_FF :" << NR_FF << std::endl;
		std::cout << "Sum :"
				  << KST800_12_FF  + KST800_13_FF  +
				     KST892_12_FF  + KST892_13_FF  + KST1425_12_FF +
				     KST1425_13_FF + KST1430_12_FF + KST1430_13_FF +
				     KST1680_12_FF + KST1680_13_FF + NR_FF << std::endl;

#ifdef 	_ROOT_AVAILABLE_

		{
			std::vector<double> integrals;
			std::vector<double> integrals_error;

			for(auto x: fcn.GetPDF().GetNormCache() ){
				integrals.push_back(x.second.first);
				integrals_error.push_back(x.second.second);

			}

			auto integral_bounds = std::minmax_element(integrals.begin(),
					integrals.end());

			auto  integral_error_bounds = std::minmax_element(integrals_error.begin(),
					integrals_error.end());

			Normalization.GetXaxis()->SetLimits(*integral_bounds.first, *integral_bounds.second);
			Normalization.GetYaxis()->SetLimits(*integral_error_bounds.first, *integral_error_bounds.second);

			for(auto x: fcn.GetPDF().GetNormCache() ){

				Normalization.Fill(x.second.first, x.second.second );
			}
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

	KST800_12_HIST.Scale( KST800_12_FF*Dalitz_Fit.Integral()/KST800_12_HIST.Integral() );
	KST800_13_HIST.Scale( KST800_13_FF*Dalitz_Fit.Integral()/KST800_13_HIST.Integral() );

	KST892_12_HIST.Scale( KST892_12_FF*Dalitz_Fit.Integral()/KST892_12_HIST.Integral() );
	KST892_13_HIST.Scale( KST892_13_FF*Dalitz_Fit.Integral()/KST892_13_HIST.Integral() );

	KST1425_12_HIST.Scale( KST1425_12_FF*Dalitz_Fit.Integral()/KST1425_12_HIST.Integral() );
	KST1425_13_HIST.Scale( KST1425_13_FF*Dalitz_Fit.Integral()/KST1425_13_HIST.Integral() );

	KST1430_12_HIST.Scale( KST1430_12_FF*Dalitz_Fit.Integral()/KST1430_12_HIST.Integral() );
	KST1430_13_HIST.Scale( KST1430_13_FF*Dalitz_Fit.Integral()/KST1430_13_HIST.Integral() );

	KST1680_12_HIST.Scale( KST1680_12_FF*Dalitz_Fit.Integral()/KST1680_12_HIST.Integral() );
	KST1680_13_HIST.Scale( KST1680_13_FF*Dalitz_Fit.Integral()/KST1680_13_HIST.Integral() );

	NR_HIST.Scale( NR_FF*Dalitz_Fit.Integral()/NR_HIST.Integral() );

	//=============================================================
	//projections
	TH1* hist2D=0;

	TCanvas canvas_3("canvas_3", "Dataset", 500, 500);
	hist2D = Dalitz_Resonances.Project3D("yz");
	hist2D->SetTitle("");
	hist2D->Draw("colz");
	canvas_3.SaveAs("plots/dalitz/Dataset1.pdf");

	TCanvas canvas_4("canvas_4", "Dataset", 500, 500);
	hist2D = Dalitz_Resonances.Project3D("xy");
	hist2D->SetTitle("");
	hist2D->Draw("colz");
	canvas_4.SaveAs("plots/dalitz/Dataset2.pdf");


	TCanvas canvas_5("canvas_5", "Fit", 500, 500);
	hist2D = Dalitz_Fit.Project3D("yz");
	hist2D->SetTitle("");
	hist2D->SetStats(0);
	hist2D->Draw("colz");
	canvas_5.SaveAs("plots/dalitz/FitResult1.pdf");


	TCanvas canvas_6("canvas_4", "Phase-space FLAT", 500, 500);
	hist2D = Dalitz_Fit.Project3D("xy");
	hist2D->SetTitle("");
	hist2D->SetStats(0);
	hist2D->Draw("colz");
	canvas_6.SaveAs("plots/dalitz/FitResult2.pdf");


	//=============================================================
	//projections
	TH1* hist=0;
	const char* axis =0;

	auto KST800_Color  = kViolet-5;
	auto KST892_Color  = kBlue;
	auto KST1425_Color = kGreen-2;
	auto KST1430_Color = kOrange-3;
	auto KST1680_Color = kYellow-2;
	auto NR_Color      = kBlack;

	double X1NDC = 0.262458;
	double Y1NDC = 0.127544;
	double X2NDC = 0.687708;
	double Y2NDC = 0.35;


	//==============================
	axis = "x";

	TCanvas canvas_x("canvas_x", "", 600, 750);



	auto legend_x = TLegend( X1NDC, Y1NDC, X2NDC, Y2NDC);
	//legend.SetHeader("M^{2}(K^{-} #pi_{1}^{+})","C"); // option "C" allows to center the header
	legend_x.SetEntrySeparation(0.3);
	legend_x.SetNColumns(2);
	legend_x.SetBorderSize(0);

	hist= Dalitz_Fit.Project3D(axis)->DrawCopy("hist");
	hist->SetLineColor(2);
	hist->SetLineWidth(2);
	hist->SetMinimum(0.001);
	hist->SetStats(0);
	hist->SetTitle("");

	legend_x.AddEntry(hist,"Fit","l");

	hist= Dalitz_Resonances.Project3D(axis)->DrawCopy("e0same");
	hist->SetMarkerStyle(8);
	hist->SetMarkerSize(0.6);
	hist->SetStats(0);

	legend_x.AddEntry(hist,"Data","lep");

	hist = KST800_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{#kappa}_{12}","l");

	hist = KST800_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{#kappa}_{13}","l");


	hist = KST892_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*(892)}_{12}","l");

	hist = KST892_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*(892)}_{13}","l");


	hist = KST1680_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K_{1}(1680)}_{12}","l");

	hist = KST1680_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K_{1}(1680)}_{13}","l");

	hist = KST1425_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*_{0}(1425)}_{12}","l");

	hist = KST1425_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*_{0}(1425)}_{13}","l");

	hist = KST1430_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*_{2}(1430)}_{12}","l");

	hist = KST1430_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineStyle(2);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{K*_{2}(1430)}_{13}","l");

	hist = NR_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(NR_Color);
	hist->SetLineStyle(5);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"NR","l");

	canvas_x.SaveAs("plots/dalitz/Proj_X.pdf");

	canvas_x.SetLogy(1);

	legend_x.Draw();

	canvas_x.SaveAs("plots/dalitz/Proj_LogX.pdf");

	//=============================================================

	axis = "y";

	TCanvas canvas_y("canvas_y", "", 600, 750);


	auto legend_y = TLegend( X1NDC, Y1NDC, X2NDC, Y2NDC);
	//legend.SetHeader("M^{2}(K^{-} #pi_{1}^{+})","C"); // option "C" allows to center the header
	legend_y.SetEntrySeparation(0.3);
	legend_y.SetNColumns(2);
	legend_y.SetBorderSize(0);

	hist= Dalitz_Fit.Project3D(axis)->DrawCopy("hist");
	hist->SetLineColor(2);
	hist->SetLineWidth(2);
	hist->SetMinimum(0.001);
	hist->SetStats(0);
	hist->SetTitle("");

	legend_y.AddEntry(hist,"Fit","l");

	hist= Dalitz_Resonances.Project3D(axis)->DrawCopy("e0same");
	hist->SetMarkerStyle(8);
	hist->SetMarkerSize(0.6);
	hist->SetStats(0);

	legend_y.AddEntry(hist,"Data","lep");

	hist = KST800_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{#kappa}_{12}","l");

	hist = KST800_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{#kappa}_{13}","l");


	hist = KST892_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*(892)}_{12}","l");

	hist = KST892_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*(892)}_{13}","l");


	hist = KST1680_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K_{1}(1680)}_{12}","l");

	hist = KST1680_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K_{1}(1680)}_{13}","l");

	hist = KST1425_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*_{0}(1425)}_{12}","l");

	hist = KST1425_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*_{0}(1425)}_{13}","l");

	hist = KST1430_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*_{2}(1430)}_{12}","l");

	hist = KST1430_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineStyle(2);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{K*_{2}(1430)}_{13}","l");

	hist = NR_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(NR_Color);
	hist->SetLineStyle(5);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"NR","l");

	canvas_y.SaveAs("plots/dalitz/Proj_Y.pdf");

	canvas_y.SetLogy(1);

	legend_y.Draw();

	canvas_y.SaveAs("plots/dalitz/Proj_LogY.pdf");


	//=============================================================



	axis = "z";

	TCanvas canvas_z("canvas_z", "", 600, 750);

	auto legend_z = TLegend( X1NDC, Y1NDC, X2NDC, Y2NDC);
	//legend.SetHeader("M^{2}(K^{-} #pi_{1}^{+})","C"); // option "C" allows to center the header
	legend_z.SetEntrySeparation(0.3);
	legend_z.SetNColumns(2);
	legend_z.SetBorderSize(0);

	hist= Dalitz_Fit.Project3D(axis)->DrawCopy("hist");
	hist->SetLineColor(2);
	hist->SetLineWidth(2);
	hist->SetMinimum(0.001);
	hist->SetStats(0);
	hist->SetTitle("");

	legend_z.AddEntry(hist,"Fit","l");

	hist= Dalitz_Resonances.Project3D(axis)->DrawCopy("e0same");
	hist->SetMarkerStyle(8);
	hist->SetMarkerSize(0.6);
	hist->SetStats(0);

	legend_z.AddEntry(hist,"Data","lep");

	hist = KST800_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{#kappa}_{12}","l");

	hist = KST800_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST800_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{#kappa}_{13}","l");


	hist = KST892_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*(892)}_{12}","l");

	hist = KST892_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST892_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*(892)}_{13}","l");


	hist = KST1680_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K_{1}(1680)}_{12}","l");

	hist = KST1680_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1680_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K_{1}(1680)}_{13}","l");

	hist = KST1425_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*_{0}(1425)}_{12}","l");

	hist = KST1425_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(KST1425_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*_{0}(1425)}_{13}","l");

	hist = KST1430_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*_{2}(1430)}_{12}","l");

	hist = KST1430_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(KST1430_Color);
	hist->SetLineStyle(2);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{K*_{2}(1430)}_{13}","l");

	hist = NR_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(NR_Color);
	hist->SetLineStyle(5);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"NR","l");

	canvas_z.SaveAs("plots/dalitz/Proj_Z.pdf");

	canvas_z.SetLogy(1);

	legend_z.Draw();

	canvas_z.SaveAs("plots/dalitz/Proj_LogZ.pdf");

	//=============================================================

	TCanvas canvas_7("canvas_7", "Normalization", 500, 500);
	Normalization.Draw("colz");


	m_app->Run();

#endif

	return 0;
}


template<typename Backend, typename Model, typename Container >
size_t generate_dataset(Backend const& system, Model const& model, std::array<double, 3> const& masses, Container& decays, size_t nevents, size_t bunch_size)
{
	const double D_MASS         = masses[0];// D+ mass
	const double K_MASS         = masses[1];// K+ mass
	const double PI_MASS        = masses[2];// pi mass

	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi pi
	hydra::PhaseSpace<3> phsp{K_MASS, PI_MASS, PI_MASS};

	//allocate memory to hold the final states particles
	hydra::Decays<3, Backend > _data(bunch_size);

	std::srand(7531594562);

	do {
		phsp.SetSeed(std::rand());

		//generate the final state particles
		phsp.Generate(D, _data.begin(), _data.end());

		auto last = _data.Unweight(model, 1.0);

		decays.insert(decays.size()==0? decays.begin():decays.end(),
				_data.begin(), _data.begin()+last );

	} while(decays.size()<nevents );

	decays.erase(decays.begin()+nevents, decays.end());

	return decays.size();

}



template<typename Amplitude, typename Model>
double fit_fraction( Amplitude const& amp, Model const& model, std::array<double, 3> const& masses, size_t nentries)
{
	const double D_MASS         = masses[0];// D+ mass
	const double K_MASS         = masses[1];// K+ mass
	const double PI_MASS        = masses[2];// pi mass

	//--------------------
	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi pi
	hydra::PhaseSpace<3> phsp{K_MASS, PI_MASS, PI_MASS};

	//norm lambda
	auto Norm = hydra::wrap_lambda( [] __hydra_dual__ (unsigned int n, hydra::complex<double>* x){

		return hydra::norm(x[0]);
	});

	//functor
	auto functor = hydra::compose(Norm, amp);


	auto amp_int   = phsp.AverageOn(hydra::device::sys, D, functor, nentries);
	auto model_int = phsp.AverageOn(hydra::device::sys, D, model,   nentries);


	return amp_int.first/model_int.first;

}

template<typename Amplitude>
TH3D histogram_component( Amplitude const& amp, std::array<double, 3> const& masses, const char* name, size_t nentries)
{
	const double D_MASS         = masses[0];// D+ mass
	const double K_MASS         = masses[1];// K+ mass
	const double PI_MASS        = masses[2];// pi mass

	TH3D Component(name,
			";"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} #pi^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(#pi^{+} #pi^{+}) [GeV^{2}/c^{4}]",
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(K_MASS  + PI_MASS,2), pow(D_MASS - PI_MASS,2),
			100, pow(PI_MASS + PI_MASS,2), pow(D_MASS -  K_MASS,2));

	//--------------------
	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);
	// Create PhaseSpace object for B0-> K pi pi
	hydra::PhaseSpace<3> phsp{K_MASS, PI_MASS, PI_MASS};

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[] __hydra_dual__ (unsigned int n, hydra::Vector4R* p ){

		double   M2_12 = (p[0]+p[1]).mass2();
		double   M2_13 = (p[0]+p[2]).mass2();
		double   M2_23 = (p[1]+p[2]).mass2();

		return hydra::make_tuple(M2_12, M2_13, M2_23);
	});

	//norm lambda
	auto Norm = hydra::wrap_lambda(
			[] __hydra_dual__ (unsigned int n, hydra::complex<double>* x){

		return hydra::norm(x[0]);
	});

	//functor
	auto functor = hydra::compose(Norm, amp);

	hydra::Decays<3, hydra::device::sys_t > events(nentries);

	phsp.Generate(D, events.begin(), events.end());

	events.Reweight(functor);

	auto particles        = events.GetUnweightedDecays();
	auto dalitz_variables = hydra::make_range( particles.begin(), particles.end(), dalitz_calculator);
	auto dalitz_weights   = events.GetWeights();

	//model dalitz histogram
	hydra::SparseHistogram<double, 3,  hydra::device::sys_t> Hist_Component{
		{100,100,100},
		{pow(K_MASS + PI_MASS,2), pow(K_MASS + PI_MASS,2),  pow(PI_MASS + PI_MASS,2)},
		{pow(D_MASS - PI_MASS,2), pow(D_MASS - PI_MASS ,2), pow(D_MASS - K_MASS,2)}
	};

	Hist_Component.Fill( dalitz_variables.begin(),
					dalitz_variables.end(), dalitz_weights.begin()  );

	for(auto entry : Hist_Component){

		size_t bin     = hydra::get<0>(entry);
		double content = hydra::get<1>(entry);
		unsigned int bins[3];
		Hist_Component.GetIndexes(bin, bins);
		Component.SetBinContent(bins[0]+1, bins[1]+1, bins[2]+1, content);

	}

	return Component;

}



#endif /* DALITZ_PLOT_INL_ */
