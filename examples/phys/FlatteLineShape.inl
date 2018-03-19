/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * FlatteLineShape.inl
 *
 *  Created on: 03/09/2018
 *      Author: Juan B de S Leite
 */

#ifndef FlatteLineShape_INL
#define FlatteLineShape_INL

#include <iostream>
#include <fstream>
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
#include <hydra/functions/FlatteLineShape.h>
#include <hydra/functions/CosHelicityAngle.h>
#include <hydra/functions/ZemachFunctions.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

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

        double theta = fCosDecayAngle((p1 + p2 + p3), (p1 + p2), p1);
        double angular = fAngularDist(theta);
        auto r = hydra::complex<double>(_par[0], _par[1]) * fLineShape((p1 + p2).mass()) * angular;

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

template<unsigned int CHANNEL,hydra::Wave L> class Flatte: public hydra::BaseFunctor<Flatte<CHANNEL,L>, hydra::complex<double>, 5>
{

    constexpr static unsigned int _I1 = CHANNEL-1;
    constexpr static unsigned int _I2 = (CHANNEL!=3)*CHANNEL;
    constexpr static unsigned int _I3 = 3-( (CHANNEL-1) + (CHANNEL!=3)*CHANNEL );

    using hydra::BaseFunctor<Flatte<CHANNEL,L>, hydra::complex<double>, 5>::_par;

public:

    Flatte() = delete;

    Flatte(hydra::Parameter const& c_re, hydra::Parameter const& c_im, hydra::Parameter const& mean, hydra::Parameter const& rho1 , hydra::Parameter const& rho2,
           double mother_mass,	double daugther1_mass, double daugther2_mass, double daugther3_mass, double radi):
            hydra::BaseFunctor<Flatte<CHANNEL,L>, hydra::complex<double>, 5>{c_re, c_im, mean, rho1, rho2},
            fLineShape(mean,rho1,rho2,mother_mass,daugther1_mass,daugther2_mass,daugther3_mass,radi)
    {}

    __hydra_dual__ Flatte( Flatte<CHANNEL,L> const& other):
            hydra::BaseFunctor<Flatte<CHANNEL,L>, hydra::complex<double>, 5>(other),
            fLineShape(other.GetLineShape())
    {}

    __hydra_dual__ inline
    Flatte<CHANNEL,L>&
    operator=( Flatte<CHANNEL,L> const& other)
    {
        if(this==&other) return *this;

        hydra::BaseFunctor<Flatte<CHANNEL,L>, hydra::complex<double>, 5>::operator=(other);
        fLineShape=other.GetLineShape();

        return *this;
    }

    __hydra_dual__ inline
    hydra::FlatteLineShape<CHANNEL,L> const& GetLineShape() const {	return fLineShape; }


    __hydra_dual__ hydra::complex<double> Evaluate(unsigned int n, hydra::Vector4R* p)  const {

        hydra::Vector4R p1 = p[_I1];
        hydra::Vector4R p2 = p[_I2];
        hydra::Vector4R p3 = p[_I3];

        fLineShape.SetParameter(0,_par[2]);
        fLineShape.SetParameter(1,_par[3]);
        fLineShape.SetParameter(2,_par[4]);

        double theta = fCosDecayAngle( (p1+p2+p3), (p1+p2), p1 );
        double angular = fAngularDist(theta);

        auto r = hydra::complex<double>(_par[0],_par[1])*fLineShape((p1+p2).mass())*angular;

        return r;
    }

private:

    mutable hydra::FlatteLineShape<CHANNEL> fLineShape;
    hydra::CosHelicityAngle fCosDecayAngle;
    hydra::ZemachFunction<L> fAngularDist;


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
   

    double Phi_MASS		  = 1.019461;
    double Phi_Width	  = 0.004266;
    double Phi_RC		  = 1.0;
    double Phi_IMC		  = 0.0;

    double f0_MASS        = 0.965;
	double f0_MAG		  = 12.341;
	double f0_Phase		  = -62.852*(M_PI / 180.0 ) + M_PI; // 2.044618312126317
    double f0_RC          = f0_MAG * cos(f0_Phase); //-5.631081433470062
    double f0_IMC         = f0_MAG * sin(f0_Phase); //10.981402592093087
    double f0_rho1        = 0.165;
    double f0_rho2        = 4.21*f0_rho1; // 0.69465

    double D_MASS         = 1.86962;
    double Kplus_MASS     = 0.493677;  // K+ mass
    double Kminus_MASS    = Kplus_MASS;


    //======================================================
	//Phi
	auto mass    = hydra::Parameter::Create().Name("MASS_Phi" ).Value(Phi_MASS ).Error(0.01).Fixed();
	auto width   = hydra::Parameter::Create().Name("WIDTH_Phi").Value(Phi_Width).Error(0.001).Fixed();
	auto coef_re = hydra::Parameter::Create().Name("Phi_RC" ).Value(Phi_RC).Fixed();
	auto coef_im = hydra::Parameter::Create().Name("Phi_IM" ).Value(Phi_IMC).Fixed();

	Resonance<1, hydra::PWave> Phi_Resonance_12(coef_re, coef_im, mass, width,
		    	D_MASS,	Kminus_MASS, Kplus_MASS, Kplus_MASS , 1.5);

	Resonance<3, hydra::PWave> Phi_Resonance_13(coef_re, coef_im, mass, width,
			    	D_MASS,	Kminus_MASS, Kminus_MASS, Kplus_MASS , 1.5);

	auto Phi_Resonance = (Phi_Resonance_12 - Phi_Resonance_13);



	 //======================================================

    //f0
    auto coef_ref0 = hydra::Parameter::Create().Name("f0_RC").Value(f0_RC).Error(0.0001).Limits(-f0_RC*20.0,+f0_RC*20.0);
    auto coef_imf0 = hydra::Parameter::Create().Name("f0_IM").Value(f0_IMC).Error(0.0001).Limits(-f0_IMC*20.0,+f0_IMC*20.0);
    auto f0Mass = hydra::Parameter::Create().Name("MASS_f0").Value(f0_MASS).Fixed();
    auto f0g1 = hydra::Parameter::Create().Name("f0_g1").Value(f0_rho1).Fixed();
    auto rg1og2 = hydra::Parameter::Create().Name("f0_g1xg2").Value(f0_rho2).Fixed();


    Flatte<1,hydra::SWave> f0_Resonance_12(coef_ref0,coef_imf0,f0Mass,f0g1,rg1og2,D_MASS,Kminus_MASS,Kplus_MASS,Kplus_MASS,1.5);
    Flatte<3,hydra::SWave> f0_Resonance_13(coef_ref0,coef_imf0,f0Mass,f0g1,rg1og2,D_MASS,Kminus_MASS,Kplus_MASS,Kplus_MASS,1.5);

    auto f0_Resonance = (f0_Resonance_12 + f0_Resonance_13);

	//======================================================

	//Total: Model |sum{ Resonaces }|^2

	//parametric lambda
	auto Norm = hydra::wrap_lambda( []__host__  __device__ ( unsigned int n, hydra::complex<double>* x){

				hydra::complex<double> r(0,0);

				for(unsigned int i=0; i<n; i++)
                    r += x[i];

				return hydra::norm(r);
	});

	//model-functor
	auto Model = hydra::compose(Norm,
		    Phi_Resonance, f0_Resonance
	);

	//--------------------
	//generator
	hydra::Vector4R B0(D_MASS, 0.0, 0.0, 0.0);
	// Create PhaseSpace object for B0-> K pi J/psi
	hydra::PhaseSpace<3> phsp{Kminus_MASS, Kplus_MASS, Kplus_MASS};

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[]__host__ __device__(unsigned int n, hydra::Vector4R* p ){

		double   M2_12 = (p[0]+p[1]).mass2();
		double   M2_13 = (p[0]+p[2]).mass2();
		double   M2_23 = (p[1]+p[2]).mass2();

		return hydra::make_tuple(M2_12, M2_13, M2_23);
	});


#ifdef 	_ROOT_AVAILABLE_

	TH3D Dalitz_Resonances("Dalitz_Resonances",
			"Dalitz - Toy Data -;"
			"M^{2}(K^{-} K_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} K_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K_{1}^{+} K_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kplus_MASS + Kplus_MASS,2), pow(D_MASS -  Kminus_MASS,2));


	TH3D Dalitz_Fit("Dalitz_Fit",
			"Dalitz - Fit -;"
			"M^{2}(K^{-} K_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} K_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K_{1}^{+} K_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kplus_MASS + Kplus_MASS,2), pow(D_MASS -  Kminus_MASS,2));


	//control plots
	TH2D Normalization("normalization",
			"Model PDF Normalization;Norm;Error",
			200, 275.0, 305.0,
			200, 0.58, 0.64);


	TH3D    Phi_12_HIST,Phi_13_HIST,
            f0_12_HIST,f0_13_HIST ;

	double  Phi_12_FF,  Phi_13_FF,
            f0_12_FF,f0_13_FF,
            Phi_all_FF, f0_all_FF
            ;
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

		generate_dataset(hydra::device::sys, Model,  {D_MASS, Kminus_MASS, Kplus_MASS}, toy_data, nentries, 3*nentries);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;

		//output
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "----------------- Device ----------------"<< std::endl;
		std::cout << "| D+ -> K- K+ K+"                         << std::endl;
		std::cout << "| Number of events :"<< toy_data.size()         << std::endl;
		std::cout << "| Time (ms)        :"<< elapsed.count()   << std::endl;
		std::cout << "-----------------------------------------"<< std::endl;


		std::cout << std::endl <<"Toy Dataset size: "<< toy_data.size() << std::endl;

        std::ofstream writer;
        writer.open("toyData.txt");

        if(!writer.is_open()){
            std::cout << "file not open" << std::endl;
        }else {

            for (auto event : toy_data) {

                double weight = hydra::get<0>(event);
                hydra::Vector4R Kminus = hydra::get<1>(event);
                hydra::Vector4R Kplus1 = hydra::get<2>(event);
                hydra::Vector4R Kplus2 = hydra::get<3>(event);

                double MKminusKplus1 = (Kminus + Kplus1).mass2();
                double MKminusKplus2 = (Kminus + Kplus2).mass2();

                writer << MKminusKplus1 << '\t' << MKminusKplus2 << std::endl;
            }

            std::cout << "toyfile.txt generated" <<std::endl;
        }



        writer.close();



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
			{pow(Kminus_MASS + Kplus_MASS,2), pow(Kminus_MASS + Kplus_MASS,2),  pow(Kplus_MASS + Kplus_MASS,2)},
			{pow(D_MASS - Kplus_MASS,2), pow(D_MASS - Kplus_MASS ,2), pow(D_MASS - Kminus_MASS,2)}
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
				hydra::PhaseSpaceIntegrator<3, hydra::device::sys_t>(D_MASS, {Kminus_MASS, Kplus_MASS, Kplus_MASS}, 500000));



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

		FunctionMinimum minimum_d =  FunctionMinimum( migrad_d(5000,100) );

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
		std::cout << "| D+ -> K- K+ K+"                         << std::endl;
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
			{pow(Kminus_MASS + Kplus_MASS,2), pow(Kminus_MASS + Kplus_MASS,2),  pow(Kplus_MASS + Kplus_MASS,2)},
			{pow(D_MASS - Kplus_MASS,2), pow(D_MASS - Kplus_MASS ,2), pow(D_MASS - Kminus_MASS,2)}
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

		auto Phi_12 = fcn.GetPDF().GetFunctor().GetFunctor(_1).GetFunctor(_0);
		auto Phi_13 = fcn.GetPDF().GetFunctor().GetFunctor(_1).GetFunctor(_1);
        auto Phi_all = fcn.GetPDF().GetFunctor().GetFunctor(_1);
		auto f0_12  = fcn.GetPDF().GetFunctor().GetFunctor(_2).GetFunctor(_0);
        auto f0_13  = fcn.GetPDF().GetFunctor().GetFunctor(_2).GetFunctor(_1);
        auto f0_all = fcn.GetPDF().GetFunctor().GetFunctor(_2);


		//==================================
		// Draw components
		//==================================
		Phi_12_HIST  =	histogram_component(Phi_12 , {D_MASS, Kminus_MASS, Kplus_MASS}, "Phi_12_HIST", nentries);
		Phi_13_HIST  =	histogram_component(Phi_13 , {D_MASS, Kminus_MASS, Kplus_MASS}, "Phi_13_HIST", nentries);
		f0_12_HIST   =	histogram_component(f0_12 , {D_MASS, Kminus_MASS, Kplus_MASS}, "f0_12_HIST", nentries);
        f0_13_HIST   =	histogram_component(f0_13 , {D_MASS, Kminus_MASS, Kplus_MASS}, "f0_13_HIST", nentries);


		//==================================
		// Fit fractions
		//==================================
		Phi_12_FF  =	fit_fraction(Phi_12 , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);
		Phi_13_FF  =	fit_fraction(Phi_13 , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);
        Phi_all_FF  =	fit_fraction(Phi_all , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);
		f0_12_FF   =	fit_fraction(f0_12 , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);
        f0_13_FF   =	fit_fraction(f0_13 , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);
        f0_all_FF  =	fit_fraction(f0_all , Opt_Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries);

		std::cout << "Phi_12_FF :" << Phi_12_FF << std::endl;
		std::cout << "Phi_13_FF :" << Phi_13_FF << std::endl;
        std::cout << "Phi_all_FF :" << Phi_all_FF << std::endl;
		std::cout << "f0_12_FF :" << f0_12_FF << std::endl;
        std::cout << "f0_13_FF :" << f0_13_FF << std::endl;
        std::cout << "f0_all_FF :" << f0_all_FF << std::endl;
		std::cout << "Sum :"
				  << Phi_12_FF  + Phi_13_FF  + f0_12_FF + f0_13_FF  << std::endl;

		std::cout << "Phi_12_FF :" << fit_fraction(Phi_Resonance_12, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;
		std::cout << "Phi_13_FF :" << fit_fraction(Phi_Resonance_13, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;
		std::cout << "Phi_FF :" << fit_fraction(Phi_Resonance, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;
		std::cout << "f0_12_FF: " << fit_fraction(f0_Resonance_12, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;
		std::cout << "f0_13_FF: " << fit_fraction(f0_Resonance_13, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;
        std::cout << "f0_FF: " << fit_fraction(f0_Resonance, Model, {D_MASS, Kminus_MASS, Kplus_MASS},  nentries) << std::endl;

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

	Phi_12_HIST.Scale(Phi_12_FF*Dalitz_Fit.Integral()/Phi_12_HIST.Integral() );
	Phi_13_HIST.Scale(Phi_13_FF*Dalitz_Fit.Integral()/Phi_13_HIST.Integral() );
	f0_12_HIST.Scale( f0_12_FF*Dalitz_Fit.Integral()/f0_12_HIST.Integral() );
    f0_13_HIST.Scale( f0_13_FF*Dalitz_Fit.Integral()/f0_13_HIST.Integral() );

    //=============================================================
	//projections
	TH1* hist2D=0;

	TCanvas canvas_3("canvas_3", "Dataset", 500, 500);
	hist2D = Dalitz_Resonances.Project3D("yz");
	hist2D->SetTitle("");
	hist2D->Draw("colz");
	canvas_3.SaveAs("Dataset1.png");

	TCanvas canvas_4("canvas_4", "Dataset", 500, 500);
	hist2D = Dalitz_Resonances.Project3D("yx");
	hist2D->SetTitle("");
	hist2D->Draw("colz");
	canvas_4.SaveAs("Dataset2.png");


	TCanvas canvas_5("canvas_5", "Fit", 500, 500);
	hist2D = Dalitz_Fit.Project3D("yz");
	hist2D->SetTitle("");
	hist2D->SetStats(0);
	hist2D->Draw("colz");
	canvas_5.SaveAs("FitResult1.png");


	TCanvas canvas_6("canvas_4", "Phase-space FLAT", 500, 500);
	hist2D = Dalitz_Fit.Project3D("yx");
	hist2D->SetTitle("");
	hist2D->SetStats(0);
	hist2D->Draw("colz");
	canvas_6.SaveAs("FitResult2.png");


	//=============================================================
	//projections
	TH1* hist=0;
	const char* axis =0;

	auto Phi_Color  = kViolet-5;
	auto f0_Color = kGreen;

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

	hist = Phi_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{#Phi}_{12}","l");

	hist = Phi_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_x.AddEntry(hist,"{#Phi}_{13}","l");

    hist = f0_12_HIST.Project3D(axis)->DrawCopy("histCsame");
    //hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_x.AddEntry(hist,"{f0}_{12}","l");

    hist = f0_13_HIST.Project3D(axis)->DrawCopy("histCsame");
    hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_x.AddEntry(hist,"{f0}_{13}","l");



	canvas_x.SaveAs("Proj_X.png");

	canvas_x.SetLogy(1);

	legend_x.Draw();

	canvas_x.SaveAs("Proj_LogX.png");

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

	hist = Phi_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{#Phi}_{12}","l");

	hist = Phi_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_y.AddEntry(hist,"{#Phi}_{13}","l");

	hist = f0_12_HIST.Project3D(axis)->DrawCopy("histCsame");
    //hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_y.AddEntry(hist,"{f0}_{12}","l");

    hist = f0_13_HIST.Project3D(axis)->DrawCopy("histCsame");
    hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_y.AddEntry(hist,"{f0}_{13}","l");



	canvas_y.SaveAs("Proj_Y.png");

	canvas_y.SetLogy(1);

	legend_y.Draw();

	canvas_y.SaveAs("Proj_LogY.png");


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

	hist = Phi_12_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{#Phi}_{12}","l");

	hist = Phi_13_HIST.Project3D(axis)->DrawCopy("histCsame");
	hist->SetLineStyle(2);
	hist->SetLineColor(Phi_Color);
	hist->SetLineWidth(2);

	legend_z.AddEntry(hist,"{#Phi}_{13}","l");

	hist = f0_12_HIST.Project3D(axis)->DrawCopy("histCsame");
    //hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_z.AddEntry(hist,"{f0}_{12}","l");

    hist = f0_13_HIST.Project3D(axis)->DrawCopy("histCsame");
    hist->SetLineStyle(2);
    hist->SetLineColor(f0_Color);
    hist->SetLineWidth(2);

    legend_z.AddEntry(hist,"{f0}_{13}","l");


	canvas_z.SaveAs("Proj_Z.png");

	canvas_z.SetLogy(1);

	legend_z.Draw();

	canvas_z.SaveAs("Proj_LogZ.png");

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
	const double Kminus_MASS         = masses[1];// K- mass
	const double Kplus_MASS        = masses[2];// K+ mass

	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for B0-> K pi pi
	hydra::PhaseSpace<3> phsp{Kminus_MASS, Kplus_MASS, Kplus_MASS};

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
	const double Kminus_MASS         = masses[1];// K- mass
	const double Kplus_MASS        = masses[2];// K+ mass

	//--------------------
	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);

	// Create PhaseSpace object for D-> K K K
	hydra::PhaseSpace<3> phsp{Kminus_MASS, Kplus_MASS, Kplus_MASS};


	//norm lambda
	auto Norm = hydra::wrap_lambda( []__host__  __device__ (unsigned int n, hydra::complex<double>* x){

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
	const double Kminus_MASS         = masses[1];// K+ mass
	const double Kplus_MASS        = masses[2];// K- mass

	TH3D Component(name,
			"M^{2}(K^{-} K_{1}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K^{-} K_{2}^{+}) [GeV^{2}/c^{4}];"
			"M^{2}(K_{1}^{+} K_{2}^{+}) [GeV^{2}/c^{4}]",
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kminus_MASS  + Kplus_MASS,2), pow(D_MASS - Kplus_MASS,2),
			100, pow(Kplus_MASS + Kplus_MASS,2), pow(D_MASS -  Kminus_MASS,2));

	//--------------------
	//generator
	hydra::Vector4R D(D_MASS, 0.0, 0.0, 0.0);
	// Create PhaseSpace object for B0-> K pi pi
	hydra::PhaseSpace<3> phsp{Kplus_MASS, Kminus_MASS, Kminus_MASS};

	// functor to calculate the 2-body masses
	auto dalitz_calculator = hydra::wrap_lambda(
			[]__host__ __device__(unsigned int n, hydra::Vector4R* p ){

		double   M2_12 = (p[0]+p[1]).mass2();
		double   M2_13 = (p[0]+p[2]).mass2();
		double   M2_23 = (p[1]+p[2]).mass2();

		return hydra::make_tuple(M2_12, M2_13, M2_23);
	});

	//norm lambda
	auto Norm = hydra::wrap_lambda(
			[]__host__  __device__ (unsigned int n, hydra::complex<double>* x){

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
			{pow(Kminus_MASS + Kplus_MASS,2), pow(Kminus_MASS + Kplus_MASS,2),  pow(Kplus_MASS + Kplus_MASS,2)},
			{pow(D_MASS - Kplus_MASS,2), pow(D_MASS - Kplus_MASS ,2), pow(D_MASS - Kminus_MASS,2)}
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



#endif /* FlatteLineShape_INL */
