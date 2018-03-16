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
 * dalitz_plot2.inl
 *
 *  Created on: Mar 16, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZ_PLOT2_INL_
#define DALITZ_PLOT2_INL_

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
#include <hydra/Random.h>

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
#include <hydra/functions/DalitzAngularDistribution.h>

//Minuit2
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"

 //Include classes from ROOT to fill
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

/*
Resonance with Breit-Wigner lineshape and Zemach angular distribution.
Representation in the [M_12, cos(theta)] plane.
*/
template<unsigned int CHANNEL, hydra::Wave L>
class Resonance: public hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>
{
	using hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>::_par;

	static constexpr unsigned int _M12=CHANNEL-1;
	static constexpr unsigned int _M23= CHANNEL==3?0:CHANNEL;


public:

	Resonance() = delete;

	Resonance(hydra::Parameter const& c_re, hydra::Parameter const& c_im,
			hydra::Parameter const& mass, hydra::Parameter const& width,
			double mother_mass,	double daugther1_mass,
			double daugther2_mass, double bachelor_mass,
			double radi):
			hydra::BaseFunctor<Resonance<CHANNEL,L>, hydra::complex<double>, 4>{c_re, c_im, mass, width},
			fLineShape(mass, width, mother_mass, daugther1_mass, daugther2_mass, bachelor_mass, radi),
			fAngularDistribution(mother_mass, daugther1_mass, daugther2_mass, bachelor_mass)
	{}


    __hydra_dual__
	Resonance( Resonance< CHANNEL,L> const& other):
	hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>(other),
	fLineShape(other.GetLineShape()),
	fAngularDistribution(other.GetAngularDistribution())
	{}

    __hydra_dual__  inline
	Resonance< CHANNEL ,L>&
	operator=( Resonance< CHANNEL ,L> const& other)
	{
		if(this==&other) return *this;

		hydra::BaseFunctor<Resonance<CHANNEL ,L>, hydra::complex<double>, 4>::operator=(other);
		fLineShape=other.GetLineShape();
		fAngularDistribution=other.GetAngularDistribution();

		return *this;
	}

    __hydra_dual__  inline
	hydra::BreitWignerLineShape<L> const& GetLineShape() const {	return fLineShape; }

    __hydra_dual__  inline
    hydra::DalitzAngularDistribution<L> const& GetAngularDistribution() const { return fAngularDistribution; }

    __hydra_dual__  inline
    hydra::complex<double> Evaluate(unsigned int n, double* masses)  const {

    	fLineShape.SetParameter(0, _par[2]);
    	fLineShape.SetParameter(1, _par[3]);

    	auto r = hydra::complex<double>(_par[0], _par[1])*
    			 fLineShape(masses[_M12])*
    			 fAngularDistribution(masses[_M12], masses[_M23]);

    	return r;

    }



private:

    __hydra_dual__  inline
    void calculate_four_vectors(const double mass, const double cos_theta,
    		hydra::Vector4R& P1, hydra::Vector4R& P2, hydra::Vector4R& P3 ){

    	const double S      = fMotherMass*fMotherMass;
    	const double M1_Sq  = fDaughter1Mass*fDaughter1Mass;
    	const double M2_Sq  = fDaughter2Mass*fDaughter2Mass;
    	const double M3_Sq  = fDaughter3Mass*fDaughter3Mass;
    	const double M12_Sq = mass*mass;
    	const double M0_P   = ::sqrt( lambda(S, M12_Sq, M3_Sq ))/2*mass;
    	const double M0_E   = (S + M12_Sq - M3_Sq)/2*mass;
    	const double gamma  = ::sqrt(1 - hydra::pow<double,2>(M0_P/M0_E));


    	//daughter 1
    	const double D1_P = ::sqrt( lambda(M12_Sq, M1_Sq, M2_Sq ) )/2*mass;
    	const double D1_E = (M12_Sq + M1_Sq - M2_Sq)/2*mass;

    	P1.set( gamma*( D1_E - (M0_P/M0_E)*D1_P*cos_theta),       //Pe
    			gamma*( D1_P*cos_theta - (M0_P/M0_E)*D1_E ) ,     //Px
    			D1_P*::sqrt( 1- hydra::pow<double,2>(cos_theta)), //Py
    			0                                                 //Pz
    		);


    	//daughter 2
    	const double D2_P = ::sqrt( lambda(M12_Sq, M1_Sq, M2_Sq ) )/2*mass;
    	const double D2_E = (M12_Sq + M2_Sq - M1_Sq)/2*mass;

    	P2.set( gamma*( D2_E + (M0_P/M0_E)*D2_P*cos_theta),       //Pe
    			gamma*( -D2_P*cos_theta - (M0_P/M0_E)*D2_E ) ,    //Px
    			-D2_P*::sqrt( 1- hydra::pow<double,2>(cos_theta)),//Py
    			0                                                 //Pz
    	);


    	//daughter 3
    	const double D3_P = ::sqrt( lambda(S, M12_Sq, M3_Sq ) )/2*mass;
    	const double D3_E = (S + M12_Sq - M3_Sq )/2*mass;

    	P3.set( gamma*( D2_E + (M0_P/M0_E)*D3_P), //Pe
    			gamma*( D2_P - (M0_P/M0_E)*D3_E ),//Px
    			0,                                //Py
    			0                                 //Pz
    	);


    }

    __hydra_dual__  inline
    double lambda(const double a, const double b, const double c){

    	return (a-b-c)*(a-b-c) - 4*b*c;
    }

	double fDaughter1Mass;
	double fDaughter2Mass;
	double fDaughter3Mass;
	double fMotherMass;

};




#endif /* DALITZ_PLOT2_INL_ */
