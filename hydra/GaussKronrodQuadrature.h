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
 * GaussKronrodQuadrature.h
 *
 *  Created on: 31/01/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODQUADRATURE_H_
#define GAUSSKRONRODQUADRATURE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/GaussKronrodRules.h>
#include <hydra/detail/functors/ProcessGaussKronrodQuadrature.h>
#include <hydra/multivector.h>
#include <hydra/detail/Integrator.h>

#include <hydra/detail/Print.h>
#include <tuple>

namespace hydra {


template<  size_t N, typename  BACKEND>
class GaussKronrodQuadrature;

template<size_t NRULE, size_t NBIN, hydra::detail::Backend  BACKEND>
class GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND>>:
public Integrator< GaussKronrodQuadrature<NRULE, NBIN, hydra::detail::BackendPolicy<BACKEND> > >
{
public:
	//tag
		typedef void hydra_integrator_tag;
	    typedef thrust::tuple<double, double, double, double, double> row_t;

		typedef thrust::host_vector<row_t>   row_list_h;
		typedef thrust::device_vector<row_t> row_list_d;

		typedef multivector<row_list_h> table_h;
		typedef multivector<row_list_d> table_d;


	GaussKronrodQuadrature(GReal_t xlower, GReal_t xupper):
		fXLower(xlower),
		fXUpper(xupper),
		fRule(GaussKronrodRuleSelector<NRULE>().fRule)
	{

		this->SetBins();
		this->SetCallTable();
	}

	GaussKronrodQuadrature(GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND> > const& other)
	{
		this->fXLower = other.GetXLower();
		this->fXUpper = other.GetXUpper();
		this->fRule   = other.GetfRule();
		this->SetBins();
		this->SetCallTable();
	}

	GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND> >&
	operator=(GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND> > const& other)
	{
		if(this==&other) return *this;

			this->fXLower = other.GetXLower();
			this->fXUpper = other.GetXUpper();
			this->fRule   = other.GetfRule();
			this->SetBins();
			this->SetCallTable();

			return *this;
	}

	template< hydra::detail::Backend  BACKEND2 >
	GaussKronrodQuadrature(GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND2> > const& other)
	{
            this->fXLower = other.GetXLower();
            this->fXUpper = other.GetXUpper();
            this->fRule   = other.GetfRule();
			this->SetBins();
			this->SetCallTable();
	}

	template< hydra::detail::Backend  BACKEND2 >
	GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND> >&
	operator=(GaussKronrodQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND2> > const& other)
	{
		if(this==&other) return *this;

		this->fXLower = other.GetXLower();
		this->fXUpper = other.GetXUpper();
		this->fRule   = other.GetfRule();
		this->SetBins();
		this->SetCallTable();

		return *this;
	}

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);

	void Print()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "GaussKronrodQuadrature begin: " << HYDRA_ENDL;
		HYDRA_MSG << "XLower: " << fXLower << HYDRA_ENDL;
		HYDRA_MSG << "XUpper: " << fXUpper << HYDRA_ENDL;
		HYDRA_MSG << "NBins: " << NBIN << HYDRA_ENDL;
		for(size_t i=0; i<NBIN; i++ ){
			HYDRA_MSG << "bin " << i <<" = ["<< fBins[i] <<", " << fBins[i+1]<< "]"<< HYDRA_ENDL;
		    	}
		fRule.Print();
		HYDRA_MSG << "GaussKronrodQuadrature end. " << HYDRA_ENDL;
	}

	GReal_t GetXLower() const
	{
		return fXLower;
	}

	void SetXLower(GReal_t xLower)
	{
		fXLower = xLower;
		this->SetBins();
		this->SetCallTable();
	}

	GReal_t GetXUpper() const
	{
		return fXUpper;
	}

	void SetXUpper(GReal_t xUpper)
	{
		fXUpper = xUpper;
		this->SetBins();
		this->SetCallTable();
	}

	const GaussKronrodRule<NRULE>& GetRule() const
	{
		return fRule;
	}

private:

    void SetBins()
    {

    	GReal_t delta = (fXUpper - fXLower)/NBIN;

    	for(size_t i=0; i<NBIN; i++ )
    	{
    		this->fBins[i] = this->fXLower + i*delta;
    	}

    	this->fBins[NBIN] = this->fXUpper;

    }


    void SetCallTable()
    {

    	fCallTable.resize(NBIN*(NRULE+1)/2);
    	table_h temp_table(NBIN*(NRULE+1)/2);

    	for(size_t bin=0; bin<NBIN; bin++)
    	{
    		for(size_t call=0; call<(NRULE+1)/2; call++)
    		{
    			GReal_t lower_lim = fBins[bin];
    			GReal_t upper_lim = fBins[bin+1];
    			GReal_t abscissa_X_P = 0;
    			GReal_t abscissa_X_M = 0;
    			GReal_t abscissa_Weight = 0;

    			thrust::tie(abscissa_X_P, abscissa_X_M, abscissa_Weight) = fRule.GetAbscissa(call , lower_lim, upper_lim);

    			GReal_t rule_GaussKronrod_Weight   = fRule.KronrodWeight[call];
    			GReal_t rule_Gauss_Weight          = fRule.GaussWeight[call];

    			size_t index = call*NBIN + bin;

    			temp_table[index]= row_t( abscissa_X_P, abscissa_X_M, abscissa_Weight, rule_GaussKronrod_Weight, rule_Gauss_Weight);
    		}
    	}

    	///for(auto row: temp_table) std::cout << row << std::endl;
    	thrust::copy( temp_table.begin(), temp_table.end(), fCallTable.begin() );

    }

	GReal_t fXLower;
	GReal_t fXUpper;
	GReal_t fBins[NBIN+1];
	GaussKronrodRule<NRULE> fRule;
	table_d fCallTable;


};


} // namespace hydra

#include <hydra/detail/GaussKronrodQuadrature.inl>

#endif /* GAUSSKRONRODQUADRATURE_H_ */
