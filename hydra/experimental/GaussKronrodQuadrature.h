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
#include <hydra/Types.h>
#include <hydra/experimental/GaussKronrodRules.h>
#include <hydra/experimental/multivector.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/Print.h>
#include <tuple>

namespace hydra {

namespace experimental {

template<size_t NRULE, size_t NBIN=200>
class GaussKronrodQuadrature: public Integrator< GaussKronrodQuadrature<NRULE, NBIN > >
{
public:

	    typedef thrust::tuple<unsigned int, double, double, double, double> row_t;

		typedef thrust::host_vector<row_t>   row_list_h;
		typedef thrust::device_vector<row_t> row_list_d;

		typedef hydra::experimental::multivector<row_list_h> table_h;
		typedef hydra::experimental::multivector<row_list_d> table_d;


	GaussKronrodQuadrature(GReal_t xlower, GReal_t xupper):
		fXLower(xlower),
		fXUpper(xupper),
		fRule(GaussKronrodRuleSelector<NRULE>().fRule)
	{

		this->SetBins();
		this->SetCallTable();
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
    			GReal_t abscissa_X       = fRule.GetAbscissa(call , lower_lim, upper_lim).first;
    			GReal_t abscissa_Weight  = fRule.GetAbscissa(call , lower_lim, upper_lim).second;
    			GReal_t rule_GaussKronrod_Weight   = fRule.KronrodWeight[call];
    			GReal_t rule_Gauss_Weight          = fRule.GaussWeight[call];

    			size_t index = call*NBIN + bin;

    			temp_table[index]= row_t(call, abscissa_X, abscissa_Weight, rule_GaussKronrod_Weight, rule_Gauss_Weight);
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

}

}

#include <hydra/experimental/detail/GaussKronrodQuadrature.inl>

#endif /* GAUSSKRONRODQUADRATURE_H_ */
