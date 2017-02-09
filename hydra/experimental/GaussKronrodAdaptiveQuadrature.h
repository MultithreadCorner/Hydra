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
 * GaussKronrodAdaptiveQuadrature.h
 *
 *  Created on: 04/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODADAPTIVEQUADRATURE_H_
#define GAUSSKRONRODADAPTIVEQUADRATURE_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/GaussKronrodRules.h>
#include <hydra/experimental/detail/functors/ProcessGaussKronrodAdaptiveQuadrature.h>
#include <hydra/experimental/multivector.h>
#include <hydra/detail/Integrator.h>

#include <hydra/detail/Print.h>
#include <tuple>

namespace hydra {

namespace experimental {

template<size_t NRULE, size_t NBIN=100>
class GaussKronrodAdaptiveQuadrature: public Integrator< GaussKronrodAdaptiveQuadrature<NRULE, NBIN > >
{
public:
	//tag
	typedef void hydra_integrator_tag;

	/*
	 * nodes
	 */
	typedef thrust::tuple<
			GBool_t, // process
			GUInt_t,  // bin
			double,  // lower
			double,  // upper
			double,  // integral
			double   // error
			> node_t;

	typedef thrust::host_vector<node_t>   node_list_h;
	typedef hydra::experimental::multivector<node_list_h> node_table_h;

	/*
	 * parameters
	 */
	typedef thrust::tuple<
			GUInt_t,// bin ,
			double, // abscissa_X_P
			double, // abscissa_X_M
			double, // Jacobian
			double, // rule_GaussKronrod_Weight
			double  // rule_Gauss_Weight
			> parameters_t;

	typedef thrust::host_vector<parameters_t>   parameters_list_h;
	typedef thrust::device_vector<parameters_t> parameters_list_d;

	typedef hydra::experimental::multivector<parameters_list_h> parameters_table_h;
	typedef hydra::experimental::multivector<parameters_list_d> parameters_table_d;

	/*
	 * call results
	 */
	typedef thrust::tuple<
			GUInt_t, // bin
			double,  // gauss
			double   // kronrod
			> call_t;

	typedef thrust::host_vector<call_t>   call_list_h;
	typedef thrust::device_vector<call_t> call_list_d;

	typedef hydra::experimental::multivector<call_list_h> call_table_h;
	typedef hydra::experimental::multivector<call_list_d> call_table_d;



	GaussKronrodAdaptiveQuadrature(GReal_t xlower, GReal_t xupper):
		fIterationNumber(0),
		fXLower(xlower),
		fXUpper(xupper),
		fMaxRelativeError(1.0e-9),
		fRule(GaussKronrodRuleSelector<NRULE>().fRule)
	{	InitNodes();}

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);

	std::pair<GReal_t, GReal_t> Accumulate();

	void Print()
	{
		size_t nNodes   =  fNodesTable.size();
		HYDRA_CALLER ;
		HYDRA_MSG << "GaussKronrodAdaptiveQuadrature begin: " << HYDRA_ENDL;
		HYDRA_MSG << "XLower: " << fXLower << HYDRA_ENDL;
		HYDRA_MSG << "XUpper: " << fXUpper << HYDRA_ENDL;
		HYDRA_MSG << "#Nodes: " << nNodes << HYDRA_ENDL;
		for(size_t i=0; i< nNodes; i++ ){
			auto node = this->fNodesTable[i];
			HYDRA_MSG << "Node ID #" << thrust::get<1>(node) <<" Interval ["
					  << thrust::get<2>(node)
					  <<", "
					  << thrust::get<3>(node)
					  << "] Result ["
					  << thrust::get<4>(node)
					  << ", "
					  << thrust::get<5>(node)
					  << "]"
					  << " Process "
					  << thrust::get<0>(node)
					  << HYDRA_ENDL;
		}
		fRule.Print();
		HYDRA_MSG << "GaussKronrodAdaptiveQuadrature end. " << HYDRA_ENDL;
	}


private:

	GReal_t GetError( GReal_t delta)
	{
		return  std::max(std::numeric_limits<GReal_t>::epsilon(),
				std::pow(200.0*std::fabs(delta ), 1.5));
	}


	void InitNodes()
	{
		GReal_t delta = (fXUpper - fXLower)/NBIN;
		fNodesTable.resize(NBIN);

		for(size_t i=0; i<NBIN; i++ )
		{
			auto node = this->fNodesTable[i];
			thrust::get<0>(node) = 	1;
			thrust::get<1>(node) = 	i;
			thrust::get<2>(node) = 	this->fXLower + i*delta;
			thrust::get<3>(node) = 	this->fXLower + (i+1)*delta;
			thrust::get<4>(node) = 	0.0;
			thrust::get<5>(node) = 	0.0;
		}

	}

	size_t CountNodesToProcess()
	{
		auto begin = fNodesTable.template vbegin<0>();
		auto end   = fNodesTable.template vend<0>();

	    size_t n=0;
		for(auto i = begin; i!=end; i++)
		if(*i)n++;
		return n;
	}

	void SetParametersTable( )
	{

		size_t nNodes =  CountNodesToProcess();

		fParametersTable.clear();
		fParametersTable.resize(nNodes*(NRULE+1)/2);
		parameters_table_h temp_table(nNodes*(NRULE+1)/2);

		//for(size_t i=0; i<nNodes; i++)
		size_t i=0;
		for(auto node : this->fNodesTable)
		{
		//	auto node = this->fNodesTable[i];
			std::cout << "node "<< thrust::get<1>(node) << std::endl;
			if(!thrust::get<0>(node))
			{
				std::cout << " >>>>>>>> no process"<< std::endl;
				continue;
			}

			for(size_t call=0; call<(NRULE+1)/2; call++)
			{
				GReal_t abscissa_X_P = 0;
				GReal_t abscissa_X_M = 0;
				GReal_t jacobian = 0;
				GReal_t fLowerLimits= thrust::get<2>(node);
				GReal_t fUpperLimits= thrust::get<3>(node);

				thrust::tie(abscissa_X_P, abscissa_X_M, jacobian)
				= fRule.GetAbscissa(call , fLowerLimits, fUpperLimits);

				GReal_t rule_GaussKronrod_Weight   = fRule.KronrodWeight[call];
				GReal_t rule_Gauss_Weight          = fRule.GaussWeight[call];

				size_t index = call*nNodes + i;


				temp_table[index]= parameters_t(thrust::get<1>(node), abscissa_X_P, abscissa_X_M,
						jacobian, rule_GaussKronrod_Weight, rule_Gauss_Weight);
			}

			i++;
		}

		for(auto row: temp_table) std::cout << row << std::endl;
		thrust::copy( temp_table.begin(), temp_table.end(), fParametersTable.begin() );

	}

	void UpdateNodes();

	GUInt_t fIterationNumber;
	GReal_t fXLower;
	GReal_t fXUpper;
	GReal_t fMaxRelativeError;
	node_table_h  fNodesTable;
	parameters_table_d fParametersTable;
	call_table_h fCallTableHost;
	call_table_d fCallTableDevice;

	/*
	GBool_t fSplit;
	RealVector_h fUpperLimits;
	RealVector_h fLowerLimits;
	result_table_h fResultTableHost;
	result_table_d fResultTableDevice;
	result_table_d fIterationResultTable;
	*/
	GaussKronrodRule<NRULE> fRule;

};

}

}

#include <hydra/experimental/detail/GaussKronrodAdaptiveQuadrature.inl>


#endif /* GAUSSKRONRODADAPTIVEQUADRATURE_H_ */
