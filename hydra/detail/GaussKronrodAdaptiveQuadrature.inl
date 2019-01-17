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
 * GaussKronrodAdaptiveQuadrature.inl
 *
 *  Created on: 01/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODADAPTIVEQUADRATURE_INL_
#define GAUSSKRONRODADAPTIVEQUADRATURE_INL_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <cmath>
#include <tuple>
#include <limits>
#include <hydra/detail/external/thrust/sort.h>
#include <hydra/detail/external/thrust/execution_policy.h>
#include <hydra/detail/external/thrust/functional.h>

namespace hydra {

template<size_t NRULE, size_t NBIN, hydra::detail::Backend BACKEND>
std::pair<GReal_t, GReal_t>
GaussKronrodAdaptiveQuadrature<NRULE,NBIN,hydra::detail::BackendPolicy<BACKEND>>::Accumulate()
{

	for(size_t index=0; index<fCallTableHost.size() ; index++ )
		{

			auto row = fCallTableHost[index];

			size_t  bin        = HYDRA_EXTERNAL_NS::thrust::get<0>(row);
			GReal_t bin_delta  = HYDRA_EXTERNAL_NS::thrust::get<1>(row)-HYDRA_EXTERNAL_NS::thrust::get<2>(row);
			GReal_t bin_result = HYDRA_EXTERNAL_NS::thrust::get<2>(row);
			HYDRA_EXTERNAL_NS::thrust::get<4>(fNodesTable[bin])   +=  bin_result;
			HYDRA_EXTERNAL_NS::thrust::get<5>(fNodesTable[bin])   +=  bin_delta;
		}

	GReal_t result=0;
	GReal_t error2=0;

	//for(auto row: fNodesTable ) std::cout <<std::setprecision(10)<< row << std::endl;

	for(size_t node=0; node<fNodesTable.size(); node++ )
	{
		//std::cout <<std::setprecision(10)<< fNodesTable[node]<< std::endl;
		if(HYDRA_EXTERNAL_NS::thrust::get<0>(fNodesTable[node])==1 )
		{
			HYDRA_EXTERNAL_NS::thrust::get<5>(fNodesTable[node])= GetError(HYDRA_EXTERNAL_NS::thrust::get<5>(fNodesTable[node]) );
			HYDRA_EXTERNAL_NS::thrust::get<0>(fNodesTable[node])=0;
			//std::cout << "====>"<< std::setprecision(10)<< fNodesTable[node]<< std::endl;
		}

		result     += HYDRA_EXTERNAL_NS::thrust::get<4>(fNodesTable[node]);
		error2    += HYDRA_EXTERNAL_NS::thrust::get<5>(fNodesTable[node])*HYDRA_EXTERNAL_NS::thrust::get<5>(fNodesTable[node]);

	}

	return std::pair<GReal_t, GReal_t>(result, sqrt(error2) );
}

template<size_t NRULE, size_t NBIN, hydra::detail::Backend BACKEND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
GaussKronrodAdaptiveQuadrature<NRULE,NBIN, hydra::detail::BackendPolicy<BACKEND>>::Integrate(FUNCTOR const& functor)
{
	std::pair<GReal_t, GReal_t> result(0,0);

	fIterationNumber=0;
	GBool_t  condition1=0;
	GBool_t  condition2=0;

	InitNodes();
	do{

		// do  not split nodes at first iteration
		if( fIterationNumber>0 ) UpdateNodes();

		//set parameters table
		SetParametersTable( );

		//set the call table to hold the evaluation results
		fCallTableHost.resize( fParametersTable.size());
		fCallTableDevice.resize( fParametersTable.size());

		//call function in parallel
		HYDRA_EXTERNAL_NS::thrust::transform(system_t(),fParametersTable.begin(), fParametersTable.end(),
				fCallTableDevice.begin(),
				ProcessGaussKronrodAdaptiveQuadrature<FUNCTOR>(functor) );

		//copy to evaluation result back to the host
		HYDRA_EXTERNAL_NS::thrust::copy(fCallTableDevice.begin(),  fCallTableDevice.end(),
				fCallTableHost.begin());

		/**
		 * \todo Re-implement Accumulate() using HYDRA_EXTERNAL_NS::thrust::sort + thust::reduce_by_key, maybe not faster but
		 * more scalable.
		 */
		result = Accumulate();

		fIterationNumber++;

		/*
		 * keep iterating while the error is larger than the required or
		 * larger than the numerical double precision
		 */

		condition1 =  result.second > sqrt(result.first*result.first)*fMaxRelativeError;
		condition2 =  result.second > std::numeric_limits<GReal_t>::epsilon();

	}
	while( condition1 &&  condition2 );

	return result;
}


template<size_t NRULE, size_t NBIN, hydra::detail::Backend BACKEND>
void GaussKronrodAdaptiveQuadrature<NRULE,NBIN,hydra::detail::BackendPolicy<BACKEND>>::UpdateNodes()
{

	HYDRA_EXTERNAL_NS::thrust::sort(HYDRA_EXTERNAL_NS::thrust::host,
				fNodesTable.begin(),
			    fNodesTable.end(),
				hydra::detail::CompareTuples<5,	HYDRA_EXTERNAL_NS::thrust::greater >());

	auto node = fNodesTable[0];

	GReal_t lower_limits = HYDRA_EXTERNAL_NS::thrust::get<2>(node);
	GReal_t upper_limits = HYDRA_EXTERNAL_NS::thrust::get<3>(node);
	GReal_t delta = upper_limits-lower_limits;

	GReal_t delta2 = delta/2.0;
	node_t new_node1(1, 0, lower_limits ,lower_limits+delta2, 0, 0);
	node_t new_node2(1, 0, lower_limits+delta2, upper_limits, 0, 0);

	fNodesTable[0] =  new_node1;
	fNodesTable.push_back( new_node2);


	HYDRA_EXTERNAL_NS::thrust::sort(HYDRA_EXTERNAL_NS::thrust::host,
			fNodesTable.begin(),
		    fNodesTable.end(),
			hydra::detail::CompareTuples<2,	HYDRA_EXTERNAL_NS::thrust::less >());


	for(size_t  i = 0; i<fNodesTable.size(); i++)
	{
		HYDRA_EXTERNAL_NS::thrust::get<1>(fNodesTable[i])=i;
	}

	//for(auto row: fNodesTable) std::cout << row << std::endl;

}


}  // namespace hydra

#endif /* GAUSSKRONRODADAPTIVEQUADRATURE_INL_ */
