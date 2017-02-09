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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace hydra {

namespace experimental {

template<size_t NRULE, size_t NBIN>
std::pair<GReal_t, GReal_t> GaussKronrodAdaptiveQuadrature<NRULE,NBIN>::Accumulate()
{

	size_t nNodes   = CountNodesToProcess();

	for(size_t node=0; node<nNodes; node++ )
	{
		GReal_t bin_delta  =0;
		GReal_t bin_result =0;

		for(size_t call=0; call<(NRULE+1)/2; call++)
		{
			size_t index = call*nNodes + node;
			auto row = fCallTableHost[index];

			//if(node==thrust::get<0>(row) )
			//{
				//bin_delta  += thrust::get<1>(row)-thrust::get<2>(row);
				//bin_result += thrust::get<2>(row) ;

			//}
			thrust::get<5>(fNodesTable[ thrust::get<0>(row)])
					+= thrust::get<1>(row)-thrust::get<2>(row);

			thrust::get<4>(fNodesTable[ thrust::get<0>(row)])
					+=thrust::get<2>(row) ;

		}


		thrust::get<5>(fNodesTable[node])= GetError(thrust::get<5>(fNodesTable[node]) );


	}

	GReal_t result=0;
	GReal_t error2=0;

	for(size_t node=0; node<fNodesTable.size(); node++ )
	{
		result    += thrust::get<4>(fNodesTable[node]);
		error2    += thrust::get<5>(fNodesTable[node])*thrust::get<5>(fNodesTable[node]);
	}

	return std::pair<GReal_t, GReal_t>(result, sqrt(error2) );
}

template<size_t NRULE, size_t NBIN>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
GaussKronrodAdaptiveQuadrature<NRULE,NBIN>::Integrate(FUNCTOR const& functor)
{
	std::pair<GReal_t, GReal_t> result(0,0);

	fIterationNumber=0;

	InitNodes();

	do{

		if( fIterationNumber>0 )UpdateNodes();
		SetParametersTable( );

		fCallTableHost.resize( fParametersTable.size());
		fCallTableDevice.resize( fParametersTable.size());

		thrust::transform(fParametersTable.begin(), fParametersTable.end(),
				fCallTableDevice.begin(),
				ProcessGaussKronrodAdaptiveQuadrature<FUNCTOR>(functor) );

	//	for(auto row: fCallTableDevice) std::cout << row << std::endl;

		thrust::copy(fCallTableDevice.begin(),  fCallTableDevice.end(),
				fCallTableHost.begin());

		result = Accumulate();

		std::cout<< fIterationNumber << "  " << result.first << "  "<< result.second << std::endl;
		fIterationNumber++;

	} while( result.second > fMaxRelativeError &&
			result.second > std::numeric_limits<GReal_t>::epsilon() );

	return result;
}


template<size_t NRULE, size_t NBIN>
void GaussKronrodAdaptiveQuadrature<NRULE,NBIN>::UpdateNodes()
{



	node_table_h temp;
	for(auto node: fNodesTable)
	{
		GReal_t lower_limits = thrust::get<2>(node);
		GReal_t upper_limits = thrust::get<3>(node);
		GReal_t delta = upper_limits-lower_limits;
		GReal_t integral = thrust::get<4>(node);
		GReal_t error    = thrust::get<5>(node);



		if(   error > fMaxRelativeError   )
		{

			GReal_t delta2 = delta/2.0;

			node_t new_node1(1, 0,lower_limits ,lower_limits+delta2, 0, 0);
			node_t new_node2(1, 0,lower_limits+delta2, upper_limits, 0, 0);
			temp.push_back( new_node1);
			temp.push_back( new_node2);

		}
		else
		{
			thrust::get<0>(node)=0;
			temp.push_back(node);
		}
	}

	fNodesTable.resize(temp.size());

	fNodesTable = temp;

	thrust::sort(thrust::host,
			fNodesTable.begin(),
		    fNodesTable.end(),
			hydra::detail::CompareTuples<2,	thrust::less >());


	for(auto i = 0; i<fNodesTable.size(); i++)
	thrust::get<1>(fNodesTable[i])=i;

	//for(auto row: fNodesTable) std::cout << row << std::endl;

}

}  // namespace experimental

}  // namespace hydra

#endif /* GAUSSKRONRODADAPTIVEQUADRATURE_INL_ */
