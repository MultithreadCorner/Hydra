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
	fIterationResultTable.clear();
	fIterationResultTable.resize( fUpperLimits.size() );

	GReal_t delta_rule = 0;
	GReal_t sum_rule   = 0;
	size_t nbins       =  fUpperLimits.size();

	for(size_t bin=0; bin<nbins; bin++ )
	{
		GReal_t bin_delta  =0;
		GReal_t bin_result =0;

		for(size_t call=0; call<(NRULE+1)/2; call++)
		{
			size_t index = call*nbins + bin;
			auto row = fResultTableHost[index];

			if(bin==thrust::get<0>(row) )
			{
				bin_delta  += thrust::get<1>(row)-thrust::get<2>(row);
				bin_result +=thrust::get<2>(row) ;
			}
		}

		fIterationResultTable[bin]=result_row_t(bin, bin_result,  GetError(bin_delta));

		sum_rule   +=bin_result;
		delta_rule +=bin_delta;
	}

	GReal_t error = GetError(delta_rule);


	thrust::sort(thrust::host, fIterationResultTable.begin(),
			fIterationResultTable.end(),
			hydra::detail::CompareTuples<2,	thrust::greater >());

	for(auto row:fIterationResultTable ) std::cout << std::setprecision(50)<< row << std::endl;

	fSplit= ((error/sum_rule) > 1.0e-6);

	return std::pair<GReal_t, GReal_t>(sum_rule, error);
}

template<size_t NRULE, size_t NBIN>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
GaussKronrodAdaptiveQuadrature<NRULE,NBIN>::Integrate(FUNCTOR const& functor)
{
	SetCallTable();

	fCallTableHost.resize( fParametersTable.size());
	fCallTableDevice.resize( fParametersTable.size());

	thrust::transform(fParametersTable.begin(), fParametersTable.end(),
			fCallTableDevice.begin(),
			ProcessGaussKronrodAdaptiveQuadrature<FUNCTOR>(functor) );

	//for(auto row: fResultTableDevice) std::cout << row << std::endl;

	thrust::copy(fResultTableDevice.begin(),  fResultTableDevice.end(),
			fResultTableHost.begin());

	return Accumulate();
}




}  // namespace experimental

}  // namespace hydra

#endif /* GAUSSKRONRODADAPTIVEQUADRATURE_INL_ */
