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
 * GenzMalikQuadrature.inl
 *
 *  Created on: 17/03/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENZMALIKQUADRATURE_INL_
#define GENZMALIKQUADRATURE_INL_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/GenzMalikRule.h>
#include <hydra/detail/GenzMalikBox.h>
#include <hydra/multivector.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/functors/ProcessGenzMalikQuadrature.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <cmath>
#include <future>



namespace hydra {


template<size_t N,hydra::detail::Backend  BACKEND>
GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
		std::array<GReal_t,N> const& UpperLimit, std::array<size_t, N> const& grid)
		{

			size_t nboxes = 1;
			hydra::detail::multiply(grid, nboxes );
			//fBoxList.resize(nboxes);

			std::array<GReal_t, N> width;

			for( size_t i=0; i<N; i++)
			{ width[i] = (UpperLimit[i] -  LowerLimit[i])/grid[i];  }

			std::array<size_t, N> mindex;
			std::array<GReal_t,N>  lower_limit;
			std::array<GReal_t,N>  upper_limit;

			for(size_t index=0; index<nboxes; index++)
			{
				hydra::detail::get_indexes( index, grid,  mindex );

				for( size_t dim=0; dim<N; dim++)
				{
					lower_limit[dim] =   LowerLimit[dim] + width[dim]*mindex[dim];
					upper_limit[dim] =   LowerLimit[dim] + width[dim]*(mindex[dim]+1);
				}
				detail::GenzMalikBox<N> box(lower_limit, upper_limit);

				fBoxList.push_back(box);

			}
		}


template<size_t N,hydra::detail::Backend  BACKEND>
GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
		std::array<GReal_t,N> const& UpperLimit, size_t nboxes)
		{

			std::array<size_t, N> grid;

			GetGrid( nboxes, grid) ;
			hydra::detail::multiply(grid, nboxes );

			std::array<GReal_t, N> width;

			for( size_t i=0; i<N; i++)
				width[i] = (UpperLimit[i] -  LowerLimit[i])/grid[i];

			std::array< size_t,N> mindex;
			std::array<GReal_t,N> lower_limit;
			std::array<GReal_t,N> upper_limit;

			for(size_t index=0; index<nboxes; index++)
			{
				hydra::detail::get_indexes( index, grid,  mindex );

				for( size_t dim=0; dim<N; dim++)
				{
					lower_limit[dim] =   LowerLimit[dim] + width[dim]*mindex[dim];
					upper_limit[dim] =   LowerLimit[dim] + width[dim]*(mindex[dim]+1);
				}

				detail::GenzMalikBox<N> box(lower_limit, upper_limit);

				fBoxList.push_back(box);

			}
		}


template<size_t N, hydra::detail::Backend  BACKEND>
GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::GenzMalikQuadrature( GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>> const& other):
fBoxList(other.GetBoxList() ),
fGenzMalikRule(other.GetGenzMalikRule() )
{}

template<size_t N, hydra::detail::Backend  BACKEND>
template<hydra::detail::Backend  BACKEND2>
GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::GenzMalikQuadrature( GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND2>> const& other):
fBoxList(other.GetBoxList() ),
fGenzMalikRule(other.GetGenzMalikRule() )
{}


template<size_t N, hydra::detail::Backend  BACKEND>
GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>&
GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>::operator=( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>> const& other)
{
	if(this==&other) return *this;

	this->fBoxList=other.GetBoxList() ;
	this->fGenzMalikRule = other.GetGenzMalikRule() ;

	return *this;
}

template<size_t N, hydra::detail::Backend  BACKEND>
template<hydra::detail::Backend  BACKEND2>
GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>&
GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>::operator=( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND2>> const& other)
{
	if(this==&other) return *this;

	this->fBoxList=other.GetBoxList() ;
	this->fGenzMalikRule = other.GetGenzMalikRule() ;

	return *this;
}

template<size_t N, hydra::detail::Backend  BACKEND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t> GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>::Integrate(FUNCTOR const& functor)
{

	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + fBoxList.size();


	HYDRA_EXTERNAL_NS::thrust::for_each(HYDRA_EXTERNAL_NS::thrust::host,first, last,
				detail::ProcessGenzMalikBox<N, FUNCTOR,const_rule_iterator, box_iterator>(functor,
						fGenzMalikRule.GetAbscissas().begin(),fGenzMalikRule.GetAbscissas().end(),
						fBoxList.begin(), fBoxList.end()));


	GReal_t integral=0;

	GReal_t    error=0;

		for(auto box:fBoxList)
		{
			integral+= box.GetIntegral();
			error   +=  box.GetErrorSq();

		}

		error=std::sqrt(error);


	return std::pair<GReal_t, GReal_t>(integral, error);
}



} // namespace hydra

#endif /* GENZMALIKQUADRATURE_INL_ */
