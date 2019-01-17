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
#include <hydra/Integrator.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/functors/ProcessGenzMalikQuadrature.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <cmath>
#include <future>



namespace hydra {

template<size_t N,hydra::detail::Backend  BACKEND>
void GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::SetGeometry(
		std::array<GReal_t,N> const& LowerLimit,
		std::array<GReal_t,N> const& UpperLimit,
		std::array<size_t, N> const& grid){

	size_t nboxes = 1;
	hydra::detail::multiply(grid, nboxes );

	fBoxList.reserve(nboxes);

	std::array<GReal_t, N> width;

	for( size_t i=0; i<N; i++)
	{ width[i] = (UpperLimit[i] -  LowerLimit[i])/grid[i];  }

	std::array<size_t, N>  mindex;
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
		//detail::GenzMalikBox<N> box(lower_limit, upper_limit);

		fBoxList.emplace_back(lower_limit, upper_limit );

	}


}


template<size_t N,hydra::detail::Backend  BACKEND>
void GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::SetGeometry(
		std::array<GReal_t,N> const& LowerLimit,
		std::array<GReal_t,N> const& UpperLimit, size_t nboxes){

			std::array<size_t, N> grid;

			GetGrid( nboxes, grid) ;
			hydra::detail::multiply(grid, nboxes );

			fBoxList.reserve(nboxes);

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

				//detail::GenzMalikBox<N> box(lower_limit, upper_limit);

				fBoxList.emplace_back(lower_limit, upper_limit );

			}
		}



template<size_t N,hydra::detail::Backend  BACKEND>
void GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::SetGeometry(
		const GReal_t (&LowerLimit)[N],
		const GReal_t (&UpperLimit)[N], const size_t (&grid)[N]){

			size_t nboxes = 1;
			hydra::detail::multiply(grid, nboxes );

			fBoxList.reserve(nboxes);

			std::array<GReal_t, N> width;

			for( size_t i=0; i<N; i++)
			{ width[i] = (UpperLimit[i] -  LowerLimit[i])/grid[i];  }

			size_t mindex[N];
			GReal_t  lower_limit[N];
			GReal_t  upper_limit[N];

			for(size_t index=0; index<nboxes; index++)
			{
				hydra::detail::get_indexes( index, grid,  mindex );

				for( size_t dim=0; dim<N; dim++)
				{
					lower_limit[dim] =   LowerLimit[dim] + width[dim]*mindex[dim];
					upper_limit[dim] =   LowerLimit[dim] + width[dim]*(mindex[dim]+1);
				}
				//detail::GenzMalikBox<N> box(lower_limit, upper_limit);

				fBoxList.emplace_back(lower_limit, upper_limit );

			}
		}


template<size_t N,hydra::detail::Backend  BACKEND>
void GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>::SetGeometry(const GReal_t (&LowerLimit)[N],
		const GReal_t (&UpperLimit)[N],	size_t nboxes)
		{

			std::array<size_t, N> grid;

			GetGrid( nboxes, grid) ;
			hydra::detail::multiply(grid, nboxes );

			fBoxList.reserve(nboxes);

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

				//detail::GenzMalikBox<N> box(lower_limit, upper_limit);

				fBoxList.emplace_back(lower_limit, upper_limit );

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

	device_box_list_type TempBoxList_d( fBoxList );

	detail::ProcessGenzMalikBox<N, FUNCTOR, rule_iterator> process_box(functor,
			fGenzMalikRule.begin(), fGenzMalikRule.end() ) ;

	HYDRA_EXTERNAL_NS::thrust::for_each(hydra::detail::BackendPolicy<BACKEND>{}, TempBoxList_d.begin(),
			TempBoxList_d.end(), process_box);


	auto result = CalculateIntegral(TempBoxList_d);

	if( result.second/result.first < fRelativeError)

		return result;

	else{

		do{
			//auto start = std::chrono::high_resolution_clock::now();

			AdaptiveIntegration(functor, TempBoxList_d);

			//auto stop = std::chrono::high_resolution_clock::now();
			//std::chrono::duration<double, std::milli> elapsed = stop - start;
			//std::cout << "AdaptiveIntegration Time (ms): " << elapsed.count() <<std::endl;

			result = CalculateIntegral(TempBoxList_d);

		} while( (result.second /result.first != 0) && result.second /result.first > fRelativeError );


	}

	return  result;
}

template<size_t N, hydra::detail::Backend  BACKEND>
template<typename FUNCTOR, typename Vector>
void GenzMalikQuadrature<N,
       hydra::detail::BackendPolicy<BACKEND>>::AdaptiveIntegration(FUNCTOR const& functor, Vector& BoxList) {


	detail::ProcessGenzMalikBox<N, FUNCTOR, rule_iterator> process_box(functor,
			fGenzMalikRule.begin(), fGenzMalikRule.end() ) ;

	//sort by error in increasing order
	//auto start = std::chrono::high_resolution_clock::now();

	HYDRA_EXTERNAL_NS::thrust::sort(BoxList.begin(), BoxList.end(), detail::CompareGenzMalikBoxes<N>());

	//auto stop = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> elapsed = stop - start;
	//std::cout << "Sort Time (ms): " << elapsed.count() <<std::endl;

	size_t n = BoxList.size()*fFraction;
	SplitBoxes(BoxList, n );

	//launch calculation
	HYDRA_EXTERNAL_NS::thrust::for_each(BoxList.end()-2*n, BoxList.end(), process_box);

}

template<size_t N, hydra::detail::Backend  BACKEND>
template<typename Vector>
void hydra::GenzMalikQuadrature<N,
         hydra::detail::BackendPolicy<BACKEND> >::SplitBoxes( Vector& boxes, size_t n ){

	Vector new_boxes;
	new_boxes.reserve(2*n);

	do{
		detail::GenzMalikBox<N> highest_error_box(boxes.back());
		boxes.pop_back();

		auto sub_boxes=highest_error_box.Divide();

        new_boxes.push_back(sub_boxes.first);
        new_boxes.push_back(sub_boxes.second);

	}while( --n );

	for(auto el:new_boxes)
		boxes.push_back(el);

}




template<size_t N, hydra::detail::Backend  BACKEND>
template<typename Vector>
std::pair<GReal_t, GReal_t>
hydra::GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND> >::CalculateIntegral( Vector const& BoxList){

	auto result = HYDRA_EXTERNAL_NS::thrust::reduce(BoxList.begin(), BoxList.end(),
			hydra::pair<double, double>(0,0) ,detail::AddResultGenzMalikBoxes() );

	return std::pair<GReal_t, GReal_t>(result.first, ::sqrt(result.second));
}

} // namespace hydra

#endif /* GENZMALIKQUADRATURE_INL_ */
