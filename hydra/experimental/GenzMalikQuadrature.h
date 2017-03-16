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
 * GenzMalikQuadrature.h
 *
 *  Created on: Feb 28, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENZMALIKQUADRATURE_H_
#define GENZMALIKQUADRATURE_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/GenzMalikRule.h>
#include <hydra/experimental/GenzMalikBox.h>
#include <hydra/experimental/multivector.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/Generic.h>

#include <algorithm>
#include <cmath>

namespace hydra {

namespace experimental {

template<  size_t N, unsigned int BACKEND=hydra::host>
class  GenzMalikQuadrature: public Integrator<GenzMalikQuadrature<BACKEND, N> >
{

public:
	//tag
	typedef void hydra_integrator_tag;

	typedef hydra::mc_host_vector<GenzMalikBox<N>> box_list_type;

	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			std::array<size_t, N> const& grid)
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
            GenzMalikBox<N> box(lower_limit, upper_limit);

			fBoxList.push_back(box);

		}
	}

	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			size_t nboxes=10)
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
			GenzMalikBox<N> box(lower_limit, upper_limit);

			fBoxList.push_back(box);

		}
	}

	template<unsigned int BACKEND2>
	GenzMalikQuadrature( GenzMalikQuadrature<N,BACKEND2> const& other):
	fBoxList(other.GetBoxList() ),
	fGenzMalikRule(other.GetGenzMalikRule() )
	{}

	template<unsigned int BACKEND2>
	GenzMalikQuadrature<N,BACKEND>& operator=( GenzMalikQuadrature<N,BACKEND2> const& other)
	{
		if(this==&other) return *this;

		this->fBoxList=other.GetBoxList() ;
		this->fGenzMalikRule = other.GetGenzMalikRule() ;

		return *this;
	}


	void Print()
	{
		fGenzMalikRule.Print();
		for(auto box: fBoxList )	box.Print();

	}

	const box_list_type& GetBoxList() const {
		return fBoxList;
	}

	const GenzMalikRule<N, BACKEND>& GetGenzMalikRule() const {
		return fGenzMalikRule;
	}

	void SetBoxList(const box_list_type& boxList) {
		fBoxList = boxList;
	}

	void SetGenzMalikRule(const GenzMalikRule<N, BACKEND>& genzMalikRule) {
		fGenzMalikRule = genzMalikRule;
	}


private:

	constexpr void GetGrid( size_t nboxes , std::array<size_t, N>& grid )
	{
		size_t ndivsion = std::llround( std::pow( 2.0, std::log2(double(nboxes))/double(N) ) );

		std::fill_n( grid.begin(), N,  ndivsion );
	}


	GenzMalikRule<  N,  BACKEND> fGenzMalikRule;
	box_list_type fBoxList;

};

}  // namespace experimental

} // namespace hydra

#endif /* GENZMALIKQUADRATURE_H_ */
