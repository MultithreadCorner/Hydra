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
#include <hydra/experimental/detail/GenzMalikBox.h>
#include <hydra/experimental/multivector.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/Generic.h>

#include <algorithm>
#include <cmath>

namespace hydra {

namespace experimental {

template<  size_t N, unsigned int BACKEND>
class  GenzMalikQuadrature: public Integrator<typename std::enable_if< (N>1),GenzMalikQuadrature<N, BACKEND>>::type  >
{

public:
	//tag
	typedef void hydra_integrator_tag;

	typedef hydra::mc_host_vector<detail::GenzMalikBox<N>> box_list_type;
	typedef typename GenzMalikRule<  N,  BACKEND>::abscissa_iterator rule_iterator;
	typedef typename GenzMalikRule<  N,  BACKEND>::const_abscissa_iterator const_rule_iterator;

	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			std::array<size_t, N> const& grid);


	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			size_t nboxes=10);


	template<unsigned int BACKEND2>
	GenzMalikQuadrature( GenzMalikQuadrature<N,BACKEND2> const& other);

	template<unsigned int BACKEND2>
	GenzMalikQuadrature<N,BACKEND>& operator=( GenzMalikQuadrature<N,BACKEND2> const& other);

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);


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

	void GetGrid( size_t nboxes , std::array<size_t, N>& grid )
	{
		size_t ndivsion = std::llround( std::pow( 2.0, std::log2(double(nboxes))/double(N) ) );

		std::fill_n( grid.begin(), N,  ndivsion );

	}


	GenzMalikRule<  N,  BACKEND> fGenzMalikRule;
	box_list_type fBoxList;

};



}  // namespace experimental

} // namespace hydra

#include <hydra/experimental/detail/GenzMalikQuadrature.inl>

#endif /* GENZMALIKQUADRATURE_H_ */
