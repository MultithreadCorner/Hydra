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

#include <cmath>

namespace hydra {

namespace experimental {

template< unsigned int BACKEND, size_t N>
class  GenzMalikQuadrature: public Integrator<GenzMalikQuadrature<BACKEND, N> >
{

public:
	//tag
	typedef void hydra_integrator_tag;

	typedef hydra::mc_host_vector<GenzMalikBox> box_list_type;

	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit, std::array<GReal_t,N> const& UpperLimit[N], GUInt_t nboxes)
	{

	}


private:

	GUInt_t DivisionsPerDimension(GUInt_t nboxes) const
	{
		return std::nearbyint(std::pow(2.0, std::log2(nboxes+1)/ N));
	}

	GenzMalikRule<  NDIM,  BACKEND> fGenzMalikRule;
	box_list_type fBoxList;

};

}  // namespace experimental

} // namespace hydra

#endif /* GENZMALIKQUADRATURE_H_ */
