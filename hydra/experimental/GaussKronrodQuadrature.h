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

#include <hydra/experimental/GaussKronrodRule.h>
#include <hydra/detail/Integrator.h>

namespace hydra {

namespace experimental {

}
template<size_t NRULE, size_t NBIN=200>
class GaussKronrodQuadrature: public Integrator< GaussKronrodQuadrature<NRULE, NBIN > >
{
public:

	GaussKronrodQuadrature(GReal_t xlower, GReal_t xupper):
	{

	}

	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);

private:



	GReal_t fXlower;
	GReal_t fXupper;
	GReal_t fBins[NBIN+1];
};

}

}



#endif /* GAUSSKRONRODQUADRATURE_H_ */
