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
 * PlainState.h
 *
 *  Created on: 30/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PLAINSTATE_H_
#define PLAINSTATE_H_

#include <limits>
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

namespace hydra {


struct PlainState
{


    size_t  fN;
    GReal_t fMin;
    GReal_t fMax;
    GReal_t fMean;
    GReal_t fM2;

    __host__ __device__
PlainState():
	fN(0),
	fMean(0),
	fM2(0),
	fMin(std::numeric_limits<GReal_t>::min() ),
    fMax(std::numeric_limits<GReal_t>::max() )
    {}

    __host__ __device__
   PlainState( PlainState const& other):
   	fN(other.fN),
   	fMean(other.fMean ),
   	fM2(other.fM2 ),
   	fMin(other.fMin  ),
    fMax(other.fMax  )
       {}



    __host__ __device__ inline
    GReal_t variance()   { return fM2 / (fN - 1); }

    __host__ __device__ inline
    GReal_t variance_n() { return fM2 / fN; }

};

}


#endif /* PLAINSTATE_H_ */
