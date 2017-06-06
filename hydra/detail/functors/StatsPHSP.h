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
 * StatsPHSP.h
 *
 *  Created on: 24/05/2017
 *      Author: Antonio Augusto Alves Junior
 */


/**
 * \file
 * \ingroup Phase-space generator
 */

#ifndef STATSPHSP_H_
#define STATSPHSP_H_

namespace hydra {

namespace detail {

struct StatsPHSP
{
	GReal_t fMean;
    GReal_t fM2;
    GReal_t fW;

};


struct AddStatsPHSP
		:public thrust::binary_function< StatsPHSP const&, StatsPHSP const&, StatsPHSP >
{


    __host__ __device__ inline
    StatsPHSP operator()( StatsPHSP const& x, StatsPHSP const& y)
    {
    	StatsPHSP result;

        GReal_t w  = x.fW + y.fW;

        GReal_t delta  = y.fMean*y.fW - x.fMean*x.fW;
        GReal_t delta2 = delta  * delta;

        result.fW   = w;

        result.fMean = (x.fMean*x.fW + y.fMean*y.fW)/w;
        result.fM2   = x.fM2   +  y.fM2;
        result.fM2  += delta2 * x.fW * y.fW / w;

        return result;
    }

};


}//namespace detail


}//namespace hydra

#endif /* STATSPHSP_H_ */
