

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
 * FlagAcceptReject.h
 *
 * Copyright 2016 Antonio Augusto Alves Junior
 *  
 * Created on : 29/03/2016
 *      Author: Antonio Augusto Alves Junior
 */


/**
 * \file
 * \ingroup phsp
 */

#ifndef FLAGACCEPTEDREJECTED_H_
#define FLAGACCEPTEDREJECTED_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>

#include <thrust/random.h>

namespace hydra
{
namespace detail
{


/**\struct FlagAcceptReject
 * Flags generated events as accepted (1) or rejected (0).
 */
struct FlagAcceptReject
{
	size_t fSeed;
	GReal_t fWmax; ///< maximum weight

	/**
	 * FlagAcceptReject constructor. It is initialized with the value of the maximum weight
	 * with which event weights will be compared.
	 */
	FlagAcceptReject(const size_t seed, const GReal_t wmax) :
		fWmax(wmax),
		fSeed(seed)
	{	}

	/**
	 * hash function. Generate hashs to be used in random number generation initialization
	 */
	__host__   __device__ inline
	size_t hash(size_t a, size_t b)
	{
		//Matthew Szudzik pairing
		//http://szudzik.com/ElegantPairing.pdf

		size_t  A = 2 * a ;
		size_t  B = 2 * b ;
		size_t  C = ((A >= B ? A * A + A + B : A + B * B) / 2);
	    return  C ;
	}
	/**
	 * operator(). Takes the events index and weight and so flag it as accepted and rejected
	 *
	 */
	__host__ __device__ GBool_t operator ()(size_t idx, GReal_t weight)
	{

		thrust::default_random_engine randEng(hash(fSeed, idx));
		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, fWmax);


		GBool_t flag = (uniDist(randEng) < weight) ? 1 : 0;
		return flag;

	}

};
}//namespace detail
}//namespace hydra



#endif /* FLAGACCEPTEDREJECTED_H_ */
