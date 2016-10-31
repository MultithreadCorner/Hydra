

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
 * DecayMothers.h
 *
 * Copyright 2016 Antonio Augusto Alves Junior
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */

#ifndef DECAYMOTHERS_H_
#define DECAYMOTHERS_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/experimental/Vector3R.h>
#include <hydra/experimental/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>

//thrust
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>


namespace hydra
{

namespace detail
{

template <size_t N, unsigned int BACKEND, typename GRND>
struct DecayMothers
{
	typedef detail::BackendTraits<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t>  vector_real;

	const GInt_t fSeed;
	const GInt_t fNDaughters;
	const GReal_t* __restrict__ fMasses;

	//constructor
	DecayMothers(vector_real const& _masses,
			const GInt_t _ndaughters, const GInt_t _seed ):
			fMasses(thrust::raw_pointer_cast(_masses.data())),
			fNDaughters(_ndaughters),
			fSeed(_seed)
	{
	}

	__host__      __device__ GReal_t pdk(const GReal_t a, const GReal_t b,
			const GReal_t c) const
	{
		//the PDK function
		GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
		x = sqrt(x) / (2 * a);
		return x;
	}

	__host__ __device__ void bbsort(GReal_t *array, GInt_t n)
	{
		// Improved bubble sort
		for (GInt_t c = 0; c < n; c++)
		{
			GInt_t nswap = 0;

			for (GInt_t d = 0; d < n - c - 1; d++)
			{
				if (array[d] > array[d + 1]) /* For decreasing order use < */
				{
					GReal_t swap = array[d];
					array[d] = array[d + 1];
					array[d + 1] = swap;
					nswap++;
				}
			}
			if (nswap == 0)
				break;
		}

	}

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

	__host__      __device__ GReal_t process(const GInt_t evt, experimental::Vector4R* particles)
	{

		GRND randEng( hash(evt,fSeed) );

		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		GReal_t fTeCmTm = 0.0;//, fWtMax = 0.0;

		fTeCmTm = particles[0]->mass(); // total energy in C.M. minus the sum of the masses

		#pragma unroll N
		for (size_t n = 0; n < fNDaughters; n++)
		{
			fTeCmTm -= fMasses[n];
		}

		GReal_t emmax = fTeCmTm + fMasses[0];
		GReal_t emmin = 0.0;
		GReal_t wtmax = 1.0;

		#pragma unroll N
		for (size_t n = 1; n < fNDaughters; n++)
		{
			emmin += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= pdk(emmax, emmin, fMasses[n]);
		}
	   //	fWtMax = 1.0 / wtmax;
		//
		//---->  get the betas of the decaying particle
		//
		//GReal_t fBeta[3];
		//fBeta[0]=0, fBeta[1]=0, fBeta[2] = 0.0;

		GReal_t _beta = particles[0]->d3mag() / particles[0]->get(0);

		if (_beta)
		{
			GReal_t w = _beta / particles[0]->d3mag();
			//fBeta[0] = particles[0]->get(0) * w;
			//fBeta[1] = particles[0]->get(1) * w;
			//fBeta[2] = particles[0]->get(2) * w;
		}

		GReal_t rno[N];
		rno[0] = 0.0;

		if (fNDaughters > 2)
		{
			#pragma unroll N
			for (size_t n = 1; n < fNDaughters - 1; n++)
				rno[n] = uniDist(randEng) ;
			bbsort(&rno[1], fNDaughters - 2);

		}
		rno[fNDaughters - 1] = 1;

		GReal_t invMas[kMAXP], sum = 0.0;

		#pragma unroll N
		for (size_t n = 0; n < fNDaughters; n++)
		{
			sum += fMasses[n];
			invMas[n] = rno[n] * fTeCmTm + sum;
		}

		//
		//-----> compute the weight of the current event
		//

		GReal_t wt  = 1.0 / wtmax;

		GReal_t pd[N];

		#pragma unroll N
		for (size_t n = 0; n < fNDaughters - 1; n++)
		{
			pd[n] = pdk(invMas[n + 1], invMas[n], fMasses[n + 1]);
			wt *= pd[n];
		}

		//
		//-----> complete specification of event (Raubold-Lynch method)
		//

		particles[1]->set(sqrt(pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
				pd[0], 0.0);

		#pragma unroll N
		for (size_t i = 1; i < fNDaughters; i++)
		{

			particles[i + 1]->set(
					sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
					-pd[i - 1], 0.0);

			GReal_t cZ = 2	* uniDist(randEng) -1 ;
			GReal_t sZ = sqrt(1 - cZ * cZ);
			GReal_t angY = 2.0 * PI	* uniDist(randEng);
			GReal_t cY = cos(angY);
			GReal_t sY = sin(angY);
			for (size_t j = 0; j <= i; j++)
			{

				GReal_t x = particles[j + 1]->get(1);
				GReal_t y = particles[j + 1]->get(2);
				particles[j + 1]->set(1, cZ * x - sZ * y);
				particles[j + 1]->set(2, sZ * x + cZ * y); // rotation around Z

				x = particles[j + 1]->get(1);
				GReal_t z = particles[j + 1]->get(3);
				particles[j + 1]->set(1, cY * x - sY * z);
				particles[j + 1]->set(3, sY * x + cY * z); // rotation around Y
			}

			if (i == (fNDaughters - 1))
				break;

			GReal_t beta = pd[i] / sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
			for (size_t j = 0; j <= i; j++)
			{

				particles[j + 1]->applyBoostTo(0, beta, 0);
			}

			//i++;
		}

		//
		//---> final boost of all particles to the mother's frame
		//
		#pragma unroll N
		for (size_t n = 0; n < fNDaughters; n++)
		{

			particles[n + 1]->applyBoostTo(*particles[0]);

		}

		//
		//---> return the weight of event
		//

		return wt;

	}
	template<typename Tuple>
		__host__      __device__ GReal_t operator()(const GInt_t evt, Tuple &particles)
			{

			constexpr size_t SIZE = thrust::tuple_size<Tuple>::value;

			experimental::Vector4R Particles[SIZE];
			//detail::set_ptrs_to_tuple(Particles,  particles);
			detail::tupleToArray(particles,  &Particles[0] );
			return process(evt, Particles);

			}



};
}//namespace detail
}//namespace hydra

#endif /* DECAYMOTHERS_H_ */
