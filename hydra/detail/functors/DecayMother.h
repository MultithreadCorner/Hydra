

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
 * DecayMother.h
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

#ifndef DECAYMOTHER_H_
#define DECAYMOTHER_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>


namespace hydra
{

namespace detail
{


template <size_t N, unsigned int BACKEND, typename GRND>
struct DecayMother
{
	typedef detail::BackendTraits<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t>  vector_real;

	const GInt_t fSeed;
	const GInt_t fNDaughters;
	GReal_t fTeCmTm;
	GReal_t fWtMax;
	GReal_t fBeta0;
	GReal_t fBeta1;
	GReal_t fBeta2;


	//const GReal_t* __restrict__ fMasses;
	GReal_t fMasses[N];

	//constructor
	DecayMother(Vector4R const& mother,
			vector_real const& _masses,
			const GInt_t _ndaughters, const GInt_t _seed):
				//fMasses(thrust::raw_pointer_cast(_masses.data())),
				fNDaughters(_ndaughters),
				fSeed(_seed)

	{

		for(size_t i=0; i<N; i++) fMasses[i]=_masses[i];

		GReal_t _fTeCmTm = mother.mass(); // total energy in C.M. minus the sum of the masses

		for (size_t n = 0; n < fNDaughters; n++)
		{
			_fTeCmTm -= _masses.data()[n];
		}

		GReal_t emmax = _fTeCmTm + _masses.data()[0];
		GReal_t emmin = 0.0;
		GReal_t wtmax = 1.0;
		for (size_t n = 1; n < fNDaughters; n++)
		{
			emmin += _masses.data()[n - 1];
			emmax += _masses.data()[n];
			wtmax *= pdk(emmax, emmin, _masses.data()[n]);
		}
		GReal_t _fWtMax = 1.0 / wtmax;

		GReal_t _beta = mother.d3mag() / mother.get(0);

		if (_beta)
		{
			GReal_t w = _beta / mother.d3mag();
			fBeta0 = mother.get(0) * w;
			fBeta1 = mother.get(1) * w;
			fBeta2 = mother.get(2) * w;
		}
		else
			fBeta0 = fBeta1 = fBeta2 = 0.0;

		fTeCmTm = _fTeCmTm;
		fWtMax = _fWtMax;


	}

	__host__ __device__
	DecayMother( DecayMother<N, BACKEND, GRND> const& other ):
	fSeed(other.fSeed ),
	fNDaughters(other.fNDaughters ),
	fTeCmTm(other.fTeCmTm ),
	fWtMax(other.fWtMax ),
	fBeta0(other.fBeta0 ),
	fBeta1(other.fBeta1 ),
	fBeta2(other.fBeta2 )
	{ for(size_t i=0; i<N; i++) fMasses[i]=other.fMasses[i]; }



	__host__      __device__ inline
	GReal_t pdk(const GReal_t a, const GReal_t b,
			const GReal_t c) const
	{
		//the PDK function
		GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
		x = sqrt(x) / (2 * a);
		return x;
	}

	__host__ __device__ inline
	void bbsort( GReal_t *array, GInt_t n)
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

	__host__   __device__ inline
	GReal_t process(const GInt_t evt, Vector4R** daugters)
	{

		GRND randEng( hash(evt,fSeed) );
		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		GReal_t rno[N];
		rno[0] = 0.0;
		rno[fNDaughters - 1] = 1.0;

		if (fNDaughters > 2)
		{
#pragma unroll N
			for (GInt_t n = 1; n < fNDaughters - 1; n++)
			{
				rno[n] =  uniDist(randEng) ;

			}

			bbsort(&rno[1], fNDaughters -2);

		}


		GReal_t invMas[kMAXP], sum = 0.0;

#pragma unroll N
		for (size_t n = 0; n < fNDaughters; n++)
		{
			//printf("%d mass=%f \n",n, fMasses[n]);
			sum += fMasses[n];
			invMas[n] = rno[n] * fTeCmTm + sum;
		}

		//
		//-----> compute the weight of the current event
		//

		GReal_t wt = fWtMax;

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

		daugters[0]->set(sqrt((GReal_t) pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
				pd[0], 0.0);

#pragma unroll N
		for (size_t i = 1; i < fNDaughters; i++)
		{

			daugters[i]->set(
					sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
					-pd[i - 1], 0.0);

			GReal_t cZ = 2 * uniDist(randEng) -1 ;
			GReal_t sZ = sqrt(1 - cZ * cZ);
			GReal_t angY = 2 * PI* uniDist(randEng);
			GReal_t cY = cos(angY);
			GReal_t sY = sin(angY);
			for (size_t j = 0; j <= i; j++)
			{

				GReal_t x = daugters[j]->get(1);
				GReal_t y = daugters[j]->get(2);
				daugters[j]->set(1, cZ * x - sZ * y);
				daugters[j]->set(2, sZ * x + cZ * y); // rotation around Z

				x = daugters[j]->get(1);
				GReal_t z = daugters[j]->get(3);
				daugters[j]->set(1, cY * x - sY * z);
				daugters[j]->set(3, sY * x + cY * z); // rotation around Y
			}

			if (i == (fNDaughters - 1))
				break;

			GReal_t beta = pd[i] / sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
			for (size_t j = 0; j <= i; j++)
			{

				daugters[j]->applyBoostTo(Vector3R(0, beta, 0));
			}

		}

		//
		//---> final boost of all particles to the mother's frame
		//
#pragma unroll N
		for (size_t n = 0; n < fNDaughters; n++)
		{

			daugters[n]->applyBoostTo(Vector3R(fBeta0, fBeta1, fBeta2));

		}

		//
		//---> return the weight of event
		//

		return wt;

	}

	template<typename Tuple>
	__host__      __device__ inline GReal_t operator()(const GInt_t evt, Tuple &particles)
	{

		constexpr size_t SIZE = thrust::tuple_size<Tuple>::value;

		Vector4R* Particles[SIZE];
		detail::set_ptrs_to_tuple(particles, &Particles[0]  );

		return process(evt, Particles);

	}


};

}//namespace detail
}//namespace hydra

#endif /* DECAYMOTHER_H_ */
