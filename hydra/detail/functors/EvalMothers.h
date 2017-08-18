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
 * EvalMothers.h
 *
 *  Created on: 24/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */

#ifndef EVALMOTHERS_H_
#define EVALMOTHERS_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/detail/functors/StatsPHSP.h>

//thrust
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

#include <type_traits>
#include <utility>

namespace hydra {

namespace detail {

template <size_t N, typename GRND, typename FUNCTOR, typename ...FUNCTORS >
struct EvalMothers
{

	typedef  thrust::tuple<FUNCTOR,FUNCTORS...> functors_tuple_type;

	typedef  thrust::tuple<typename FUNCTOR::return_type,
			typename FUNCTORS::return_type...>  return_tuple_type;

    typedef typename hydra::detail::tuple_cat_type<thrust::tuple<GReal_t> , return_tuple_type>::type
    		result_tuple_type;

	GInt_t fSeed;
	GReal_t fMasses[N];
	functors_tuple_type fFunctors ;

	//constructor
	EvalMothers(const GReal_t (&masses)[N], const GInt_t _seed,
			FUNCTOR const& functor, FUNCTORS const& ...functors ):
			fSeed(_seed),
			fFunctors( thrust::make_tuple(functor,functors...))
	{
		for(size_t i=0; i<N; i++)
			fMasses[i] = masses[i];
	}

	//copy
	EvalMothers(EvalMothers<N, GRND,FUNCTOR, FUNCTORS...> const& other):
		fFunctors(other.fFunctors),
			fSeed(other.fSeed)
	{

#pragma unroll N
		for(size_t i=0; i<N; i++)
			fMasses[i] = other.fMasses[i];
	}


	__host__      __device__ inline
	static GReal_t pdk(const GReal_t a, const GReal_t b,
			const GReal_t c)
	{
		//the PDK function
		return sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);;
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
	constexpr static size_t hash(const size_t a, const size_t b)
		{
			//Matthew Szudzik pairing
			//http://szudzik.com/ElegantPairing.pdf

			//size_t  A = 2 * a ;
			//size_t  B = 2 * b ;
			//size_t  C = ((A >= B ? A * A + A + B : A + B * B) / 2);
		    return   (((2 * a) >=  (2 * b) ? (2 * a) * (2 * a) + (2 * a) + (2 * b) : (2 * a) + (2 * b) * (2 * b)) / 2);
		}

	__host__      __device__ GReal_t process(const GInt_t evt,
			Vector4R (&particles)[N+1])
	{

		GRND randEng( hash(evt,fSeed) );

		thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		GReal_t fTeCmTm = 0.0;

		fTeCmTm = particles[0].mass();

		#pragma unroll N
		for (size_t n = 0; n < N; n++)
		{
			fTeCmTm -= fMasses[n];
		}

		GReal_t emmax = fTeCmTm + fMasses[0];
		GReal_t emmin = 0.0;
		GReal_t wtmax = 1.0;

		#pragma unroll N
		for (size_t n = 1; n < N; n++)
		{
			emmin += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= pdk(emmax, emmin, fMasses[n]);
		}

		GReal_t rno[N];
		rno[0] = 0.0;



		if (N > 2)
		{
			#pragma unroll N
			for (size_t n = 1; n < N - 1; n++)
				rno[n] = uniDist(randEng) ;
			    bbsort(&rno[1], N - 2);

		}

		rno[N - 1] = 1;
		GReal_t invMas[N], sum = 0.0;

		#pragma unroll N
		for (size_t n = 0; n < N; n++)
		{
			sum += fMasses[n];
			invMas[n] = rno[n] * fTeCmTm + sum;
		}


		//-----> compute the weight of the current event

		GReal_t wt  = 1.0 / wtmax;

		GReal_t pd[N];

		#pragma unroll N
		for (size_t n = 0; n < N - 1; n++)
		{
			pd[n] = pdk(invMas[n + 1], invMas[n], fMasses[n + 1]);
			wt *= pd[n];
		}


		//-----> complete specification of event (Raubold-Lynch method)


		particles[1].set(sqrt(pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
				pd[0], 0.0);

		#pragma unroll N
		for (size_t i = 1; i < N; i++)
		{

			particles[i + 1].set(
					sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
					-pd[i - 1], 0.0);

			GReal_t cZ = 2	* uniDist(randEng) -1 ;
			GReal_t sZ = sqrt(1 - cZ * cZ);
			GReal_t angY = 2.0 * PI	* uniDist(randEng);
			GReal_t cY = cos(angY);
			GReal_t sY = sin(angY);
			for (size_t j = 0; j <= i; j++)
			{

				GReal_t x = particles[j + 1].get(1);
				GReal_t y = particles[j + 1].get(2);
				particles[j + 1].set(1, cZ * x - sZ * y);
				particles[j + 1].set(2, sZ * x + cZ * y); // rotation around Z

				x = particles[j + 1].get(1);
				GReal_t z = particles[j + 1].get(3);
				particles[j + 1].set(1, cY * x - sY * z);
				particles[j + 1].set(3, sY * x + cY * z); // rotation around Y
			}

			if (i == (N - 1))
				break;

			GReal_t beta = pd[i] / sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
			for (size_t j = 0; j <= i; j++)
			{

				particles[j+1 ].applyBoostTo(0, beta, 0);
			}

			}

		//
		//---> final boost of all particles to the mother's frame
		//
		#pragma unroll N
		for (size_t n = 0; n < N; n++)
		{

			particles[n+1].applyBoostTo(particles[0]);

		}

		//
		//---> return the weight of event
		//

		return wt;


	}


	template<typename Tuple>
	__host__   __device__ result_tuple_type
	operator()( Tuple &particles)
	{

		typedef typename hydra::detail::tuple_type<N+1,
				Vector4R>::type Tuple_t;

		constexpr size_t SIZE = thrust::tuple_size<Tuple_t>::value;

		Vector4R Particles[SIZE];

		Particles[0]= thrust::get<1>(particles);
		size_t evt  = thrust::get<0>(particles);
		GReal_t weight = process(evt, Particles);

		Tuple_t particles1{};

		hydra::detail::assignArrayToTuple(particles1, Particles   );

		return_tuple_type tmp = hydra::detail::invoke(particles1, fFunctors);

		return thrust::tuple_cat(thrust::make_tuple(weight), tmp );

	}

};

}//namespace detail

}//namespace hydra




#endif /* EVALMOTHERS_H_ */
