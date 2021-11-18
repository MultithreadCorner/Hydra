/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * EvalMother.h
 *
 *  Created on: 24/05/2017
 *      Author: Antonio Augusto Alves Junior
 */



#ifndef EVALMOTHER_H_
#define EVALMOTHER_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>

#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/StatsPHSP.h>

//thrust
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/random.h>

#include <type_traits>
#include <utility>


namespace hydra {

namespace detail {


template <std::size_t N, typename GRND, typename FUNCTOR, typename ...FUNCTORS >
struct EvalMother
{
	typedef  hydra_thrust::tuple<FUNCTOR,FUNCTORS...> functors_tuple_type;

	typedef  hydra_thrust::tuple<typename FUNCTOR::return_type,
			typename FUNCTORS::return_type...>  return_tuple_type;

	typedef typename hydra::detail::tuple_cat_type<hydra_thrust::tuple<GReal_t> , return_tuple_type>::type
			result_tuple_type;


	std::size_t  fSeed;

	GReal_t fECM;
	GReal_t fMaxWeight;
	GReal_t fBeta0;
	GReal_t fBeta1;
	GReal_t fBeta2;


	GReal_t fMasses[N];
	functors_tuple_type fFunctors ;

	//constructor
	EvalMother(Vector4R const& mother,
			const GReal_t (&masses)[N],
			double maxweight, double ecm, std::size_t seed,
			FUNCTOR const& functor, FUNCTORS const& ...functors ):
			fMaxWeight(maxweight),
			fECM(ecm),
			fSeed(seed),
			fFunctors( hydra_thrust::make_tuple(functor,functors...))
	{

		for(std::size_t i=0; i<N; i++)
			fMasses[i]=masses[i];

		GReal_t beta = mother.d3mag() / mother.get(0);

		if (beta)
		{
			GReal_t w = beta / mother.d3mag();
			fBeta0 = mother.get(0) * w;
			fBeta1 = mother.get(1) * w;
			fBeta2 = mother.get(2) * w;
		}
		else
			fBeta0 = fBeta1 = fBeta2 = 0.0;


	}

	__hydra_host__ __hydra_device__
	EvalMother( EvalMother<N, GRND, FUNCTOR,FUNCTORS...> const& other ):
	fFunctors(other.fFunctors),
	fSeed(other.fSeed ),
	fECM(other.fECM ),
	fMaxWeight(other.fMaxWeight ),
	fBeta0(other.fBeta0 ),
	fBeta1(other.fBeta1 ),
	fBeta2(other.fBeta2 )
	{ for(std::size_t i=0; i<N; i++) fMasses[i]=other.fMasses[i]; }



	__hydra_host__      __hydra_device__ inline
	static GReal_t pdk(const GReal_t a, const GReal_t b,
			const GReal_t c)
	{
		//the PDK function
		//GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
		//x = sqrt( x ) / (2 * a);
		return ::sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);
	}

	__hydra_host__ __hydra_device__ inline
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


	__hydra_host__   __hydra_device__ inline
	constexpr static std::size_t hash(const std::size_t a, const std::size_t b)
	{
		//Matthew Szudzik pairing
		//http://szudzik.com/ElegantPairing.pdf
        return   (((2 * a) >=  (2 * b) ? (2 * a) * (2 * a) + (2 * a) + (2 * b) : (2 * a) + (2 * b) * (2 * b)) / 2);
	}

	__hydra_host__   __hydra_device__ inline
	GReal_t process(std::size_t evt, Vector4R (&daugters)[N])
	{

		GRND randEng( fSeed );//hash(evt,fSeed) );
		randEng.discard(evt+3*N);
		hydra_thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

		GReal_t rno[N];
		rno[0] = 0.0;
		rno[N - 1] = 1.0;

		if (N > 2)
		{
//#pragma unroll N
			for (std::size_t n = 1; n < N - 1; n++)
			{
				rno[n] =  uniDist(randEng) ;

			}

			bbsort(&rno[1], N -2);

		}


		GReal_t invMas[N], sum = 0.0;

//#pragma unroll N
		for (std::size_t n = 0; n < N; n++)
		{
			//printf("%d mass=%f \n",n, fMasses[n]);
			sum += fMasses[n];
			invMas[n] = rno[n] * fECM + sum;
		}

		//
		//-----> compute the weight of the current event
		//

		GReal_t wt = fMaxWeight;

		GReal_t pd[N];

//#pragma unroll N
		for (std::size_t n = 0; n < N - 1; n++)
		{
			pd[n] = pdk(invMas[n + 1], invMas[n], fMasses[n + 1]);
			wt *= pd[n];
		}

		//
		//-----> complete specification of event (Raubold-Lynch method)
		//

		daugters[0].set(::sqrt((GReal_t) pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
				pd[0], 0.0);

//#pragma unroll N
		for (std::size_t i = 1; i < N; i++)
		{

			daugters[i].set(
					::sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
					-pd[i - 1], 0.0);

			GReal_t cZ = 2 * uniDist(randEng) -1 ;
			GReal_t sZ = ::sqrt(1 - cZ * cZ);
			GReal_t angY = 2 * PI* uniDist(randEng);
			GReal_t cY = ::cos(angY);
			GReal_t sY = ::sin(angY);
			for (std::size_t j = 0; j <= i; j++)
			{

				GReal_t x = daugters[j].get(1);
				GReal_t y = daugters[j].get(2);
				daugters[j].set(1, cZ * x - sZ * y);
				daugters[j].set(2, sZ * x + cZ * y); // rotation around Z

				x = daugters[j].get(1);
				GReal_t z = daugters[j].get(3);
				daugters[j].set(1, cY * x - sY * z);
				daugters[j].set(3, sY * x + cY * z); // rotation around Y
			}

			if (i == (N - 1))
				break;

			GReal_t beta = pd[i] / ::sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
			for (std::size_t j = 0; j <= i; j++)
			{

				daugters[j].applyBoostTo(Vector3R(0, beta, 0));
			}

		}

		//
		//---> final boost of all particles to the mother's frame
		//
//#pragma unroll N
		for (std::size_t n = 0; n < N; n++)
		{

			daugters[n].applyBoostTo(Vector3R(fBeta0, fBeta1, fBeta2));

		}

		//
		//---> return the weight of event
		//

		return wt;

	}

	template< typename I>
	__hydra_host__   __hydra_device__
	inline result_tuple_type operator()( I evt )
	{
		typedef typename hydra::detail::tuple_type<N,
				Vector4R>::type Tuple_t;

		constexpr std::size_t SIZE = hydra_thrust::tuple_size<Tuple_t>::value;

		Vector4R Particles[SIZE];

		GReal_t weight = process(evt, Particles);

		Tuple_t particles{};

		hydra::detail::assignArrayToTuple(particles, Particles   );

		return_tuple_type tmp = hydra::detail::invoke(particles, fFunctors);


		return hydra_thrust::tuple_cat(hydra_thrust::make_tuple(weight), tmp );

	}

};

}//namespace detail

}//namespace hydra


#endif /* EVALMOTHER_H_ */
