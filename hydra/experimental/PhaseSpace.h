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

/*-
 * PhaseSpace.h
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */


/**
 * \file
 * \ingroup phsp
 */


#ifndef _PHASESPACE_H_
#define _PHASESPACE_H_

#include <array>
#include <vector>
#include <string>
#include <map>
//#include <omp.h>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
//#include <math.h>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/experimental/Vector3R.h>
#include <hydra/experimental/Vector4R.h>
#include <hydra/experimental/Events.h>
#include <hydra/experimental/detail/functors/DecayMother.h>
#include <hydra/experimental/detail/functors/DecayMothers.h>
#include <hydra/experimental/detail/functors/EvalMother.h>
#include <hydra/experimental/detail/functors/EvalMothers.h>
#include <hydra/experimental/detail/functors/StatsPHSP.h>


#include <hydra/detail/functors/FlagAcceptReject.h>
#include <hydra/detail/functors/IsAccepted.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/experimental/detail/launch_decayers.inl>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <thrust/distance.h>


#include <thrust/system/omp/execution_policy.h>




namespace hydra {

namespace experimental {

template <size_t N, typename GRND=thrust::random::default_random_engine>
class PhaseSpace {

public:

	/**
	 * PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * - _MotherMass: the mass of the mother particle in Gev/c*c
	 * - _Masses: STL vector with the mass of the daughter particles.
	 */

	typedef typename hydra::detail::tuple_type<N, experimental::Vector4R>::type particle_tuple;


	PhaseSpace(const GReal_t motherMass, const GReal_t (&daughtersMasses)[N]) :
		fSeed(1)
{
		for(size_t i=0;i<N;i++)
			fMasses[i]= daughtersMasses[i];

		GReal_t fTeCmTm = 0.0;

		fTeCmTm = motherMass; // total energy in C.M. minus the sum of the masses

		for (size_t n = 0; n < N; n++)
			fTeCmTm -= daughtersMasses[n];

		if (fTeCmTm < 0.0) {
			std::cout << "Not enough energy for this decay. Exit." << std::endl;
			exit(1);
		}

}


	~PhaseSpace() {}

	template<typename FUNCTOR, hydra::detail::Backend BACKEND>
	std::pair<GReal_t, GReal_t> AverageOn(hydra::detail::BackendPolicy<BACKEND>const&,
			experimental::Vector4R const& mother, FUNCTOR const& functor, size_t n)
	{

		detail::EvalMother<N,GRND,FUNCTOR>
		evaluator( mother,fMasses, fSeed,functor);

		thrust::counting_iterator<GLong_t> first(0);

		thrust::counting_iterator<GLong_t> last = first + n;

		detail::StatsPHSP result = detail::launch_evaluator(hydra::detail::BackendPolicy<BACKEND>(),
				first, last, evaluator );
		return std::make_pair(result.fMean, sqrt(result.fM2) );

	}



	template<typename FUNCTOR, hydra::detail::Backend BACKEND>
	void Evaluate(hydra::detail::BackendPolicy<BACKEND>const&,
				experimental::Vector4R const& mother, FUNCTOR const& functor, size_t n)
		{

			detail::EvalMothers<N,GRND,FUNCTOR>
			evaluator(functor, mother,fMasses, fSeed);

			thrust::counting_iterator<GLong_t> first(0);

			thrust::counting_iterator<GLong_t> last = first + n;

			detail::StatsPHSP result = detail::launch_evaluator(hydra::detail::BackendPolicy<BACKEND>(),
					first, last, evaluator );
			return std::make_pair(result.fMean, sqrt(result.fM2) );

	}

	template<typename Iterator>
	void Generate(experimental::Vector4R const& mother, Iterator begin, Iterator end)
	{
		/**
		 * Run the generator and calculate the maximum weight. It takes as input the fourvector of the mother particle
		 * in any system of reference. The daughters will be generated in this system.
		*/

#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif


	detail::DecayMother<N,typename  hydra::detail::IteratorTraits<Iterator>::system_t,GRND> decayer(mother,fMasses, fSeed);
			detail::launch_decayer(begin, end, decayer );

	}

	template<typename Iterator1, typename Iterator2>
	void Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin)
	{
		/**
		 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
		 * in any system of reference. The daughters will be generated in this system.
		 */

#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif


	detail::DecayMothers<N, typename  hydra::detail::IteratorTraits<Iterator1>::system_t,GRND> decayer(fMasses, fSeed);
		detail::launch_decayer(begin, end, daughters_begin, decayer );



	}

	inline GInt_t GetSeed() const	{
			return fSeed;
		}

	inline void SetSeed(GInt_t _seed) 	{
				fSeed=_seed;
			}


	/**
	 * PDK function
	 */
	inline GReal_t PDK(const GReal_t a, const GReal_t b, const GReal_t c) const {
		//the PDK function
		GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
		x = sqrt(x) / (2 * a);
		return x;
	}

private:



	GInt_t  fSeed;///< seed.
	GReal_t fMasses[N];




};

}  // namespace experimental

}//namespace hydra
#include <hydra/detail/PhaseSpace.inl>

#endif /* PHASESPACE_H_ */
