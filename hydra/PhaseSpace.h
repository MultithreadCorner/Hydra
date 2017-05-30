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



#ifndef _PHASESPACE_H_
#define _PHASESPACE_H_

#include <array>
#include <vector>
#include <initializer_list>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/detail/functors/FlagAcceptReject.h>
#include <hydra/detail/functors/IsAccepted.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/Events.h>
#include <hydra/detail/functors/DecayMother.h>
#include <hydra/detail/functors/DecayMothers.h>
#include <hydra/detail/functors/EvalMother.h>
#include <hydra/detail/functors/EvalMothers.h>
#include <hydra/detail/functors/StatsPHSP.h>



#include <hydra/detail/launch_decayers.inl>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <thrust/distance.h>

/**
 * \file
 * \ingroup Phase-space generator
 */

namespace hydra {


template <size_t N, typename GRND=thrust::random::default_random_engine>
class PhaseSpace {

public:

	/**
	 * PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses array with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace(const GReal_t motherMass, const GReal_t (&daughtersMasses)[N]);

	/**
	 * PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses array with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace(const GReal_t motherMass, std::array<GReal_t,N> const& daughtersMasses);

	/**
	 * PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses list with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace(const GReal_t motherMass, std::initializer_list<GReal_t> const& daughtersMasses);

	/**
	 * Copy constructor.
	 * @param other
	 */
	PhaseSpace( PhaseSpace<N,GRND>const& other);

	/**
	 * Calculate the mean and the \f$ \sqrt(variance)\f$  of a functor over the phase-space with n-samples.
	 * @param policy  Back-end;
	 * @param mother  Mother particle four-vector;
	 * @param functor Functor;
	 * @param n Number of samples;
	 * @return std::pair with the mean and the \f$ \sqrt(variance)\f$
	 */
	template<typename FUNCTOR, hydra::detail::Backend BACKEND>
	std::pair<GReal_t, GReal_t> AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy,
			Vector4R const& mother, FUNCTOR const& functor, size_t n) ;

	/**
	 * Calculate the mean and the \f$ \sqrt(variance)\f$  of a functor over the phase-space given a list of mother particles.
	 * @param policy Back-end;
	 * @param begin Iterator pointing to the begin of list of mother particles;
	 * @param end   Iterator pointing to the end of list of mother particles;
	 * @param functor Functor;
	 * @return std::pair with the mean and the \f$ \sqrt(variance)\f$
	 */
	template<typename FUNCTOR, hydra::detail::Backend BACKEND, typename Iterator>
	std::pair<GReal_t, GReal_t> AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy,
			Iterator begin, Iterator end, FUNCTOR const& functor);

	/**
	 * Evaluate a list of functors  over the phase-space
	 * @param policy  Back-end;
	 * @param begin Iterator pointing to the begin of list of output range;
	 * @param end   Iterator pointing to the end of list of output range;
	 * @param mother Mother particle four-vector;
	 * @param functors Functors;
	 */
	template<typename ...FUNCTOR, hydra::detail::Backend BACKEND, typename Iterator>
	void Evaluate(hydra::detail::BackendPolicy<BACKEND>const& policy, Iterator begin, Iterator end,
			Vector4R const& mother, FUNCTOR const& ...functors);

	/**
	 * Evaluate a list of functors  over the phase-space given a list vectors.
	 * @param policy Back-end;
	 * @param mbegin Iterator pointing to the begin of list of mother particles;
	 * @param mend Iterator pointing to the end of list of mother particles;
	 * @param begin Iterator pointing to the begin of list of output range;
	 * @param functors Functors;
	 */
	template<typename ...FUNCTOR, hydra::detail::Backend BACKEND,
	typename IteratorMother, typename Iterator>
	void Evaluate(hydra::detail::BackendPolicy<BACKEND>const& policy, IteratorMother mbegin,
			IteratorMother mend, Iterator begin, FUNCTOR const& ...functors);

    /**
     * Generate a phase-space  given a mother particle and a output range.
     * @param mother Mother particle.
     * @param begin Iterator pointing to the begin output range.
     * @param end Iterator pointing to the end output range.
     */
	template<typename Iterator>
	void Generate(Vector4R const& mother, Iterator begin, Iterator end);

	/**
	 * Generate a phase-space  given a range of mother particles and a output range.
	 * @param begin Iterator pointing to the begin of range of mother particles.
	 * @param end Iterator pointing to the end  of range of mother particles.
	 * @param daughters_begin Iterator pointing to the begin of range of daughter particles.
	 */
	template<typename Iterator1, typename Iterator2>
	void Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin);

	/**
	 * Get seed of the underlying generator;
	 * @return
	 */
	inline GInt_t GetSeed() const;

	/**
	 * Set seed of the underlying generator;
	 * @param _seed
	 */
	inline void SetSeed(GInt_t _seed) ;


	/**
	 * PDK function
	 */



private:

	inline GReal_t PDK(const GReal_t a, const GReal_t b, const GReal_t c) const ;

	size_t  fSeed;///< seed.
	GReal_t fMasses[N];

};


}//namespace hydra

#include <hydra/detail/PhaseSpace.inl>

#endif /* PHASESPACE_H_ */
