/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
//#include <hydra/Events.h>
#include <hydra/detail/functors/DecayMother.h>
#include <hydra/detail/functors/DecayMothers.h>
#include <hydra/detail/functors/EvalMother.h>
#include <hydra/detail/functors/EvalMothers.h>
#include <hydra/detail/functors/StatsPHSP.h>
#include <hydra/detail/Print.h>
#include <hydra/detail/functors/CheckEnergy.h>
#include <hydra/Tuple.h>
#include <hydra/detail/Hash.h>

#include <hydra/Decays.h>

#include <hydra/detail/launch_decayers.inl>

#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/random.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/equal.h>

#include <array>
#include <vector>
#include <utility>
#include <initializer_list>

namespace hydra {


/**
 * \ingroup phsp
 * \brief This class implements phase-space Monte Carlo generation in hydra.
 *
 * The events are generated in the center-of-mass frame, but the decay products are finally boosted using the betas of the original particle.
 * The code is based on the Raubold and Lynch method as documented in [F. James, Monte Carlo Phase Space, CERN 68-15 (1968)](https://cds.cern.ch/record/275743).
 *
 * Note that Momentum, Energy units are @f$GeV/C@f$ , @f$GeV/C^2@f$ .
 *
 *\tparam N is the number of particles in final state.
 *\tparam GRND underlying random number generator. See the options in HYDRA_EXTERNAL_NS::thrust::random namespace.
 */
template <size_t N, typename GRND=HYDRA_EXTERNAL_NS::thrust::random::default_random_engine>
class PhaseSpace {

public:

	/**
	 * @brief PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses array with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace(const GReal_t (&daughtersMasses)[N]);

	/**
	 * @brief PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses array with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace( std::array<GReal_t,N> const& daughtersMasses);

	/**
	 * @brief PhaseSpace ctor. Constructor of the phase-space generator takes as input parameters:
	 * @param motherMass mass of the mother particle in Gev/c*c;
	 * @param daughtersMasses list with the masses of the daughter particles in Gev/c*c;
	 */
	PhaseSpace(std::initializer_list<GReal_t> const& daughtersMasses);

	/**
	 * @brief Copy constructor.
	 * @param other
	 */
	PhaseSpace( PhaseSpace<N,GRND>const& other);

	/**
	 * @brief Copy constructor.
	 * @param other
	 */
	template<typename GRND2>
	PhaseSpace( PhaseSpace<N,GRND2>const& other);


	/**
		 * @brief Copy constructor.
		 * @param other
		 */
	PhaseSpace<N,GRND>&
	operator=( PhaseSpace<N,GRND>const& other);

		/**
		 * @brief Copy constructor.
		 * @param other
		 */
		template<typename GRND2>
		PhaseSpace<N,GRND>&
		operator=( PhaseSpace<N,GRND2>const& other);




	/**
	 * @brief Calculate the mean and the \f$ \sqrt(variance)\f$  of a functor over the phase-space with n-samples.
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
	 * @brief Calculate the mean and the \f$ \sqrt(\sigma)\f$  of a functor over the phase-space given a list of mother particles.
	 * @param policy Back-end;
	 * @param begin Iterator pointing to the begin of list of mother particles;
	 * @param end   Iterator pointing to the end of list of mother particles;
	 * @param functor Functor;
	 * @return std::pair with the mean and the \f$ \sqrt(\sigma)\f$
	 */
	template<typename FUNCTOR,  typename Iterator>
	std::pair<GReal_t, GReal_t> AverageOn(Iterator begin, Iterator end, FUNCTOR const& functor);


	/**
	 * @brief Evaluate a list of functors  over the phase-space
	 * @param policy  Back-end;
	 * @param begin Iterator pointing to the begin of list of output range;
	 * @param end   Iterator pointing to the end of list of output range;
	 * @param mother Mother particle four-vector;
	 * @param functors Functors;
	 */
	template<typename Iterator, typename ...FUNCTOR >
	void Evaluate(Vector4R const& mother, Iterator begin, Iterator end, FUNCTOR const& ...functors);

	/**
	 * @brief Evaluate a list of functors  over the phase-space given a list vectors.
	 * @param policy Back-end;
	 * @param mbegin Iterator pointing to the begin of list of mother particles;
	 * @param mend Iterator pointing to the end of list of mother particles;
	 * @param begin Iterator pointing to the begin of list of output range;
	 * @param functors Functors;
	 */
	template<typename ...FUNCTOR, typename IteratorMother, typename Iterator>
	void Evaluate(IteratorMother mbegin, IteratorMother mend,
			Iterator begin, FUNCTOR const& ...functors);

	//Evaluate range semantics interface ----------------------------

	/**
	 * @brief Evaluate a list of functors  over the phase-space
	 * @param mother mother particle
	 * @param result container for store the results
	 * @param functors
	 * @return A Range object pointing to the @param result container
	 */
	template<typename ...FUNCTOR, typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
			 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Evaluate(Vector4R const& mother, Iterable&& iterable, FUNCTOR const& ...functors);

	/**
	 * @brief Evaluate a list of functors  over the phase-space given a list vectors.
	 * @param mothers list of mother particles;
	 * @param result container for store the results
	 * @param functors
	 * @return A Range object pointing to the @param result container
	 */
	template<typename ...FUNCTOR, typename IterableMother, typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value &&
	hydra::detail::is_iterable<IterableMother>::value,
				 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Evaluate(IterableMother&& mothers, Iterable&& result, FUNCTOR const& ...functors);

	//--------------------------------------------------------------------

    /**
     * @brief Generate a phase-space  given a mother particle and a output range.
     * @param mother Mother particle.
     * @param begin Iterator pointing to the begin output range.
     * @param end Iterator pointing to the end output range.
     */
	template<typename Iterator>
	void Generate(Vector4R const& mother, Iterator begin, Iterator end);

	/**
	 * @brief Generate a phase-space  given a range of mother particles and a output range.
	 * @param begin Iterator pointing to the begin of range of mother particles.
	 * @param end Iterator pointing to the end  of range of mother particles.
	 * @param daughters_begin Iterator pointing to the begin of range of daughter particles.
	 */
	template<typename Iterator1, typename Iterator2>
	void Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin);

	/**
	 * @brief Generate a phase-space  given a mother particle and a output range.
	 * @param mother Mother particle.
	 * @param begin Iterator pointing to the begin output range.
	 * @param end Iterator pointing to the end output range.
	 */
	template<typename Iterator, hydra::detail::Backend BACKEND>
	void Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy ,Vector4R const& mother, Iterator begin, Iterator end);

	/**
	 * @brief Generate a phase-space  given a range of mother particles and a output range.
	 * @param begin Iterator pointing to the begin of range of mother particles.
	 * @param end Iterator pointing to the end  of range of mother particles.
	 * @param daughters_begin Iterator pointing to the begin of range of daughter particles.
	 */
	template<typename Iterator1, typename Iterator2, hydra::detail::Backend BACKEND>
	void Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy , Iterator1 begin, Iterator1 end, Iterator2 daughters_begin);

	// Generate range semantics ------------------------------------------------
	/**
	 * @brief Generate a phase-space  given a mother particle and a output range.
	 * @param mother Mother particle.
	 * @param begin Iterator pointing to the begin output range.
	 * @param end Iterator pointing to the end output range.
	 */
	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
				 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Generate(Vector4R const& mother, Iterable&& events);

	/**
	 * @brief Generate a phase-space  given a range of mother particles and a output range.
	 * @param begin Iterator pointing to the begin of range of mother particles.
	 * @param end Iterator pointing to the end  of range of mother particles.
	 * @param daughters_begin Iterator pointing to the begin of range of daughter particles.
	 */
	template<typename IterableMothers, typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value &&
		hydra::detail::is_iterable<IterableMothers>::value,
					 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
	Generate( IterableMothers&& mothers, Iterable&& daughters);

	//--------------------------------------------------------------------------


	/**
	 * @brief Get seed of the underlying generator;
	 * @return
	 */
	inline GInt_t GetSeed() const;

	/**
	 * @brief Set seed of the underlying generator;
	 * @param _seed
	 */
	inline void SetSeed(GInt_t _seed) ;

	const GReal_t* GetMasses() const {
		return fMasses;
	}





private:
/**
	 * @brief PDK function
	 */
	inline GReal_t PDK(const GReal_t a, const GReal_t b, const GReal_t c) const ;

	template<typename Iterator>
	inline bool EnergyChecker(Iterator first, Iterator last) const;

	inline bool EnergyChecker( Vector4R const& mother) const;



	size_t  fSeed;///< seed.
	GReal_t fMasses[N];

};


}//namespace hydra

#include <hydra/detail/PhaseSpace.inl>

#endif /* PHASESPACE_H_ */
