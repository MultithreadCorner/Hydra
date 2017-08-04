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
 * Event.h
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef _EVENTS_H_
#define _EVENTS_H_

#include <array>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <utility>
#include <thrust/copy.h>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/multivector.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/FlagAcceptReject.h>



namespace hydra {

template<size_t N, typename BACKEND>
class Events;

/**
 * @ingroup phsp
 * @brief Events is a container struct to hold all the information corresponding the generated events.
 * Mother's four-vectors are not stored.
 *
 * When iterating over the container, an event with N particles in the final state is
 * returned as a tuple with the element 0 storing the weight and the other
 * elements storing the four-vectors (hydra::Vector4R) of each particle.
 * The particles are stored in the same order of the array of masses.
 *
 * @tparam N number of particles in the final state.
 * @tparam BACKEND memory space(backend) to store the events.
 */
template<size_t N, hydra::detail::Backend BACKEND>
class Events<N, hydra::detail::BackendPolicy<BACKEND> > {


	typedef hydra::detail::BackendPolicy<BACKEND> system_t;

	typedef typename system_t::template container<Vector4R::args_type> super_t;
public:
	typedef multivector<super_t> vector_particles;
	typedef typename multivector<super_t>::iterator vector_particles_iterator;
	typedef typename multivector<super_t>::const_iterator vector_particles_const_iterator;

	typedef typename system_t::template container<GReal_t>  vector_real;
	typedef typename system_t::template container<GReal_t>::iterator vector_real_iterator;
	typedef typename system_t::template container<GReal_t>::const_iterator vector_real_const_iterator;

	typedef typename system_t::template container<GBool_t>  vector_bool;
	typedef typename system_t::template container<GBool_t>::iterator  vector_bool_iterator;
	typedef typename system_t::template container<GBool_t>::const_iterator  vector_bool_const_iterator;

	typedef  decltype( hydra::detail::get_zip_iterator( vector_real_iterator(), std::array< vector_particles_iterator,N>() )) iterator;
	typedef  decltype( hydra::detail::get_zip_iterator( vector_real_const_iterator(),
			std::array< vector_particles_const_iterator,N>() )) const_iterator;
	typedef   typename iterator::value_type value_type;
	typedef   typename iterator::reference  reference_type;

	constexpr static size_t particles = N ;


    /**
     * @brief Default constructor.
     */
	Events():
		fNEvents(0),
		fMaxWeight(0){}

	/**
	 * @brief Constructor with n elements.
	 * @param size_t nevents
	 */
	Events(size_t nevents);

	/**
	 * @brief Cross-backend copy constructor.
	 * @param Events<N,BACKEND2> const& other
	 */
	template<hydra::detail::Backend BACKEND2>
	Events(Events<N,hydra::detail::BackendPolicy<BACKEND2>> const& other);

	/**
	 * @brief Copy constructor.
	 * @param Events<N,BACKEND> const& other
	 */
	Events(Events<N,hydra::detail::BackendPolicy<BACKEND>> const& other);

	/**
	 * @brief Move constructor.
	 * Move the resources from other to *this and set other=default.
	 * @param Events<N,BACKEND> && other
	 */
	Events(Events<N, hydra::detail::BackendPolicy<BACKEND>> && other);

	/**
	 * @brief Assignment operator
	 * Assignment operator.
	 * @param Events<N,BACKEND> other
	 * @return Events<N,BACKEND>
	 */
	Events<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=(Events<N,hydra::detail::BackendPolicy<BACKEND>> const& other);

	/**
	 * @brief Generic assignment operator
	 * Cross-backend assignment operator.
	 * @param Events<N,BACKEND2>const& other
	 * @return
	 */
	template<hydra::detail::Backend BACKEND2>
	Events<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=(Events<N,hydra::detail::BackendPolicy<BACKEND2>> const& other);

	/**
	 * @brief Move assignment operator
	 * Move the resources from other to *this and set other=default.
	 * @param Events<N,BACKEND2>&& other
	 * @return
	 */

	Events<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=(Events<N,hydra::detail::BackendPolicy<BACKEND>>&& other);

	~Events(){};

	/**
	 * @brief Get maximum weight in the container.
	 * Get maximum weight in the container.
	 * @return fMaxWeight
	 */
	GReal_t GetMaxWeight() const {
		return fMaxWeight;
	}

	/**
	 * @brief Get number of events in the container.
	 *
	 * @return size of the container.
	 */
	size_t GetNEvents() const {
		return fFlags.size();
	}

    /**
     * @brief Get flags begin
     * Get begin iterator to accepted-rejected flags.
     * @return
     */
	vector_bool_const_iterator FlagsBegin() const{
		return fFlags.begin();
	}

	/**
	 * @brief Get flags end
	 * @return
	 */
	vector_bool_const_iterator FlagsEnd() const{
		return fFlags.end();
	}

	vector_real_const_iterator WeightsBegin() const{
		return fWeights.begin();
	}

	vector_real_const_iterator WeightsEnd() const{
		return fWeights.end();
	}

	vector_particles_const_iterator DaughtersBegin(GInt_t i)const{

		return fDaughters[i].cbegin();
	}

	vector_particles_const_iterator DaughtersEnd(GInt_t i)	const{

		return fDaughters[i].cend();
	}

	vector_bool_iterator FlagsBegin() {
		return fFlags.begin();
	}

	vector_bool_iterator FlagsEnd() {
		return fFlags.end();
	}

	vector_real_iterator WeightsBegin() {
		return fWeights.begin();
	}

	vector_real_iterator WeightsEnd() {
		return fWeights.end();
	}

	vector_particles_iterator DaughtersBegin(GInt_t i){

		return fDaughters[i].begin();
	}

	vector_particles_iterator DaughtersEnd(GInt_t i)	{

		return fDaughters[i].end();
	}

	void SetMaxWeight(GReal_t maxWeight) {

		fMaxWeight = maxWeight;
	}


	reference_type operator[](size_t i)
	{

		return fBegin[i];
	}

	reference_type operator[](size_t i) const
	{

		return fConstBegin[i];
	}



	iterator begin(){ return fBegin; }

	iterator  end(){ return fEnd; }

	const_iterator begin() const{ return fConstBegin; }

	const_iterator  end() const{ return fConstEnd; }

	const_iterator cbegin() const{ return fConstBegin; }

	const_iterator  cend() const{ return fConstEnd; }

	size_t capacity() const  {return fFlags.capacity();}

	void resize(size_t n);

	size_t Unweight(size_t seed);

	size_t size() const { return fFlags.size(); }


private:

	vector_bool MoveFlags()	{
		return std::move(fFlags);
	}

	vector_real MoveWeights(){
		return std::move(fWeights);
	}

	std::array<vector_particles,N> MoveDaughters(){
		return std::move(fDaughters);
	}

	size_t fNEvents;    ///< Number of events.
	GReal_t fMaxWeight;  ///< Maximum weight of the generated events.
	vector_bool fFlags; ///< Vector of flags. Accepted events are flagged 1 and rejected 0.
	vector_real fWeights; ///< Vector of event weights.
	std::array<vector_particles,N> fDaughters; ///< Array of daughter particle vectors.
	iterator fBegin;
	iterator fEnd;
	const_iterator fConstBegin;
	const_iterator fConstEnd;

};



}// namespace hydra

#include <hydra/detail/Events.inl>

#endif /* EVENT_H_ */
