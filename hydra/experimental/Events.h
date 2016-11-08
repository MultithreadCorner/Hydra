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

/**
 * \file
 * \ingroup phsp
 */


#ifndef _EVENTS_H_
#define _EVENTS_H_

#include <array>
#include <vector>
#include <string>
#include <map>
#include <omp.h>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <utility>
#include <thrust/copy.h>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/experimental/Vector3R.h>
#include <hydra/experimental/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/FlagAcceptReject.h>
#include <hydra/experimental/multivector.h>

namespace hydra {

namespace experimental {
/*! \struct Events
 * Events is a container struct to hold all the information corresponding the generated events.
 * Mother four-vectors are not stored.
 */
template<size_t N, unsigned int BACKEND>
struct Events {

	typedef hydra::detail::BackendTraits<BACKEND> system_t;

	typedef typename system_t::template container<Vector4R::args_type> super_t;

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


    /*!
	 * Constructor takes as parameters the number of particles and number of events.
	 */
    Events() = delete;

	Events(size_t nevents) :
		fNEvents(nevents),
		fMaxWeight(0)
	{

		for (GInt_t d = 0; d < N; d++)
		{
			fDaughters[d].resize(fNEvents);
		}

		fWeights.resize(fNEvents);
		fFlags.resize(fNEvents);

		std::array< vector_particles_iterator,N> begins;
		std::array< vector_particles_iterator,N> ends;

		std::array< vector_particles_const_iterator,N> cbegins;
		std::array< vector_particles_const_iterator,N> cends;



#pragma unroll N
		for(int i =0; i < N; i++){
			begins[i]= fDaughters[i].begin();
			ends[i]= fDaughters[i].end();

			cbegins[i]= fDaughters[i].cbegin();
			cends[i]= fDaughters[i].cend();


		}
		 fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
		 fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

		 fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
		 fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );

	}
//---------
	// copy

	template<unsigned int BACKEND2>
	Events(Events<N,BACKEND2> const& other):
			fNEvents(other.GetNEvents()),
			fMaxWeight(other.GetMaxWeight())
		{

#pragma unroll N
		for (GInt_t d = 0; d < N; d++)
		{
			fDaughters[d].resize(fNEvents);
		}

		fWeights.resize(fNEvents);
		fFlags.resize(fNEvents);

			for (GInt_t i = 0; i < N; i++)
			{
				thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
			}

			thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
			thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

			std::array< vector_particles_iterator,N> begins;
			std::array< vector_particles_iterator,N> ends;

			std::array< vector_particles_const_iterator,N> cbegins;
			std::array< vector_particles_const_iterator,N> cends;



#pragma unroll N
			for(int i =0; i < N; i++){
				begins[i]= fDaughters[i].begin();
				ends[i]= fDaughters[i].end();

				cbegins[i]= fDaughters[i].cbegin();
				cends[i]= fDaughters[i].cend();


			}
			fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
			fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

			fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
			fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );

		}
	//---------
		// copy


		Events(Events<N,BACKEND> const& other):
				fNEvents(other.GetNEvents()),
				fMaxWeight(other.GetMaxWeight())
			{

	#pragma unroll N
			for (GInt_t d = 0; d < N; d++)
			{
				fDaughters[d].resize(fNEvents);
			}

			fWeights.resize(fNEvents);
			fFlags.resize(fNEvents);

				for (GInt_t i = 0; i < N; i++)
				{
					thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
				}

				thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
				thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

				std::array< vector_particles_iterator,N> begins;
				std::array< vector_particles_iterator,N> ends;

				std::array< vector_particles_const_iterator,N> cbegins;
				std::array< vector_particles_const_iterator,N> cends;



	#pragma unroll N
				for(int i =0; i < N; i++){
					begins[i]= fDaughters[i].begin();
					ends[i]= fDaughters[i].end();

					cbegins[i]= fDaughters[i].cbegin();
					cends[i]= fDaughters[i].cend();


				}
				fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
				fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

				fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
				fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );

			}
/*
 * move
 */

	Events(Events<N,BACKEND> && other):
	fNEvents(other.GetNEvents()),
	fMaxWeight(other.GetMaxWeight()),
	fWeights(std::move(other.WeightsMove())),
	fFlags(std::move(other.FlagsMove())),
	fDaughters(std::move(other.DaughtersMove()))
	{

		std::array< vector_particles_iterator,N> begins;
		std::array< vector_particles_iterator,N> ends;

		std::array< vector_particles_const_iterator,N> cbegins;
		std::array< vector_particles_const_iterator,N> cends;



#pragma unroll N
		for(int i =0; i < N; i++){
			begins[i]= fDaughters[i].begin();
			ends[i]= fDaughters[i].end();

			cbegins[i]= fDaughters[i].cbegin();
			cends[i]= fDaughters[i].cend();


		}
		fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
		fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

		fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
		fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );
	}

	template<unsigned int BACKEND2>
	Events<N,BACKEND>& operator=(Events<N,BACKEND2> const& other)
	{

		this->fNEvents=other.GetNEvents();
		this->fMaxWeight=other.GetMaxWeight();

#pragma unroll N
		for (GInt_t d = 0; d < N; d++)
		{
			fDaughters[d].resize(fNEvents);
		}

		fWeights.resize(fNEvents);
		fFlags.resize(fNEvents);


		for (GInt_t i = 0; i < N; i++)
		{
			thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
		}

		thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
		thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

		std::array< vector_particles_iterator,N> begins;
		std::array< vector_particles_iterator,N> ends;

		std::array< vector_particles_const_iterator,N> cbegins;
		std::array< vector_particles_const_iterator,N> cends;



#pragma unroll N
		for(int i =0; i < N; i++){
			begins[i]= fDaughters[i].begin();
			ends[i]= fDaughters[i].end();

			cbegins[i]= fDaughters[i].cbegin();
			cends[i]= fDaughters[i].cend();


		}
		this->fBegin = hydra::detail::get_zip_iterator(this->fWeights.begin(), begins );
		this->fEnd = hydra::detail::get_zip_iterator(this->fWeights.end(), ends );

		this->fConstBegin = hydra::detail::get_zip_iterator(this->fWeights.cbegin(), cbegins );
		this->fConstEnd = hydra::detail::get_zip_iterator(this->fWeights.cend(), cends );


		return *this;
	}

	~Events(){};

	GReal_t GetMaxWeight() const {
		return fMaxWeight;
	}

	size_t GetNEvents() const {
		return fFlags.size();
	}


	vector_bool_const_iterator FlagsBegin() const{
		return fFlags.begin();
	}

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



	GULong_t Unweight(size_t seed)
	{
		/**
		 * Flag the accepted and rejected events
		 */



		GULong_t count = 0;
		if(N==2)
		{
			thrust::fill(fFlags.begin(), fFlags.end(), kTrue);
			count = fNEvents;
		}
		else
		{

		auto w = thrust::max_element(fWeights.begin(),fWeights.end());
		fMaxWeight=*w;
		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + fNEvents;


		thrust::transform(first, last, fWeights.begin(),
				fFlags.begin(), hydra::detail::FlagAcceptReject(seed, fMaxWeight));

		count = thrust::count(fFlags.begin(), fFlags.end(),
				kTrue);

		}

		return count;

	}

	vector_bool MoveFlags()	{
		return std::move(fFlags);
	}

	vector_real MoveWeights(){
		return std::move(fWeights);
	}

	std::array<vector_particles,N> MoveDaughters(){
		return std::move(fDaughters);
	}


private:



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

}  // namespace experimental

}// namespace hydra

#endif /* EVENT_H_ */
