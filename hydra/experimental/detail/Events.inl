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
 * Events.inl
 *
 *  Created on: 09/11/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef EVENTS_INL_
#define EVENTS_INL_

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


template<size_t N, unsigned int BACKEND>
Events<N, BACKEND>::Events(size_t nevents):
fNEvents(nevents),
fMaxWeight(0)
{

	for(size_t d = 0; d < N; d++)
		fDaughters[d].resize(fNEvents);

	fWeights.resize(fNEvents);
	fFlags.resize(fNEvents);

	std::array< vector_particles_iterator,N> begins;
	std::array< vector_particles_iterator,N> ends;

	std::array< vector_particles_const_iterator,N> cbegins;
	std::array< vector_particles_const_iterator,N> cends;

#pragma unroll N
	for(int i =0; i < N; i++){
		begins[i] = fDaughters[i].begin();
		ends[i]   = fDaughters[i].end();
		cbegins[i]= fDaughters[i].cbegin();
		cends[i]  = fDaughters[i].cend();
	}

	fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
	fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

	fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
	fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );

}

template<size_t N, unsigned int BACKEND>
template<unsigned int BACKEND2>
Events<N, BACKEND>::Events(hydra::experimental::Events<N,BACKEND2> const& other):
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

	std::array<vector_particles_iterator,N> begins;
	std::array<vector_particles_iterator,N> ends;

	std::array< vector_particles_const_iterator,N> cbegins;
	std::array< vector_particles_const_iterator,N> cends;

#pragma unroll N
	for(int i =0; i < N; i++){
		begins[i] = fDaughters[i].begin();
		ends[i]   = fDaughters[i].end();
		cbegins[i]= fDaughters[i].cbegin();
		cends[i]  = fDaughters[i].cend();
	}

	fBegin = hydra::detail::get_zip_iterator(fWeights.begin(), begins );
	fEnd = hydra::detail::get_zip_iterator(fWeights.end(), ends );

	fConstBegin = hydra::detail::get_zip_iterator(fWeights.cbegin(), cbegins );
	fConstEnd = hydra::detail::get_zip_iterator(fWeights.cend(), cends );

}

template<size_t N, unsigned int BACKEND>
Events<N, BACKEND>::Events(hydra::experimental::Events<N,BACKEND> const& other):
fNEvents(other.GetNEvents()),
fMaxWeight(other.GetMaxWeight())
{

#pragma unroll N
	for (GInt_t d = 0; d < N; d++)
		fDaughters[d].resize(fNEvents);

	fWeights.resize(fNEvents);
	fFlags.resize(fNEvents);

	for (GInt_t i = 0; i < N; i++)
	{
		thrust::copy(other.DaughtersBegin(i),
				other.DaughtersEnd(i), this->DaughtersBegin(i));
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

template<size_t N, unsigned int BACKEND>
Events<N, BACKEND>::Events(hydra::experimental::Events<N,BACKEND> && other):
fNEvents(other.GetNEvents()),
fMaxWeight(other.GetMaxWeight()),
fWeights(std::move(other.MoveWeights())),
fFlags(std::move(other.MoveFlags())),
fDaughters(std::move(other.MoveDaughters()))
{
	std::array< vector_particles_iterator,N> begins;
	std::array< vector_particles_iterator,N> ends;

	std::array< vector_particles_const_iterator,N> cbegins;
	std::array< vector_particles_const_iterator,N> cends;

	other=Events<N, BACKEND>();

#pragma unroll N
	for(int i =0; i < N; i++){
		begins[i]= this->fDaughters[i].begin();
		ends[i]= this->fDaughters[i].end();

		cbegins[i]= this->fDaughters[i].cbegin();
		cends[i]= this->fDaughters[i].cend();
	}

	fBegin = hydra::detail::get_zip_iterator(this->fWeights.begin(), begins );
	fEnd = hydra::detail::get_zip_iterator(this->fWeights.end(), ends );

	fConstBegin = hydra::detail::get_zip_iterator(this->fWeights.cbegin(), cbegins );
	fConstEnd = hydra::detail::get_zip_iterator(this->fWeights.cend(), cends );
}

template<size_t N, unsigned int BACKEND>
Events<N,BACKEND>& Events<N, BACKEND>::operator=(hydra::experimental::Events<N,BACKEND> const& other)
{

	if(this==&other) return *this;

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

template<size_t N, unsigned int BACKEND>
template<unsigned int BACKEND2>
Events<N,BACKEND>& Events<N, BACKEND>::operator=(hydra::experimental::Events<N,BACKEND2> const& other)
		{

	if(this==&other) return *this;

			this->fNEvents=other.GetNEvents();
			this->fMaxWeight=other.GetMaxWeight();

#pragma unroll N
			for (GInt_t d = 0; d < N; d++)
			{
				this->fDaughters[d].resize(fNEvents);
			}

			this->fWeights.resize(fNEvents);
			this->fFlags.resize(fNEvents);


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
				begins[i]= this->fDaughters[i].begin();
				ends[i]= this->fDaughters[i].end();

				cbegins[i]= this->fDaughters[i].cbegin();
				cends[i]= this->fDaughters[i].cend();


			}
			this->fBegin = hydra::detail::get_zip_iterator(this->fWeights.begin(), begins );
			this->fEnd = hydra::detail::get_zip_iterator(this->fWeights.end(), ends );

			this->fConstBegin = hydra::detail::get_zip_iterator(this->fWeights.cbegin(), cbegins );
			this->fConstEnd = hydra::detail::get_zip_iterator(this->fWeights.cend(), cends );


			return *this;
		}

template<size_t N, unsigned int BACKEND>
Events<N,BACKEND>& Events<N, BACKEND>::operator=(hydra::experimental::Events<N,BACKEND> && other)
{

	if(this==&other) return *this;

	this->fNEvents=other.GetNEvents();
	this->fMaxWeight=other.GetMaxWeight();

	this->fWeights   = std::move(other.MoveWeights());
	this->fFlags     = std::move(other.MoveFlags());

#pragma unroll N
	for(int i =0; i < N; i++){
	this->fDaughters[i] = std::move(other.MoveDaughters()[i]);
	}
	other= Events<N, BACKEND>();

	std::array< vector_particles_iterator,N> begins;
	std::array< vector_particles_iterator,N> ends;

	std::array< vector_particles_const_iterator,N> cbegins;
	std::array< vector_particles_const_iterator,N> cends;

#pragma unroll N
	for(int i =0; i < N; i++){
		begins[i]= this->fDaughters[i].begin();
		ends[i]= this->fDaughters[i].end();

		cbegins[i]= this->fDaughters[i].cbegin();
		cends[i]= this->fDaughters[i].cend();


	}
	this->fBegin = hydra::detail::get_zip_iterator(this->fWeights.begin(), begins );
	this->fEnd = hydra::detail::get_zip_iterator(this->fWeights.end(), ends );

	this->fConstBegin = hydra::detail::get_zip_iterator(this->fWeights.cbegin(), cbegins );
	this->fConstEnd = hydra::detail::get_zip_iterator(this->fWeights.cend(), cends );


	return *this;
}

template<size_t N, unsigned int BACKEND>
size_t Events<N, BACKEND>::Unweight(size_t seed)
{

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

template<size_t N, unsigned int BACKEND>
void Events<N, BACKEND>::resize(size_t n){
#pragma unroll N
		for (GInt_t d = 0; d < N; d++){
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
			begins[i]= this->fDaughters[i].begin();
			ends[i]= this->fDaughters[i].end();

			cbegins[i]= this->fDaughters[i].cbegin();
			cends[i]= this->fDaughters[i].cend();
		}

		fBegin = hydra::detail::get_zip_iterator(this->fWeights.begin(), begins );
		fEnd = hydra::detail::get_zip_iterator(this->fWeights.end(), ends );

		fConstBegin = hydra::detail::get_zip_iterator(this->fWeights.cbegin(), cbegins );
		fConstEnd = hydra::detail::get_zip_iterator(this->fWeights.cend(), cends );

	}

}  // namespace experimental

}  // namespace hydra



#endif /* EVENTS_INL_ */
