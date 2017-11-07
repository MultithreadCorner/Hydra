/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
//#include <omp.h>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <utility>
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/FlagAcceptReject.h>
#include <hydra/multivector.h>

namespace hydra {


template<size_t N, hydra::detail::Backend BACKEND>
Events<N, hydra::detail::BackendPolicy<BACKEND>>::Events(size_t nevents):
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

template<size_t N, hydra::detail::Backend  BACKEND>
template<hydra::detail::Backend  BACKEND2>
Events<N, hydra::detail::BackendPolicy<BACKEND>>::Events(Events<N,hydra::detail::BackendPolicy<BACKEND2>> const& other):
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
		HYDRA_EXTERNAL_NS::thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
	}

	HYDRA_EXTERNAL_NS::thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
	HYDRA_EXTERNAL_NS::thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

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

template<size_t N, hydra::detail::Backend BACKEND>
Events<N, hydra::detail::BackendPolicy<BACKEND>>::Events(Events<N,hydra::detail::BackendPolicy<BACKEND>> const& other):
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
		HYDRA_EXTERNAL_NS::thrust::copy(other.DaughtersBegin(i),
				other.DaughtersEnd(i), this->DaughtersBegin(i));
	}

	HYDRA_EXTERNAL_NS::thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
	HYDRA_EXTERNAL_NS::thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

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

template<size_t N, hydra::detail::Backend BACKEND>
Events<N, hydra::detail::BackendPolicy<BACKEND>>::Events(Events<N,hydra::detail::BackendPolicy<BACKEND>> && other):
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

	other= Events<N,hydra::detail::BackendPolicy<BACKEND>>(0);

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

template<size_t N, hydra::detail::Backend  BACKEND>
Events<N,hydra::detail::BackendPolicy<BACKEND>>&
Events<N, hydra::detail::BackendPolicy<BACKEND>>::operator=(Events<N,hydra::detail::BackendPolicy<BACKEND>> const& other)
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

	if(fNEvents==0) return *this;

	for (GInt_t i = 0; i < N; i++)
	{
		HYDRA_EXTERNAL_NS::thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
	}

	HYDRA_EXTERNAL_NS::thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
	HYDRA_EXTERNAL_NS::thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

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

template<size_t N, hydra::detail::Backend  BACKEND>
template<hydra::detail::Backend  BACKEND2>
Events<N,hydra::detail::BackendPolicy<BACKEND>>&
Events<N, hydra::detail::BackendPolicy<BACKEND>>::operator=(Events<N,hydra::detail::BackendPolicy<BACKEND2>> const& other)
		{

	//if(this==&other) return *this;

			this->fNEvents=other.GetNEvents();
			this->fMaxWeight=other.GetMaxWeight();

#pragma unroll N
			for (GInt_t d = 0; d < N; d++)
			{
				this->fDaughters[d].resize(fNEvents);
			}

			this->fWeights.resize(fNEvents);
			this->fFlags.resize(fNEvents);

			if(fNEvents==0) return *this;

			for (GInt_t i = 0; i < N; i++)
			{
				HYDRA_EXTERNAL_NS::thrust::copy(other.DaughtersBegin(i), other.DaughtersEnd(i), this->DaughtersBegin(i));
			}

			HYDRA_EXTERNAL_NS::thrust::copy(other.WeightsBegin(), other.WeightsEnd(), this->WeightsBegin());
			HYDRA_EXTERNAL_NS::thrust::copy(other.FlagsBegin(), other.FlagsEnd(), this->FlagsBegin());

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

template<size_t N, hydra::detail::Backend  BACKEND>
Events<N,hydra::detail::BackendPolicy<BACKEND>>&
Events<N, hydra::detail::BackendPolicy<BACKEND>>::operator=(Events<N,hydra::detail::BackendPolicy<BACKEND>> && other)
{

	if(this==&other || fNEvents==0 ) return *this;

	this->fNEvents=other.GetNEvents();
	this->fMaxWeight=other.GetMaxWeight();

	this->fWeights   = std::move(other.MoveWeights());
	this->fFlags     = std::move(other.MoveFlags());

#pragma unroll N
	for(int i =0; i < N; i++){
	this->fDaughters[i] = std::move(other.MoveDaughters()[i]);
	}
	other= Events<N, hydra::detail::BackendPolicy<BACKEND>>();

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

template<size_t N, hydra::detail::Backend  BACKEND>
size_t Events<N, hydra::detail::BackendPolicy<BACKEND>>::Unweight(size_t seed)
{

	GULong_t count = 0;
	if(N==2)
	{
		HYDRA_EXTERNAL_NS::thrust::fill(fFlags.begin(), fFlags.end(), kTrue);
		count = fNEvents;
	}
	else
	{

		auto w = HYDRA_EXTERNAL_NS::thrust::max_element(fWeights.begin(),fWeights.end());
		fMaxWeight=*w;
		// create iterators
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + fNEvents;


		HYDRA_EXTERNAL_NS::thrust::transform(first, last, fWeights.begin(),
				fFlags.begin(), hydra::detail::FlagAcceptReject(seed, fMaxWeight));

		count = HYDRA_EXTERNAL_NS::thrust::count(fFlags.begin(), fFlags.end(),
				kTrue);

	}

	return count;

}

template<size_t N, hydra::detail::Backend  BACKEND>
void Events<N, hydra::detail::BackendPolicy<BACKEND>>::resize(size_t n){

	fNEvents=n;

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


}  // namespace hydra



#endif /* EVENTS_INL_ */
