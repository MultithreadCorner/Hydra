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
 * Weights.h
 *
 *  Created on: 11/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WEIGHTS_H_
#define WEIGHTS_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
//thrust
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>

namespace hydra {

template<typename Backend>
class Weights;

template<hydra::detail::Backend BACKEND >
class Weights<hydra::detail::BackendPolicy<BACKEND>>{

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t> storage_type;
	typedef typename storage_type::iterator iterator;
	typedef typename storage_type::const_iterator const_iterator;

public:

	  Weights():
		fSumW(0),
		fSumW2(0),
		fData()
       {}

	  Weights(size_t n):
	 		fSumW(0),
	 		fSumW2(0),
	 		fData(n)
	        {}

	  Weights(size_t n, GReal_t value):
		  fSumW(0),
		  fSumW2(0),
		  fData(n, value)
	  {}

	template<typename Iterator>
	Weights(Iterator first, Iterator last):
		fSumW(0),
		fSumW2(0)
	{
		fData.resize(thrust::distance( first, last ));
		thrust::copy(first, last, this->begin());
		fSumW  =  MakeSumW();
		fSumW2 =  MakeSumW2();
	}

	Weights( Weights<hydra::detail::BackendPolicy<BACKEND>> const& other):
		fSumW(other.GetSumW()),
		fSumW2(other.GetSumW2()),
		fData(other.GetData())
	{}

	Weights( Weights<hydra::detail::BackendPolicy<BACKEND>>&& other):
		fSumW(other.GetSumW()),
		fSumW2(other.GetSumW2()),
		fData(other.MoveData())
	{}

	template<hydra::detail::Backend BACKEND2 >
	Weights( Weights<hydra::detail::BackendPolicy<BACKEND2>> const& other):
	fSumW(other.GetSumW()),
	fSumW2(other.GetSumW2())
	{
		fData.resize(thrust::distance(other.begin(), other.end()));
		thrust::copy(other.begin(), other.end(), this->begin());
	}

	Weights<hydra::detail::BackendPolicy<BACKEND>>&
	operator=( Weights< hydra::detail::BackendPolicy<BACKEND>> const& other){

		if(this==&other)return *this;
		fSumW  = other.GetSumW();
		fSumW2 = other.GetSumW2();
		fSumW2 = other.GetData();
		return *this;
	}
	
	Weights<hydra::detail::BackendPolicy<BACKEND>>&
	operator=( Weights< hydra::detail::BackendPolicy<BACKEND>>&& other){

		if(this==&other)return *this;
		fSumW  = other.GetSumW();
		fSumW2 = other.GetSumW2();
		fSumW2 = other.MoveData();
		return *this;
	}
	
	template<hydra::detail::Backend BACKEND2 >
	Weights<hydra::detail::BackendPolicy<BACKEND>>&
	operator=( Weights< hydra::detail::BackendPolicy<BACKEND2>> const& other){

		if(this==&other)return *this;
		fSumW  = other.GetSumW();
		fSumW2 = other.GetSumW2();
		fData.resize(thrust::distance(other.begin(), other.end()));
		thrust::copy(other.begin(), other.end(), this->begin());
		return *this;
	}


	iterator begin(){
		return fData.begin();
	}

	iterator end(){
		return fData.end();
	}

	const_iterator begin() const{
		return fData.cbegin();
	}

	const_iterator end() const{
		return fData.cend();
	}

	GReal_t GetSumW() const {
		return fSumW;
	}

	void SetSumW(GReal_t sumW) {
		fSumW = sumW;
	}

	GReal_t GetSumW2() const {
		return fSumW2;
	}

	void SetSumW2(GReal_t sumW2) {
		fSumW2 = sumW2;
	}

private:
	
	GReal_t MakeSumW()
	{
		return thrust::reduce(fData.begin(), fData.end(), 0.0);
	}

	GReal_t MakeSumW2()
	{
		auto sq = []__host__ __device__ (GReal_t x)
		{
			return x*x;
		};

		return thrust::reduce(thrust::make_transform_iterator(fData.begin(), sq),
				              thrust::make_transform_iterator(fData.end()  , sq), 0.0);
	}

	const storage_type& GetData() const {
		return fData;
	}

	storage_type MoveData(){
		return std::move(fData);
	}

	GReal_t  fSumW;
	GReal_t  fSumW2;
	vector_t fData;


};


}  // namespace hydra



#endif /* WEIGHTS_H_ */
