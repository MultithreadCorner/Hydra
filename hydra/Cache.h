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
 * Cache.h
 *
 *  Created on: 13/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CACHE_H_
#define CACHE_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/multiarray.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Tuple.h>

//thrust
#include <thrust/distance.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
//std
#include <utility>

namespace hydra {

namespace detail {

template< typename ...Functors>
struct CacheEvaluator {

	typedef detail::tuple_type<N, GReal_t> return_type;

	 CacheEvaluator(Functors... functors):
	   fFunctors(hydra::make_tuple(functors...))
	 { }

	 __host__ __device__
	 CacheEvaluator(CacheEvaluator<Functors...> const& other):
	   fFunctors(other->fFunctors)
	 { }

	 __host__ __device__
	 CacheEvaluator<Functors...>&
	 operator=(CacheEvaluator<Functors...> const& other)
	 {
		 if(this==&other) return *this;
		 fFunctors= other.fFunctors;
		 return *this;
	 }

	template<typename T>
	__host__ __device__
	return_type operator()(T& value){

		return detail::invoke(value, fFunctors);
	}

	hydra::tuple<Functors...> fFunctors;
};

}  // namespace detail

template< typename Backend, typename ...Functors>
class Cache;


template< hydra::detail::Backend BACKEND, typename ...Functors>
class Cache< hydra::detail::BackendPolicy<BACKEND>, Functors...>{
public:


	typedef multiarray<sizeof...(Functors),GReal_t, hydra::detail::BackendPolicy<BACKEND> > storage_type;
	typedef typename storage_type::iterator iterator;
	typedef typename storage_type::const_iterator const_iterator;

	Cache()=delete;

	template< typename Iterator>
	Cache(Iterator first, Iterator last, Functors&... functors)
	{
		fData.resize(thrust::distance(first, last));
		thrust::transform( first, last,
				fData.begin(), detail::CacheEvaluator(functors...) );
	}

	Cache(Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...> const& other):
		fData(other.GetData())
	    {}

	Cache(Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...> && other):
			fData(other.MoveData())
		{}

	template< hydra::detail::Backend BACKEND2>
	Cache(Cache<hydra::detail::BackendPolicy<BACKEND2>,Functors...> const& other)
	{
		fData.resize(thrust::distance(other.begin(), other.end()));
		thrust::copy(other.begin(), other.end(), this->begin());
	}

	Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...>&
	operator=(Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...> const& other){

		if(this==&other) return *this;
		fData =other.GetData();
		return *this;
	}

	Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...>&
	operator=(Cache< hydra::detail::BackendPolicy<BACKEND>,Functors...> && other){

		if(this==&other) return *this;
		fData =other.GetData();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	Cache< hydra::detail::BackendPolicy<BACKEND>,Functors...>&
	operator=(Cache<hydra::detail::BackendPolicy<BACKEND2>,Functors...> const& other)
	{
		if(this==&other) return *this;
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

private:

	const storage_type& GetData() const {
		return fData;
	}

	storage_type MoveData(){
			return std::move(fData);
	}

	storage_type fData;
};


}  // namespace hydra



#endif /* CACHE_H_ */
