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
#include <hydra/multivector.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Tuple.h>

//thrust
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/tuple.h>
//std
#include <utility>

namespace hydra {

namespace detail {

template< typename ...Functors>
struct CacheEvaluator {

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> return_type;

	 CacheEvaluator(Functors... functors):
	   fFunctors(hydra::make_tuple(functors...))
	 { }

	 __hydra_host__ __hydra_device__
	 CacheEvaluator(CacheEvaluator<Functors...> const& other):
	   fFunctors(other.fFunctors)
	 { }

	 __hydra_host__ __hydra_device__
	 CacheEvaluator<Functors...>&
	 operator=(CacheEvaluator<Functors...> const& other)
	 {
		 if(this==&other) return *this;
		 fFunctors= other.fFunctors;
		 return *this;
	 }

	template<typename T>
	__hydra_host__ __hydra_device__
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

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> tuple_type;

public:

	typedef multivector< tuple_type, hydra::detail::BackendPolicy<BACKEND> > storage_type;
	typedef typename storage_type::iterator iterator;
	typedef typename storage_type::const_iterator const_iterator;
	typedef typename storage_type::value_type value_type;

	Cache()=delete;

	template< typename Iterator>
	Cache(Iterator first, Iterator last, Functors&... functors)
	{
		SetCacheIndex(functors...);

		fData.resize(HYDRA_EXTERNAL_NS::thrust::distance(first, last));
		HYDRA_EXTERNAL_NS::thrust::transform( first, last,
				fData.begin(), detail::CacheEvaluator<Functors...>(functors...) );
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
		fData.resize(HYDRA_EXTERNAL_NS::thrust::distance(other.begin(), other.end()));
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), this->begin());
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
		fData =other.MoveData();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	Cache< hydra::detail::BackendPolicy<BACKEND>,Functors...>&
	operator=(Cache<hydra::detail::BackendPolicy<BACKEND2>,Functors...> const& other)
	{
		if(this==&other) return *this;
		fData.resize(HYDRA_EXTERNAL_NS::thrust::distance(other.begin(), other.end()));
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), this->begin());
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

	value_type& operator[](size_t i){
		return  fData[i];
	}

	const value_type& operator[](size_t i) const {
			return  fData[i];
	}


private:

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == sizeof...(Functors)), void>::type
	SetCacheIndexHelper(HYDRA_EXTERNAL_NS::thrust::tuple<Functors&...>){ }

	template<size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(Functors)), void>::type
	SetCacheIndexHelper(HYDRA_EXTERNAL_NS::thrust::tuple<Functors&...> functors){

		HYDRA_EXTERNAL_NS::thrust::get<I>(functors).SetCacheIndex(I);

		SetCacheIndexHelper<I+1>(functors);
	}


	void SetCacheIndex(Functors& ...functors){

		SetCacheIndexHelper(HYDRA_EXTERNAL_NS::thrust::tie(functors...));
	}

	const storage_type& GetData() const {
		return fData;
	}

	storage_type MoveData(){
			return std::move(fData);
	}

	storage_type fData;
};


template< hydra::detail::Backend BACKEND, typename Iterator, typename ...Functors>
auto make_cache(hydra::detail::BackendPolicy<BACKEND>, Iterator first, Iterator last, Functors&... functors)
->Cache<hydra::detail::BackendPolicy<BACKEND>, Functors...>
{

	return Cache<hydra::detail::BackendPolicy<BACKEND>,Functors...>( first, last, functors...);
}


}  // namespace hydra



#endif /* CACHE_H_ */
