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
 * LogLikelihood.h
 *
 *  Created on: 06/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup fit
 */


#ifndef _LOGLIKELIHOOD_H_
#define _LOGLIKELIHOOD_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/TypeTraits.h>
//#include <hydra/Point.h>
//#include <hydra/PointVector.h>

#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/functional.h>


#if(0)
namespace hydra{


namespace detail{


template<typename FUNCTOR, typename PointType, typename IteratorData, typename IteratorCache>
struct LogLikelihood
{
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<IteratorData>::value_type data_value_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<IteratorCache>::value_type cache_value_type;

	typedef PointType point_type;

	LogLikelihood(FUNCTOR const& functor,
			GReal_t sumW, GReal_t sumW2,
			IteratorData dbegin, IteratorCache cbegin,
		    GBool_t weighted, GBool_t cached):
		fDataBegin(dbegin),
		fCacheBegin(cbegin),
		fFunctor(functor),
		fSumW(sumW),
		fSumW2(sumW2),
		fWeighted(weighted),
		fCached(cached)
	{}

	__hydra_host__ __hydra_device__ inline
	LogLikelihood( LogLikelihood<FUNCTOR, PointType,IteratorData, IteratorCache> const& other):
	  fDataBegin(other.fDataBegin),
	  fCacheBegin(other.fCacheBegin),
	  fFunctor(other.fFunctor),
	  fWeighted(other.fWeighted),
	  fCached(other.fCached),
	  fSumW(other.fSumW),
	  fSumW2(other.fSumW2)
	{}

	__hydra_exec_check_disable__
	__hydra_host__ __hydra_device__
	~LogLikelihood(){}

    template<typename U = cache_value_type >
	__hydra_host__ __hydra_device__ inline
	GReal_t operator()(size_t index, const typename std::enable_if< !std::is_same<U,
	       	null_type>::value, void >::type* dummy=0 ){

    	          auto      C = (cache_value_type) fCacheBegin[index];
    	          auto      X = ((point_type) fDataBegin[index]).GetCoordinates() ;
    	        GReal_t     W = ((point_type) fDataBegin[index]).GetWeight() ;


		return fCached? W*log(fFunctor( X, C )) :  W*log(fFunctor( X ));
	}

    template<typename U = cache_value_type >
   	__hydra_host__ __hydra_device__ inline
   	GReal_t operator()(size_t index, const typename std::enable_if< std::is_same<U,
   	       	null_type>::value, void >::type* dummy=0 ){


        auto X = ((point_type) fDataBegin[index]).GetCoordinates() ;
        GReal_t  W = ((point_type) fDataBegin[index]).GetWeight() ;

		return  W*log(fFunctor( X ));
	}



private:
	IteratorData  fDataBegin;
	IteratorCache  fCacheBegin;
	FUNCTOR fFunctor;
	GBool_t fWeighted;
	GBool_t fCached;
	GReal_t fSumW;
	GReal_t fSumW2;
};

}//namespace detail


}//namespace hydra
#endif

#endif /* LOGLIKELIHOOD_H_*/
