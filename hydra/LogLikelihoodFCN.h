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
 * LogLikelihoodFCN.h
 *
 *  Created on: 03/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup fit
 */

#ifndef LOGLIKELIHOODFCN_H_
#define LOGLIKELIHOODFCN_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/FCN.h>
#include <hydra/detail/functors/LogLikelihood.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/Print.h>

#include <thrust/distance.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

#include <Minuit2/FCNBase.h>
#include <vector>
#include <unordered_map>
#include <cassert>


namespace hydra{


template<typename FUNCTOR, typename IteratorData,typename IteratorCache >
class LogLikelihoodFCN:public FCN<LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>,IteratorData, IteratorCache>
{

public:


	LogLikelihoodFCN(FUNCTOR& functor, IteratorData begin, IteratorData end):
		FCN<LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>,IteratorData, IteratorCache>(begin, end),
		fFunctor(functor),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
	{}

	LogLikelihoodFCN(FUNCTOR& functor, IteratorData begin, IteratorData end, IteratorCache cend):
		FCN<LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>,IteratorData, IteratorCache>(begin, end, cend),
		fFunctor(functor),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
		{}

	LogLikelihoodFCN( LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache> const& other):
		FCN<LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>,IteratorData, IteratorCache>(other),
		fFunctor(other.GetFunctor()),
		fMAxValue(other.GetMAxValue())
		{}

	LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>&
	operator=(LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache> const& other)
	{
		FCN<LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>,IteratorData, IteratorCache>::operator=(other);
		this->fFunctor=other.GetFunctor();
		this->fMAxValue=other.GetMAxValue();
		return *this;
	}


	//GReal_t Eval( const std::vector<double>& parameters ) const;


	template<typename U= FUNCTOR>
	typename thrust::detail::enable_if< detail::is_hydra_pdf<U>::value, GReal_t>::type
	Eval( const std::vector<double>& parameters ) const
	{
		using thrust::system::detail::generic::select_system;
		typedef typename thrust::iterator_system<IteratorData>::type System;
		System system;

		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + this->GetNEvents();

		GReal_t final;
		GReal_t init=0;

		//set functor with the new parameters
		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		//std::cout << "Mean1 " << parameters[0] << std::endl;
		fFunctor.SetParameters(parameters);
		//fFunctor.PrintRegisteredParameters();


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<U, IteratorData, IteratorCache>(fFunctor.GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + final ;
	}

	template<typename U=FUNCTOR>
	typename thrust::detail::enable_if< detail::is_hydra_sum_pdf<U>::value, GReal_t>::type
	Eval( const std::vector<double>& parameters ) const
	{

		using thrust::system::detail::generic::select_system;
		typedef typename thrust::iterator_system<IteratorData>::type System;
		System system;

		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + this->GetNEvents();

		GReal_t final;
		GReal_t init=0;

		//set functor with the new parameters
		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		//std::cout << "Mean1 " << parameters[0] << std::endl;
		fFunctor.SetParameters(parameters);
		//fFunctor.PrintRegisteredParameters();

		/*
		 * avoid evaluate inconsistent coefficients configurations
		 * returning quickly the highest NLL value already calculated
		 */
		if( fFunctor.IsFractioned() && (fFunctor.GetCoefSum() < 0.0 || fFunctor.GetCoefSum() > 0.0))
			return fMAxValue;


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<U, IteratorData, IteratorCache>(fFunctor.GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());

		GReal_t  r= this->GetSumW() + fFunctor.IsExtended()*(fFunctor.GetCoefSum() -this->GetSumW()*log(fFunctor.GetCoefSum())) - final;

		fMAxValue = fMAxValue<r?r:fMAxValue;


		return r ;
	}



	FUNCTOR& GetFunctor() const{
		return fFunctor;
	}

	GReal_t GetMAxValue() const {
		return fMAxValue;
	}

private:
	FUNCTOR& fFunctor;
	mutable GReal_t fMAxValue;


};

//conveniency function
template<typename FUNCTOR, typename IteratorData >
LogLikelihoodFCN<FUNCTOR, IteratorData, thrust::constant_iterator<null_type>>
make_loglikehood_fcn(FUNCTOR& functor, IteratorData begin, IteratorData end)
{
	return LogLikelihoodFCN<FUNCTOR, IteratorData, thrust::constant_iterator<null_type>>(functor,  begin, end);
}

template<typename FUNCTOR, typename IteratorData, typename IteratorCache>
LogLikelihoodFCN<FUNCTOR, IteratorData, IteratorCache>
make_loglikehood_fcn(FUNCTOR& functor,  IteratorData begin, IteratorData end, IteratorCache cache )
{
	return LogLikelihoodFCN<FUNCTOR, IteratorData, IteratorCache>(functor, begin, end, cache);
}

}//namespace hydra

#include <hydra/detail/LogLikelihoodFCN.inl>

#endif /* LOGLIKELIHOODFCN_H_ */
