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

#ifndef _LOGLIKELIHOODFCN_H_
#define _LOGLIKELIHOODFCN_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/FCN.h>
#include <hydra/detail/functors/LogLikelihood.h>
#include <hydra/PointVector.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/Print.h>
#include <hydra/UserParameters.h>

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


template<typename PDF, typename PointType, typename IteratorData,typename IteratorCache >
class LogLikelihoodFCN: public FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>
{

public:


	LogLikelihoodFCN(PDF& functor, IteratorData begin, IteratorData end):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(functor,begin, end),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
		{}

	LogLikelihoodFCN(PDF& functor, IteratorData begin, IteratorData end, IteratorCache cend):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(functor,begin, end, cend),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
		{}

	LogLikelihoodFCN( LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache> const& other):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(other),
		fMAxValue(other.GetMAxValue())
		{}

	LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>&
	operator=(LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache> const& other)
	{
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>::operator=(other);
		this->fMAxValue=other.GetMAxValue();
		return *this;
	}


	template<typename U= PDF>
	typename thrust::detail::enable_if< hydra::detail::is_hydra_pdf<U>::value, GReal_t>::type
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
		if (1 )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		//std::cout << "Mean1 " << parameters[0] << std::endl;
		this->GetPDF().SetParameters(parameters);
		this->GetPDF().PrintRegisteredParameters();


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<typename U::functor_type, PointType,
				IteratorData, IteratorCache>( this->GetPDF().GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + final ;
	}

	template<typename U=PDF>
	typename thrust::detail::enable_if< hydra::detail::is_hydra_sum_pdf<U>::value, GReal_t>::type
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

		this->GetPDF().SetParameters(parameters);

		/*
		 * avoid evaluate inconsistent coefficients configurations
		 * returning quickly the highest NLL value already calculated
		 */
		if( this->GetPDF().IsFractioned() &&
				( this->GetPDF().GetCoefSum() < 0.0 || this->GetPDF().GetCoefSum() > 0.0) )
			return fMAxValue;


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<typename U::functor_type, PointType,
				IteratorData, IteratorCache>( this->GetPDF().GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());

		GReal_t  r = this->GetSumW() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -
						this->GetSumW()*log(this->GetPDF().GetCoefSum() ) ) - final;

		fMAxValue = fMAxValue<r?r:fMAxValue;
	//	std::cout << std::setprecision(16)<<r << std::endl;

		return r;
	}



	GReal_t GetMAxValue() const {
		return fMAxValue;
	}

private:

	mutable GReal_t fMAxValue;


};

//conveniency function
template<typename PDF, typename PointType, unsigned int BACKEND>
auto make_loglikehood_fcn(PDF& functor,
		PointVector< PointType, BACKEND> const& data)
-> LogLikelihoodFCN<PDF, PointType,
typename PointVector< PointType, BACKEND>::const_iterator,
thrust::constant_iterator<null_type> >
{
	return LogLikelihoodFCN<PDF, PointType,
			typename PointVector< PointType, BACKEND>::const_iterator,
			thrust::constant_iterator<null_type>>(functor,  data.cbegin(),  data.cend());
}

template<typename PDF,  typename PointType,	template<typename...> class Vector,
template<typename...> class Allocator, typename Tuple,  unsigned int BACKEND>
auto make_loglikehood_fcn(PDF& functor,
		PointVector< PointType, BACKEND> const& data,
		multivector<Vector<Tuple, Allocator< Tuple >>> const& cache )
-> LogLikelihoodFCN<PDF,  PointType, typename PointVector< PointType, BACKEND>::iterator,
typename multivector<Vector<Tuple, Allocator< Tuple >>>::iterator >
{
	return LogLikelihoodFCN<PDF, PointType,
			typename PointVector<PointType,BACKEND>::iterator,
			typename multivector<Vector<Tuple, Allocator< Tuple >>>::iterator >( functor, data.cbegin(), data.cend(), cache.cbegin());
}


}//namespace hydra


#endif /* LOGLIKELIHOODFCN_H_ */
