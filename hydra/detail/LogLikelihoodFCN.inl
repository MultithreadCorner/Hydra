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
 * LogLikelihoodFCN.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup Data Fitting
 */


#ifndef LOGLIKELIHOODFCN_INL_
#define LOGLIKELIHOODFCN_INL_

namespace hydra{


/*
template<typename FUNCTOR,  typename IteratorData, typename IteratorCache >
typename thrust::detail::enable_if< detail::is_hydra_functor<FUNCTOR>::value, GReal_t>::type
LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>::Eval( const std::vector<double>& parameters ) const
{
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


		final=thrust::transform_reduce(first, last,
				LogLikelihood<FUNCTOR, IteratorData, IteratorCache>(fFunctor, fFunctor.GetNorm(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + final ;
	}

template<typename FUNCTOR,  typename IteratorData, typename IteratorCache >
typename thrust::detail::enable_if< detail::is_hydra_pdf<FUNCTOR>::value, GReal_t>::type
LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>::Eval( const std::vector<double>& parameters ) const
{
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


		final=thrust::transform_reduce(first, last,
				LogLikelihood<FUNCTOR, IteratorData, IteratorCache>(fFunctor, fFunctor.GetNorm(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + final ;
	}

template<typename FUNCTOR,  typename IteratorData, typename IteratorCache >
typename thrust::detail::enable_if< detail::is_hydra_sum_pdf<FUNCTOR>::value, GReal_t>::type
LogLikelihoodFCN<FUNCTOR,IteratorData, IteratorCache>::Eval( const std::vector<double>& parameters ) const
{
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


		final=thrust::transform_reduce(first, last,
				LogLikelihood<FUNCTOR, IteratorData, IteratorCache>(fFunctor, fFunctor.GetNorm(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + fFunctor.IsExtended()*fFunctor. GetCoefSum() + final ;
	}

*/

} // namespace hydra
#endif /* LOGLIKELIHOODFCN_INL_ */
