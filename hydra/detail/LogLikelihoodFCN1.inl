/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * LogLikelihoodFCN1.inl
 *
 *  Created on: 14/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN1_INL_
#define LOGLIKELIHOODFCN1_INL_

#include <hydra/FCN.h>
#include <hydra/Pdf.h>
#include <hydra/detail/functors/LogLikelihood1.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/inner_product.h>

namespace hydra {


template<typename Functor, typename Integrator, typename IteratorD, typename ...IteratorW>
class LogLikelihoodFCN< Pdf<Functor,Integrator> , IteratorD, IteratorW...>: public FCN<LogLikelihoodFCN< Pdf<Functor,Integrator>,IteratorD, IteratorW... > >{

public:


	/**
	 * @brief LogLikelihoodFCN constructor for non-cached models.
	 *
	 * @param functor hydra::PDF instance.
	 * @param begin  IteratorD pointing to the begin of the dataset.
	 * @param end   IteratorD pointing to the end of the dataset.
	 */
	LogLikelihoodFCN(Pdf<Functor,Integrator> const& functor, IteratorD begin, IteratorD end, IteratorW ...wbegin):
		FCN<LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>>(functor,begin, end, wbegin...)
		{}

	LogLikelihoodFCN(LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>const& other):
		FCN<LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>>(other)
		{}

	LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD>&
	operator=(LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>const& other)
	{
		if(this==&other) return  *this;
		FCN<LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>>::operator=(other);

		return  *this;
	}

	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M==0), double >::type
	Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<IteratorD>::type System;
		typedef typename Pdf<Functor,Integrator>::functor_type functor_type;
		System system;

		// create iterators
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + this->GetDataSize();

		GReal_t final;
		GReal_t init=0;

		if (INFO >= Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		const_cast< LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>* >(this)->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood1<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::transform_reduce(select_system(system),
				this->begin(), this->end(), NLL, init, HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>());

		return (GReal_t)this->GetDataSize() -final ;
	}

	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M>0), double >::type
	Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<typename FCN<LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>>::iterator>::type System;
		typedef typename Pdf<Functor,Integrator>::functor_type functor_type;
		System system;

		// create iterators
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + this->GetDataSize();

		GReal_t final;
		GReal_t init=0;

		if (INFO >= Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		const_cast< LogLikelihoodFCN< Pdf<Functor,Integrator>, IteratorD, IteratorW...>* >(this)->GetPDF().SetParameters(parameters);


		auto NLL = detail::LogLikelihood2<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::inner_product(select_system(system), this->begin(), this->end(),this->wbegin(),
				init,HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>(),NLL );

		return (GReal_t)this->GetDataSize() -final ;
	}

};

template< typename Functor, typename Integrator,  typename Iterator, typename ...Iterators>
auto make_loglikehood_fcn(Pdf<Functor,Integrator> const& pdf, Iterator first, Iterator last,  Iterators... weights )
-> LogLikelihoodFCN< Pdf<Functor,Integrator>, Iterator , Iterators... >
{
	return LogLikelihoodFCN< Pdf<Functor,Integrator>, Iterator >(pdf, first, last, weights...);
}

}  // namespace hydra





#endif /* LOGLIKELIHOODFCN_INL_ */
