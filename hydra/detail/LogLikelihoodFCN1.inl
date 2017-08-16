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
 * LogLikelihoodFCN1.inl
 *
 *  Created on: 14/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN1_INL_
#define LOGLIKELIHOODFCN1_INL_

#include <hydra/FCN2.h>

#include <hydra/detail/functors/LogLikelihood1.h>
#include <thrust/transform_reduce.h>

namespace hydra {

template<template<typename F, typename I> class PDF, typename Functor, typename Integrator, typename Iterator>
class LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>: public FCN2<LogLikelihoodFCN1<PDF<Functor,Integrator>,Iterator > >{

public:


	/**
	 * @brief LogLikelihoodFCN constructor for non-cached models.
	 *
	 * @param functor hydra::PDF instance.
	 * @param begin  iterator pointing to the begin of the dataset.
	 * @param end   iterator pointing to the end of the dataset.
	 */
	LogLikelihoodFCN1(PDF<Functor,Integrator>& functor, Iterator begin, Iterator end):
		FCN2<LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>>(functor,begin, end)
		{}

	LogLikelihoodFCN1(LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>const& other):
		FCN2<LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>>(other)
		{}

	LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>&
	operator=(LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>const& other)
	{
		if(this==&other) return  *this;
		FCN2<LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator>>::operator=(other);

		return  *this;
	}


	GReal_t Eval( const std::vector<double>& parameters ) const{

		using   thrust::system::detail::generic::select_system;
		typedef typename thrust::iterator_system<Iterator>::type System;
		typedef typename PDF<Functor,Integrator>::functor_type functor_type;
		System system;

		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + this->GetDataSize();

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

		this->GetPDF().SetParameters(parameters);
		//this->GetPDF().PrintRegisteredParameters();

		auto NLL = detail::LogLikelihood1<functor_type>(this->GetPDF().GetFunctor());

		final = thrust::transform_reduce(select_system(system),
				this->begin(), this->end(), NLL, init, thrust::plus<GReal_t>());

		return (GReal_t)this->GetDataSize()-final ;
	}

};

template<template<typename F, typename I> class PDF, typename Functor, typename Integrator,  typename Iterator>
auto make_loglikehood_fcn(Iterator first, Iterator last, PDF<Functor,Integrator>& functor)
-> LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator >
{
	return LogLikelihoodFCN1<PDF<Functor,Integrator>, Iterator >( functor, first, last);
}

}  // namespace hydra





#endif /* LOGLIKELIHOODFCN1_INL_ */
