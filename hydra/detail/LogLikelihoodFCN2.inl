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
 * LogLikelihoodFCN2.inl
 *
 *  Created on: Aug 16, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN2_INL_
#define LOGLIKELIHOODFCN2_INL_


#include <hydra/FCN.h>
#include <hydra/PDFSumExtendable.h>
#include <hydra/detail/functors/LogLikelihood1.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/inner_product.h>

namespace hydra {

template<typename ...Pdfs, typename Iterator >
class LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator>: public FCN<LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator > >{

public:


	/**
	 * @brief LogLikelihoodFCN constructor for non-cached models.
	 *
	 * @param functor hydra::PDF instance.
	 * @param begin  iterator pointing to the begin of the dataset.
	 * @param end   iterator pointing to the end of the dataset.
	 */
	LogLikelihoodFCN(PDFSumExtendable<Pdfs...>& functor, Iterator begin, Iterator end):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>>(functor,begin, end)
		{}

	LogLikelihoodFCN(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>const& other):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>>(other)
		{}

	LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>&
	operator=(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>const& other)
	{
		if(this==&other) return  *this;
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, Iterator>>::operator=(other);
		return  *this;
	}


	GReal_t Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type System;
		typedef typename PDFSumExtendable<Pdfs...>::functor_type functor_type;
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

		this->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood1<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::transform_reduce(select_system(system), this->begin(), this->end(),
				NLL, init, HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>());

		GReal_t  r = (GReal_t)this->GetDataSize() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -	this->GetDataSize()*log(this->GetPDF().GetCoefSum() ) ) - final;



		return r;

	}


};


template<typename ...Pdfs, typename IteratorD , typename ...IteratorW>
class LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>: public FCN<LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW ...> >
{

public:


	/**
	 * @brief LogLikelihoodFCN constructor for non-cached models.
	 *
	 * @param functor hydra::PDF instance.
	 * @param begin  iterator pointing to the begin of the dataset.
	 * @param end   iterator pointing to the end of the dataset.
	 */
	LogLikelihoodFCN(PDFSumExtendable<Pdfs...>& functor, IteratorD begin, IteratorD end, IteratorW ...wbegin):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>>(functor,begin, end, wbegin...)
		{}

	LogLikelihoodFCN(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>const& other):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>>(other)
		{}

	LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>&
	operator=(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>const& other)
	{
		if(this==&other) return  *this;
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>>::operator=(other);
		return  *this;
	}


	GReal_t Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<typename FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>>::iterator>::type System;
		typedef typename PDFSumExtendable<Pdfs...>::functor_type functor_type;
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

		this->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood2<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::inner_product(select_system(system), this->begin(), this->end(),this->wbegin(),
				 init,HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>(),NLL );

		GReal_t  r = (GReal_t)this->GetDataSize() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -	this->GetDataSize()*log(this->GetPDF().GetCoefSum() ) ) - final;



		return r;

	}


};



template<typename... Pdfs,  typename Iterator, typename ...Iterators>
auto make_loglikehood_fcn(PDFSumExtendable<Pdfs...>& functor, Iterator first, Iterator last, Iterators... weights )
-> LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator,Iterators... >
{
	return LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator, Iterators...>( functor, first, last, weights...);
}


}  // namespace hydra






#endif /* LOGLIKELIHOODFCN2_INL_ */
