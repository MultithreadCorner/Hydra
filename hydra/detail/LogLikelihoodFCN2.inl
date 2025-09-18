/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>
#include <hydra/detail/external/hydra_thrust/inner_product.h>

namespace hydra {

template<typename ...Pdfs, typename IteratorD , typename ...IteratorW>
class LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>: public FCN<LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW ...>, true >
{

public:

	typedef void likelihood_estimator_type;

	LogLikelihoodFCN()=delete;


	LogLikelihoodFCN(PDFSumExtendable<Pdfs...> const& functor, IteratorD begin, IteratorD end, IteratorW ...wbegin):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>, true>(functor,begin, end, wbegin...)
		{}

	LogLikelihoodFCN(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>const& other):
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>, true>(other)
		{}

	LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>&
	operator=(LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>const& other)
	{
		if(this==&other) return  *this;
		FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>, true>::operator=(other);
		return  *this;
	}

	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M==0), double >::type
	Eval( const std::vector<double>& parameters ) const{


		using   hydra::thrust::system::detail::generic::select_system;
		typedef typename hydra::thrust::iterator_system<IteratorD>::type System;
		typedef typename PDFSumExtendable<Pdfs...>::functor_type functor_type;
		System system;

		// create iterators
		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + this->GetDataSize();

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

		const_cast< LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>*  >(this)->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood1<functor_type>(this->GetPDF().GetFunctor());

		final = hydra::thrust::transform_reduce(select_system(system), this->begin(), this->end(),
				NLL, init, hydra::thrust::plus<GReal_t>());

		GReal_t  r = (GReal_t)this->GetDataSize() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -	this->GetDataSize()*::log(this->GetPDF().GetCoefSum() ) ) - final;

		return r;

	}

	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M>0), double >::type
	Eval( const std::vector<double>& parameters ) const{

		using   hydra::thrust::system::detail::generic::select_system;
		typedef typename hydra::thrust::iterator_system<typename FCN<LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>, true>::iterator>::type System;
		typedef typename PDFSumExtendable<Pdfs...>::functor_type functor_type;
		System system;

		// create iterators
		hydra::thrust::counting_iterator<size_t> first(0);
		hydra::thrust::counting_iterator<size_t> last = first + this->GetDataSize();

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

		const_cast< LogLikelihoodFCN<PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>*  >(this)->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood2<functor_type>(this->GetPDF().GetFunctor());

		final = hydra::thrust::inner_product(select_system(system), this->begin(), this->end(),this->wbegin(),
				 init,hydra::thrust::plus<GReal_t>(),NLL );

		GReal_t  r = (GReal_t)this->GetDataSize() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -	this->GetDataSize()*::log(this->GetPDF().GetCoefSum() ) ) - final;

		return r;

	}
};



template<typename... Pdfs,  typename Iterator, typename ...Iterators >
inline typename std::enable_if< hydra::detail::is_iterator<Iterator>::value  && detail::are_iterators<Iterators...>::value,
LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator,Iterators...  >>::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& functor, Iterator first, Iterator last, Iterators... weights )
{
	return LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator, Iterators...>( functor, first, last, weights...);
}


template<typename ...Pdfs, typename Iterable, typename ...Iterables >
inline typename std::enable_if<   (!detail::is_iterator<Iterable>::value) &&
                                  ((sizeof...(Iterables)==0) || !detail::are_iterators<Iterables...>::value) &&
                                  (!hydra::detail::is_hydra_dense_histogram< typename std::remove_reference<Iterable>::type>::value) && //is not dense/sparse histogram
		                          (!hydra::detail::is_hydra_sparse_histogram<typename std::remove_reference<Iterable>::type>::value) &&
								  detail::is_iterable<Iterable>::value && detail::are_iterables<Iterables...>::value  ,
LogLikelihoodFCN< PDFSumExtendable<Pdfs...>,
                  decltype(std::declval<Iterable>().begin() ),
                  decltype(std::declval<Iterables>().begin())... > >::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& functor, Iterable&& points, Iterables&&... weights ){

	return make_loglikehood_fcn( functor,
			std::forward<Iterable>(points).begin(),
			std::forward<Iterable>(points).end(),
			std::forward<Iterables>(weights).begin() ...);
}

template<typename ...Pdfs, typename Histogram>
inline typename std::enable_if<detail::is_hydra_dense_histogram<Histogram>::value ||
                               detail::is_hydra_sparse_histogram<Histogram>::value,
 LogLikelihoodFCN< PDFSumExtendable<Pdfs...>,
                     decltype(std::declval<const Histogram&>().GetBinsCenters().begin()),
                     decltype(std::declval<const Histogram&>().GetBinsContents().begin()) >>::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& functor, Histogram const&  points)
{
	return LogLikelihoodFCN< PDFSumExtendable<Pdfs...>,
            decltype(std::declval<const Histogram>().GetBinsCenters().begin()),
            decltype(std::declval<const Histogram>().GetBinsContents().begin()) >( functor,
			points.GetBinsCenters().begin(),
			points.GetBinsCenters().end(),
			points.GetBinsContents().begin() );
}


}  // namespace hydra






#endif /* LOGLIKELIHOODFCN2_INL_ */
