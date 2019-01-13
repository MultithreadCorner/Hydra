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
 * LogLikelihoodFCN3.inl
 *
 *  Created on: 09/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN3_INL_
#define LOGLIKELIHOODFCN3_INL_



#include <hydra/FCN.h>
#include <hydra/PDFSumNonExtendable.h>
#include <hydra/detail/functors/LogLikelihood1.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/inner_product.h>

#include <utility>

namespace hydra {

/**
 * \ingroup fit
 * \brief LogLikehood object for composed models represented by hydra::PDFSumNonExtendable<Pdfs...> objects
 * \tparam Functor
 * \tparam Integrator
 * \tparam IteratorD
 * \tparam IteratorW
 */
template<typename ...Pdfs, typename IteratorD , typename ...IteratorW>
class LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>: public FCN<LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW ...> >
{

public:

	LogLikelihoodFCN(PDFSumNonExtendable<Pdfs...>const& functor, IteratorD begin, IteratorD end, IteratorW ...wbegin):
		FCN<LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>>(functor,begin, end, wbegin...)
		{}

	LogLikelihoodFCN(LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>const& other):
		FCN<LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>>(other)
		{}

	LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>&
	operator=(LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>const& other)
	{
		if(this==&other) return  *this;
		FCN<LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>>::operator=(other);
		return  *this;
	}


	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M==0), double >::type
	Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<IteratorD>::type System;
		typedef typename PDFSumNonExtendable<Pdfs...>::functor_type functor_type;
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

		const_cast< LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>*  >(this)->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood1<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::transform_reduce(select_system(system), this->begin(), this->end(),
				NLL, init, HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>());

		GReal_t  r = (GReal_t)this->GetDataSize()  - final;



		return r;

	}

	template<size_t M = sizeof...(IteratorW)>
	inline typename std::enable_if<(M>0), double >::type
	Eval( const std::vector<double>& parameters ) const{

		using   HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
		typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<typename FCN<LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>>::iterator>::type System;
		typedef typename PDFSumNonExtendable<Pdfs...>::functor_type functor_type;
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

		const_cast< LogLikelihoodFCN<PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>*  >(this)->GetPDF().SetParameters(parameters);

		auto NLL = detail::LogLikelihood2<functor_type>(this->GetPDF().GetFunctor());

		final = HYDRA_EXTERNAL_NS::thrust::inner_product(select_system(system), this->begin(), this->end(),this->wbegin(),
				init,HYDRA_EXTERNAL_NS::thrust::plus<GReal_t>(),NLL );

		GReal_t  r = (GReal_t)this->GetDataSize()  - final;



		return r;

	}


};


/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param first iteraror pointing to begin of data range
 * @param last iteraror pointing to end of data range
 * @param weights iteraror pointing to begin of weights range
 * @return
 */

template<typename... Pdfs,  typename Iterator, typename ...Iterators>
auto make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...>const& pdf, Iterator first, Iterator last, Iterators... weights)
-> LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, Iterator,Iterators...  >
{
	return LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, Iterator,Iterators... >(pdf,first,last,weights...);
}


template<typename ...Pdfs, typename Iterable, typename ...Iterables, typename U >
inline typename std::enable_if< (!hydra::detail::is_hydra_dense_histogram<Iterable>::value) &&
								(!hydra::detail::is_hydra_sparse_histogram<Iterable>::value) &&
								hydra::detail::is_iterable<Iterable>::value &&
								U::value,
LogLikelihoodFCN<  PDFSumNonExtendable<Pdfs...>,
                     decltype(std::declval< const Iterable>().begin()),
                     decltype(std::declval< const Iterables>().begin())... > >::type
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...> const& functor, Iterable const& points, Iterables const&... weights )
{
	return make_loglikehood_fcn( functor, points.begin(), points.end(), weights.begin()...);
}

template<typename ...Pdfs, typename T, size_t N, hydra::detail::Backend BACKEND,typename D>
LogLikelihoodFCN<  PDFSumNonExtendable<Pdfs...>,
                     decltype(std::declval<DenseHistogram<T,N,detail::BackendPolicy<BACKEND>,D>>().GetBinsCenters().begin()),
                     decltype(std::declval<DenseHistogram<T,N,detail::BackendPolicy<BACKEND>,D>&>().GetBinsContents().begin()) >
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...> const& functor,
		DenseHistogram<T,N,detail::BackendPolicy<BACKEND>,D> & points)
{

	return make_loglikehood_fcn( functor, points.GetBinsCenters().begin(),
			points.GetBinsCenters().end(),
			points.GetBinsContents().begin());
}

template<typename ...Pdfs, typename T, size_t N, hydra::detail::Backend BACKEND,typename D>
LogLikelihoodFCN<  PDFSumNonExtendable<Pdfs...>,
                     decltype(std::declval<SparseHistogram<T,N,detail::BackendPolicy<BACKEND>,D>>().GetBinsCenters().begin()),
                     decltype(std::declval<SparseHistogram<T,N,detail::BackendPolicy<BACKEND>,D>>().GetBinsContents().begin()) >
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...> const& functor,
		SparseHistogram<T,N,detail::BackendPolicy<BACKEND>,D> & points)
{
	return make_loglikehood_fcn( functor, points.GetBinsCenters().begin(),
			points.GetBinsCenters().end(),
			points.GetBinsContents().begin());
}

}  // namespace hydra


#endif /* LOGLIKELIHOODFCN3_INL_ */
