/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * LogLikelihoodFCN2.h
 *
 *  Created on: 13/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN_H_
#define LOGLIKELIHOODFCN_H_


#include <hydra/FCN.h>
#include <hydra/Pdf.h>
#include <hydra/PDFSumExtendable.h>
#include <hydra/PDFSumNonExtendable.h>
#include <hydra/detail/HistogramTraits.h>
#include <hydra/detail/Iterable_traits.h>

namespace hydra {

namespace detail {

template<typename ...Iterators>
struct are_iterators: std::conditional<sizeof...(Iterators)==0, std::true_type,
		     detail::all_true<detail::is_iterator<Iterators>::value...> >::type{};

template<typename ...Iterables>
struct are_iterables: std::conditional<sizeof...(Iterables)==0, std::true_type,
		     detail::all_true<detail::is_iterable<Iterables>::value...> >::type{};


}  // namespace detail

template<typename PDF, typename Iterator, typename... Extensions>
class LogLikelihoodFCN;

/**
 * \ingroup fit
 * \brief LogLikehood object for not composed models represeted by hydra::Pdf objects
 * \tparam Functor
 * \tparam Integrator
 * \tparam IteratorD
 * \tparam IteratorW
 */
template<typename Functor, typename Integrator, typename IteratorD, typename ...IteratorW>
class LogLikelihoodFCN< Pdf<Functor,Integrator> , IteratorD, IteratorW...>;

/**
 * \class LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>
 * \ingroup fit
 */
template<typename ...Pdfs, typename IteratorD , typename ...IteratorW>
class LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, IteratorD, IteratorW...>;

/**
 * \ingroup fit
 */
template<typename ...Pdfs, typename IteratorD , typename ...IteratorW>
class LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, IteratorD, IteratorW...>;

/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param first iteraror pointing to begin of data range
 * @param last iteraror pointing to end of data range
 * @param weights iteraror pointing to begin of weights range
 * @return
 */
template< typename Functor, typename Integrator,  typename Iterator, typename ...Iterators>
inline typename std::enable_if< detail::is_iterator<Iterator>::value && detail::are_iterators<Iterators...>::value,
LogLikelihoodFCN< Pdf<Functor,Integrator>, Iterator , Iterators... > >::type
make_loglikehood_fcn(Pdf<Functor,Integrator> const& pdf, Iterator first, Iterator last,  Iterators... weights );


/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param first iteraror pointing to begin of data range
 * @param last iteraror pointing to end of data range
 * @param weights iteraror pointing to begin of weights range
 * @return
 */
template<typename... Pdfs,  typename Iterator, typename ...Iterators >
inline typename std::enable_if< detail::is_iterator<Iterator>::value && detail::are_iterators<Iterators...>::value,
LogLikelihoodFCN< PDFSumExtendable<Pdfs...>, Iterator, Iterators...> >::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& functor, Iterator first, Iterator last, Iterators... weights );


/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param first iteraror pointing to begin of data range
 * @param last iteraror pointing to end of data range
 * @param weights iteraror pointing to begin of weights range
 * @return
 */

template<typename... Pdfs,  typename Iterator, typename ...Iterators >
inline typename std::enable_if< hydra::detail::is_iterator<Iterator>::value && detail::are_iterators<Iterators...>::value,
LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>, Iterator,Iterators...  >>::type
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...>const& pdf, Iterator first, Iterator last, Iterators... weights);


//----------------------------------------
//interface to iterables

/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param points "iterable" storing the data
 * @param weights "iterables" storing the weights
 * @return
 */
template< typename Functor, typename Integrator, typename Iterable, typename ...Iterables >
inline typename std::enable_if< (!detail::is_iterator<Iterable>::value) &&
								((sizeof...(Iterables)==0) || !detail::are_iterators<Iterables...>::value) &&
								(!hydra::detail::is_hydra_dense_histogram< typename std::remove_reference<Iterable>::type>::value) &&
								(!hydra::detail::is_hydra_sparse_histogram<typename std::remove_reference<Iterable>::type>::value) &&
								detail::is_iterable<Iterable>::value &&
								detail::are_iterables<Iterables...>::value,
LogLikelihoodFCN< Pdf<Functor,Integrator>, decltype(std::declval<Iterable>().begin()),
                  decltype(std::declval<Iterables >().begin())... >>::type
make_loglikehood_fcn(Pdf<Functor,Integrator> const& pdf, Iterable&& points, Iterables&&... weights );



/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param points "iterable" storing the data
 * @param weights "iterables" storing the weights
 * @return
 */
template<typename ...Pdfs, typename Iterable, typename... Iterables>
inline typename std::enable_if<   (!detail::is_iterator<Iterable>::value) &&
                                  ((sizeof...(Iterables)==0) || !detail::are_iterators<Iterables...>::value) &&
                                  (!hydra::detail::is_hydra_dense_histogram< typename std::remove_reference<Iterable>::type>::value) &&
		                          (!hydra::detail::is_hydra_sparse_histogram<typename std::remove_reference<Iterable>::type>::value) &&
								  detail::is_iterable<Iterable>::value && detail::are_iterables<Iterables...>::value,
LogLikelihoodFCN<  PDFSumExtendable<Pdfs...>, decltype(std::declval<Iterable>().begin()),
                     decltype(std::declval<Iterables>().begin())...> >::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& functor, Iterable&& points, Iterables&& ...weights );


/**
 * \ingroup fit
 * \brief Conveniency function to build up loglikehood fcns
 * @param pdf hydra::Pdf object
 * @param first iteraror pointing to begin of data range
 * @param last iteraror pointing to end of data range
 * @param weights iteraror pointing to begin of weights range
 * @return
 */
template<typename ...Pdfs, typename Iterable, typename ...Iterables>
inline typename std::enable_if< (!detail::is_iterator<Iterable>::value) &&
								((sizeof...(Iterables)==0) || !detail::are_iterators<Iterables...>::value) &&
								(!hydra::detail::is_hydra_dense_histogram< typename std::remove_reference<Iterable>::type>::value) &&
								(!hydra::detail::is_hydra_sparse_histogram<typename std::remove_reference<Iterable>::type>::value) &&
								hydra::detail::is_iterable<Iterable>::value &&
								detail::are_iterables<Iterables...>::value,
LogLikelihoodFCN<  PDFSumNonExtendable<Pdfs...>, decltype(std::declval< Iterable>().begin()),
                     decltype(std::declval< Iterables>().begin())... > >::type
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...> const& functor, Iterable&& points, Iterables&&... weights );




//----------------------------------------
//handy interface to histograms

/**
 * \ingroup fit
 * \brief Convenience function to build up loglikehood fcns for densely and sparsely binned datasets.
 * @param pdf hydra::Pdf<Functor,Integrator> object.
 * @param data histogram storing the data.
 * @return hydra::LogLikelihoodFCN instance hydra::Pdf<Functor,Integrator>  for .
 */
template< typename Functor, typename Integrator, typename Histogram>
inline typename std::enable_if<detail::is_hydra_dense_histogram<Histogram>::value ||
                               detail::is_hydra_sparse_histogram<Histogram>::value,
LogLikelihoodFCN< Pdf<Functor,Integrator>,
				  decltype(std::declval<const Histogram>().GetBinsCenters().begin()),
                  decltype( std::declval<const Histogram>().GetBinsContents().begin())>>::type
make_loglikehood_fcn(Pdf<Functor,Integrator> const& pdf, Histogram const& points );


/**
 * \ingroup fit
 * \brief Convenience function to build up loglikehood fcns for densely and sparsely binned datasets.
 * @param pdf hydra::PDFSumExtendable<Pdfs...> object.
 * @param data histogram storing the data.
 * @return hydra::LogLikelihoodFCN instance for hydra::PDFSumExtendable<Pdfs...>.
 */
template<typename ...Pdfs, typename Histogram>
inline typename std::enable_if<detail::is_hydra_dense_histogram<Histogram>::value ||
                               detail::is_hydra_sparse_histogram<Histogram>::value,
 LogLikelihoodFCN< PDFSumExtendable<Pdfs...>,
                     decltype(std::declval<const Histogram&>().GetBinsCenters().begin()),
                     decltype(std::declval<const Histogram&>().GetBinsContents().begin()) >>::type
make_loglikehood_fcn(PDFSumExtendable<Pdfs...> const& pdf, Histogram const&  data);


/**
 * \ingroup fit
 * \brief Convenience function to build up loglikehood fcns for densely and sparsely binned datasets
 * @param pdf hydra::PDFSumNonExtendable<Pdfs...> object
 * @param data histogram storing the data
 * @return hydra::LogLikelihoodFCN instance for hydra::PDFSumNonExtendable<Pdfs...>
 */
template<typename ...Pdfs, typename Histogram>
inline typename std::enable_if<detail::is_hydra_dense_histogram<Histogram>::value ||
                               detail::is_hydra_sparse_histogram<Histogram>::value,
 LogLikelihoodFCN< PDFSumNonExtendable<Pdfs...>,
                     decltype(std::declval<const Histogram&>().GetBinsCenters().begin()),
                     decltype(std::declval<const Histogram&>().GetBinsContents().begin()) >>::type
make_loglikehood_fcn(PDFSumNonExtendable<Pdfs...> const& pdf, Histogram const&  data);


}  // namespace hydra

#include<hydra/detail/LogLikelihoodFCN1.inl>
#include<hydra/detail/LogLikelihoodFCN2.inl>
#include<hydra/detail/LogLikelihoodFCN3.inl>

#endif /* LOGLIKELIHOODFCN2_H_ */
