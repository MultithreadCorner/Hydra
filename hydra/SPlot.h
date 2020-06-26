/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * SPlot.h
 *
 *  Created on: 04/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPLOT_H_
#define SPLOT_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/Pdf.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/PDFSumExtendable.h>
#include <hydra/detail/AddPdfBase.h>
#include <hydra/Tuple.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>
#include <hydra/detail/functors/ProcessSPlot.h>

#include <Eigen/Dense>

#include <initializer_list>
#include <utility>

namespace hydra {

/**
 *  \ingroup fit
 *  \class SPlot
 *
 *  Implementation of {s}_{Plot} technique for statistical unfolding of sample
 *  containing events from different sources.
 *  The sPlots are applicable in the context extended Likelihood fits, which are performed
 *  on the data sample to determine the yields of the various sources.
 *
 *  This class requires Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page).
 *
 *  Reference:  Nucl.Instrum.Meth.A555:356-369,2005
 */
template <typename Iterator, typename PDF1,  typename PDF2, typename ...PDFs>
class SPlot: public detail::AddPdfBase<PDF1,PDF2,PDFs...>
{

	//this typedef is actually a check. If the AddPdf is not built with
	//hydra::pdf, AddPdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::AddPdfBase<PDF1,PDF2,PDFs...>::type base_type;
    typedef typename hydra_thrust::iterator_system<Iterator>::type system_type;

public:

	typedef hydra_thrust::tuple<PDF1, PDF2, PDFs...> pdfs_tuple_type;

	typedef hydra_thrust::tuple<
			    typename PDF1::functor_type,
				typename PDF2::functor_type,
				typename PDFs::functor_type...> functors_tuple_type;

	typedef detail::SWeights<typename PDF1::functor_type,
			                 typename PDF2::functor_type,
			                 typename PDFs::functor_type...> transformer;

	typedef hydra_thrust::transform_iterator<transformer, Iterator > iterator;
	typedef typename  hydra_thrust::iterator_traits<iterator>::value_type value_type;

	template<int I>
	using siterator = hydra_thrust::transform_iterator< detail::GetSWeight<I>, iterator >;

	constexpr static size_t npdfs = sizeof...(PDFs)+2;

	SPlot()=delete;

    /**
     * SPlot constructor.
     *
     * @param pdf PDFSumExtendable<PDF1, PDF2, PDFs...> object, already optimized.
     * @param first Iterator pointing to the beginning of the data range used to optimize ```pdf```
     * @param last  Iterator pointing to the end of the data range used to optimize ```pdf```.
     */
	SPlot( PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf, Iterator first, Iterator last):
		fPDFs( pdf.GetPDFs() ),
		fFunctors( pdf.GetFunctors()),
		fCovMatrix( Eigen::Matrix<double, npdfs, npdfs>{} ),
	    fBegin( iterator( first, transformer(  pdf.GetFunctors(), Eigen::Matrix<double, npdfs, npdfs>{} ))),
		fEnd (iterator( last , transformer(  pdf.GetFunctors(), Eigen::Matrix<double, npdfs, npdfs>{} )))

	{
		for(size_t i=0;i<npdfs; i++)
			fCoeficients[i] = pdf.GetCoeficient(i);

		//fCovMatrix << 0.0, 0.0, 0.0, 0.0;


		Eigen::Matrix<double, npdfs, npdfs>  init = Eigen::Matrix<double, npdfs, npdfs>::Zero();
		//init << 0.0, 0.0, 0.0, 0.0;

		fCovMatrix = hydra_thrust::transform_reduce(system_type(), first, last,
				detail::CovMatrixUnary<
				 typename PDF1::functor_type,
				 typename PDF2::functor_type,
				 typename PDFs::functor_type...>(fCoeficients, fFunctors ),
				 init, detail::CovMatrixBinary() );

		Eigen::Matrix<double, npdfs, npdfs> inverseCovMatrix = fCovMatrix.inverse();

		fBegin = iterator( first, transformer(fCoeficients, fFunctors, inverseCovMatrix ));
		fEnd   = iterator( last , transformer(fCoeficients, fFunctors, inverseCovMatrix ));


	}

	/**
	 * SPlot copy constructor.
	 *
	 * @param other
	 */
	SPlot(SPlot<Iterator, PDF1, PDF2, PDFs...> const& other ):
		fPDFs(other.GetPDFs() ),
		fFunctors(other.GetFunctors()),
    	fBegin(other.begin()),
	    fEnd(other.end()),
	    fCovMatrix(other.GetCovMatrix() )
	{
		for( size_t i=0; i< npdfs; i++ ){
			fCoeficients[i]=other.GetCoeficient(i);
		}
	}

	/**
	 * SPlot assignment operator.
	 *
	 * @param other
	 */
	SPlot<Iterator, PDF1, PDF2, PDFs...>
	operator=(SPlot<Iterator, PDF1, PDF2, PDFs...> const& other ){

		if(this==&other) return *this;

		fPDFs=other.GetPDFs();
		fFunctors=other.GetFunctors();
		fBegin=other.begin();
		fEnd=other.end();
		fCovMatrix=other.GetCovMatrix();

		for( size_t i=0; i< npdfs; i++ ){
			fCoeficients[i]=other.GetCoeficient(i);
		}

		return *this;
	}

	/**
	 * Get reference to constant of PDF objects.
	 * @return hydra::tuple of the PDF objects
	 */
	inline const pdfs_tuple_type&
	GetPDFs() const {
		return fPDFs;
	}

	/**
	 * Get reference to constant of normalized Functor objects.
	 * @return hydra::tuple of the Functor objects
	 */
	inline const functors_tuple_type& GetFunctors() const {
		return fFunctors;
	}

	/**
	 * Get the yield corresponding to the PDF i.
	 * @param i index of PDF
	 * @return hydra::Parameter
	 */
	inline	const Parameter& GetCoeficient(size_t i) const {
		return fCoeficients[i];
	}

	/**
	 * Get the covariance matrix of between the yields of PDFs.
	 * @return Eigen::Matrix<double, npdfs, npdfs>
	 */
	Eigen::Matrix<double, npdfs, npdfs>
	GetCovMatrix() const {

		return fCovMatrix;
	}

	/**
	 * Get an iterator pointing to beginning of the range of the s-weights corresponding to the PDF i.
	 * @param hydra placeholder (_0, _1, ..., _N)
	 * @return iterator
	 */
	template<unsigned int I>
	siterator<I> begin(placeholders::placeholder<I>) {

		return siterator<I>(fBegin, detail::GetSWeight<I>());
	}

	/**
	 * Get an iterator pointing to end of the range of the s-weights corresponding to the PDF i.
	 * @param hydra placeholder (_0, _1, ..., _N)
	 * @return iterator
	 */
	template<unsigned int I>
	siterator<I> end(placeholders::placeholder<I>) {

		return siterator<I>(fEnd, detail::GetSWeight<I>());
	}


	/**
	 * Get an iterator pointing to end of the range of the s-weights.
	 * @return iterator
	 */
	iterator begin() {
		return fBegin;
	}

	/**
	 * Get an iterator pointing to end of the range of the s-weights.
	 * @return iterator
	 */
	iterator end() {
		return fEnd;
	}

	iterator begin() const {
		return fBegin;
	}

	iterator end() const {
		return fEnd;
	}

	/**
	 * Get a range with the s-weights  to the PDF i.
	 * @param hydra placeholder (_0, _1, ..., _N)
	 * @return Range<siterator<I>>
	 */
	template<unsigned int I>
	hydra::Range<siterator<I>>
	operator()(placeholders::placeholder<I>  p){

		return hydra::make_range( begin(p), end(p));
	}

	/**
	 * Get a range with the s-weights.
	 * @return hydra::Range<iterator>
	 */
	template<unsigned int I>
	hydra::Range<iterator>
	operator()(){

		return hydra::make_range( begin(), end());
	}

	/**
	 * Subscript operator to get a range with the s-weights  to the PDF i.
	 * @param hydra placeholder (_0, _1, ..., _N)
	 * @return hydra::Range<iterator>
	 */
	template<unsigned int I>
	hydra::Range<iterator>
	operator[]( placeholders::placeholder<I>  p){

		return hydra::make_range( begin(p), end(p));
	}

	/**
	 * Subscript operator.
	 * @param index i
	 * @return value_type
	 */
	value_type operator[](size_t i){
		return fBegin[i];
	}

private:

	Parameter           fCoeficients[npdfs];
	pdfs_tuple_type     fPDFs;
	functors_tuple_type fFunctors;

	Eigen::Matrix<double, npdfs, npdfs> fCovMatrix;
	iterator fBegin;
	iterator fEnd;

};

/**
 * Convenience function for instantiating SPlot objects using type deduction
 *
 * @param pdf PDFSumExtendable<PDF1, PDF2, PDFs...> optimized object
 * @param first iterator pointing to beginning of the data range.
 * @param last iterator pointing to end of the data  range.
 * @return
 */
template <typename Iterator, typename PDF1,  typename PDF2, typename ...PDFs>
typename std::enable_if< detail::is_iterator<Iterator>::value,
                 SPlot<Iterator, PDF1, PDF2, PDFs...> >::type
make_splot(PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf, Iterator first, Iterator last) {

 return 	SPlot<Iterator, PDF1, PDF2, PDFs...>(pdf, first,last);
}

/**
 * Convenience function for instantiating SPlot objects using type deduction
 *
 * @param pdf PDFSumExtendable<PDF1, PDF2, PDFs...> optimized object
 * @param data iterable representing the data-range
 * @return
 */
template <typename Iterable, typename PDF1,  typename PDF2, typename ...PDFs>
typename std::enable_if< detail::is_iterable<Iterable>::value,
                  SPlot< decltype(std::declval<Iterable>().begin()), PDF1, PDF2, PDFs...> >::type
make_splot(PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf, Iterable&& data){

 return 	SPlot< decltype(std::declval<Iterable>().begin()) ,
		                 PDF1, PDF2, PDFs...>(pdf, std::forward<Iterable>(data).begin(),
		  	  	  	  	  	  	  	  	  std::forward<Iterable>(data).end());
}

}  // namespace hydra

#endif /* SPLOT_H_ */
