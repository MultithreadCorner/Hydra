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


	SPlot( PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf, Iterator first, Iterator last):
		fPDFs( pdf.GetPDFs() ),
		fFunctors( pdf.GetFunctors()),
	    fBegin( iterator( first, transformer(  pdf.GetFunctors(), Eigen::Matrix<double, npdfs, npdfs>{} ))),
		fEnd (iterator( last , transformer(  pdf.GetFunctors(), Eigen::Matrix<double, npdfs, npdfs>{} )))

	{
		for(size_t i=0;i<npdfs; i++)
			fCoeficients[i] = pdf.GetCoeficient(i);

		fCovMatrix << 0.0, 0.0, 0.0, 0.0;


		Eigen::Matrix<double, npdfs, npdfs>  init;
		init << 0.0, 0.0, 0.0, 0.0;

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

	inline const pdfs_tuple_type&
	GetPDFs() const {
		return fPDFs;
	}

	inline const functors_tuple_type& GetFunctors() const {
		return fFunctors;
	}

	inline	const Parameter& GetCoeficient(size_t i) const {
		return fCoeficients[i];
	}

	Eigen::Matrix<double, npdfs, npdfs>
	GetCovMatrix() const {

		return fCovMatrix;
	}

	template<unsigned int I>
	siterator<I> begin(placeholders::placeholder<I>) {

		return siterator<I>(fBegin, detail::GetSWeight<I>());
	}

	template<unsigned int I>
	siterator<I> end(placeholders::placeholder<I>) {

		return siterator<I>(fEnd, detail::GetSWeight<I>());
	}

	iterator begin() {
		return fBegin;
	}

	iterator end() {
		return fEnd;
	}

	iterator begin() const {
		return fBegin;
	}

	iterator end() const {
		return fEnd;
	}

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


template <typename Iterator, typename PDF1,  typename PDF2, typename ...PDFs>
typename std::enable_if< detail::is_iterator<Iterator>::value,
                 SPlot<Iterator, PDF1, PDF2, PDFs...> >::type
make_splot(PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf, Iterator first, Iterator last) {

 return 	SPlot<Iterator, PDF1, PDF2, PDFs...>(pdf, first,last);
}


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
