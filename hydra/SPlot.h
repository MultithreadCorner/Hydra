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
#include <hydra/multiarray.h>
#include <hydra/Distance.h>
#include <hydra/detail/AddPdfBase.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/Eigen/Dense>

#include <initializer_list>
#include <utility>

namespace hydra {

template < typename PDF1,  typename PDF2, typename ...PDFs>
class SPlot: public detail::AddPdfBase<PDF1,PDF2,PDFs...>
{
typedef typename detail::tuple_type<(sizeof...(PDFs)+2)*(sizeof...(PDFs)+2),double>::type matrix_t;

public:
	//this typedef is actually a check. If the AddPdf is not built with
	//hydra::pdf, AddPdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::AddPdfBase<PDF1,PDF2,PDFs...>::type base_type;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple<PDF1, PDF2, PDFs...> pdfs_tuple_type;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple<typename PDF1::functor_type,
				typename  PDF2::functor_type,
				typename  PDFs::functor_type...> functors_tuple_type;

	constexpr static size_t npdfs = sizeof...(PDFs)+2;

	template<size_t N, size_t I>
	struct index
	{
		constexpr static size_t x= I/N;
		constexpr static size_t y= I%N;
	};

	SPlot( PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf):
		fPDFs( pdf.GetPDFs() ),
		fFunctors( pdf.GetFunctors())
	{
		for(size_t i=0;i<npdfs; i++)
			fCoeficients[i] = pdf.GetCoeficient( i);
	}

	SPlot(SPlot<PDF1, PDF2, PDFs...> const& other ):
		fPDFs(other.GetPDFs() ),
		fFunctors(other.GetFunctors())
	{
		for( size_t i=0; i< npdfs; i++ ){
			fCoeficients[i]=other.GetCoeficient(i);
		}
	}

	inline const pdfs_tuple_type&
	GetPDFs() const
	{
		return fPDFs;
	}

	inline const functors_tuple_type& GetFunctors() const
	{
		return fFunctors;
	}

	inline	const Parameter& GetCoeficient(size_t i) const
	{
		return fCoeficients[i];
	}


	template<typename InputIterator, typename OutputIterator>
	inline HYDRA_EXTERNAL_NS::Eigen::Matrix<double, sizeof...(PDFs)+2, sizeof...(PDFs)+2>
	Generate(InputIterator input_begin, InputIterator input_end,	OutputIterator output_begin);

	template<typename InputIterable, typename OutputIterable>
	inline typename std::enable_if<	hydra::detail::is_iterable<InputIterable>::value &&
		hydra::detail::is_iterable<OutputIterable>::value,
	     HYDRA_EXTERNAL_NS::Eigen::Matrix<double, sizeof...(PDFs)+2, sizeof...(PDFs)+2>>::type
	Generate(InputIterable&& input, OutputIterable&& output);





private:

	template<size_t I, typename ...T>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == sizeof...(T)),void >::type
	SetCovMatrix( HYDRA_EXTERNAL_NS::thrust::tuple<T...> const&, HYDRA_EXTERNAL_NS::Eigen::Matrix<double, npdfs, npdfs>&)
	{ }

	template<size_t I=0, typename ...T>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < sizeof...(T)),void >::type
	SetCovMatrix( HYDRA_EXTERNAL_NS::thrust::tuple<T...> const& tpl,
			HYDRA_EXTERNAL_NS::Eigen::Matrix<double, npdfs, npdfs>& fCovMatrix  )
	{

		fCovMatrix(index< npdfs, I>::x, index< npdfs, I>::y )=HYDRA_EXTERNAL_NS::thrust::get<I>(tpl);
		SetCovMatrix<I+1, T...>(tpl, fCovMatrix);
	}

	Parameter    fCoeficients[npdfs];
	pdfs_tuple_type fPDFs;
	functors_tuple_type fFunctors;
	//HYDRA_EXTERNAL_NS::Eigen::Matrix<double, npdfs, npdfs> fCovMatrix;


};


template < typename PDF1,  typename PDF2, typename ...PDFs>
SPlot<PDF1, PDF2, PDFs...> make_splot(PDFSumExtendable<PDF1, PDF2, PDFs...> const& pdf)
{
 return 	SPlot<PDF1, PDF2, PDFs...>(pdf);
}

}  // namespace hydra


#include <hydra/detail/SPlot.inl>

#endif /* SPLOT_H_ */
