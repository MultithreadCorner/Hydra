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
#include <hydra/detail/BackendTraits.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/AddPdf.h>
#include <hydra/multiarray.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <initializer_list>
#include <utility>

namespace hydra {

template < typename PDF1,  typename PDF2, typename ...PDFs>
class SPlot: public detail::AddPdfBase<PDF1,PDF2,PDFs...>
{

public:
	//this typedef is actually a check. If the AddPdf is not built with
	//hydra::pdf, AddPdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::AddPdfBase<PDF1,PDF2,PDFs...>::type base_type;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple<PDF1, PDF2, PDFs...> pdfs_tuple_type;
	constexpr static size_t npdfs = sizeof...(PDFs)+2;

	SPlot( AddPdf<PDF1, PDF2, PDFs...> const& pdf):
		fPDFs( pdf.GetPDFs() )
	{
		for(size_t i=0;i<npdfs; i++)
			fCoeficients[i] = pdf.GetCoeficient( i);
	}

	SPlot(SPlot<PDF1, PDF2, PDFs...> const& other ):
		fPDFs(other.GetPDFs() )
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

	inline	const Parameter& GetCoeficient(size_t i) const
	{
		return fCoeficients[i];
	}


	template<typename InputIterator, typename OutputIterator>
	inline void Generate(InputIterator in_begin, InputIterator in_end,
			OutputIterator out_begin) const;


private:

	Parameter    fCoeficients[npdfs];
	pdfs_tuple_type fPDFs;


};

}  // namespace hydra


#include <hydra/detail/SPlot.inl>

#endif /* SPLOT_H_ */
