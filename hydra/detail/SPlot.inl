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
 * SPlot.inl
 *
 *  Created on: 12/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPLOT_INL_
#define SPLOT_INL_

#include <hydra/multiarray.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
//http://coliru.stacked-crooked.com/a/57ddddf668a4433f
namespace hydra {

template <typename PDF1, typename PDF2, typename ...PDFs>
template<typename InputIterator, typename OutputIterator>
inline void SPlot<PDF1,PDF2,PDFs...>::Generate(InputIterator in_begin, InputIterator in_end,
		OutputIterator out_begin) const	{

	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type system;

	typedef typename detail::tuple_type<(sizeof...(PDFs))*(sizeof...(PDFs)),double>::type matrix_t;

	auto bsize    = hydra::distance(in_begin, in_end);
    auto functors = fPDFs.GetFunctors();

    //_____________________________________
    // covariance matrix calculation

    matrix_t init{0};
    matrix_t = HYDRA_EXTERNAL_NS::thrust::transform_reduce(system(), in_begin, in_end,
    		detail::CovMatrixUnary( functors ), init, detail::CovMatrixBinary());

    //_____________________________________


}


} // namespace hydra

#endif /* SPLOT_INL_ */
