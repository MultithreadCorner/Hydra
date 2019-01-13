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
 * SPlot.inl
 *
 *  Created on: 12/09/2017
 *      Author: Antonio Augusto Alves Junior
 */



#include <hydra/multiarray.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/functors/ProcessSPlot.h>


namespace hydra {

template <typename PDF1, typename PDF2, typename ...PDFs>
template<typename InputIterator, typename OutputIterator>
inline HYDRA_EXTERNAL_NS::Eigen::Matrix<double, sizeof...(PDFs)+2, sizeof...(PDFs)+2>
SPlot<PDF1,PDF2,PDFs...>::Generate(InputIterator in_begin, InputIterator in_end,
		OutputIterator out_begin)	{

	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<InputIterator>::type system;

	auto bsize    = hydra::distance(in_begin, in_end);

    //_____________________________________
    // covariance matrix calculation

    matrix_t init;
    matrix_t covmatrix= HYDRA_EXTERNAL_NS::thrust::transform_reduce(system(), in_begin, in_end,
    		detail::CovMatrixUnary<typename PDF1::functor_type, typename  PDF2::functor_type,
			typename  PDFs::functor_type...>(fCoeficients, fFunctors ),
    		init, detail::CovMatrixBinary< matrix_t>());

    HYDRA_EXTERNAL_NS::Eigen::Matrix<double, npdfs, npdfs> fCovMatrix;

    SetCovMatrix(covmatrix, fCovMatrix);

    //_____________________________________
    // calculate the sweights

    HYDRA_EXTERNAL_NS::thrust::transform(system(), in_begin, in_end,
    		out_begin, detail::SWeights<typename PDF1::functor_type, typename  PDF2::functor_type,
			typename  PDFs::functor_type...>(fCoeficients, fFunctors, fCovMatrix.inverse() ));

    return  fCovMatrix;

}

template <typename PDF1, typename PDF2, typename ...PDFs>
template<typename InputIterable, typename OutputIterable>
inline typename std::enable_if<	hydra::detail::is_iterable<InputIterable>::value &&
		hydra::detail::is_iterable<OutputIterable>::value,
	     HYDRA_EXTERNAL_NS::Eigen::Matrix<double, sizeof...(PDFs)+2, sizeof...(PDFs)+2>>::type
SPlot<PDF1,PDF2,PDFs...>::Generate(InputIterable&& input, OutputIterable&& output)	{

	return this->Generate(std::forward<InputIterable>(input).begin(),
					std::forward<InputIterable>(input).end(),
	           std::forward<OutputIterable>(output).begin() );
}






} // namespace hydra


