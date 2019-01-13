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
 * HistogramTraits.h
 *
 *  Created on: 03/07/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HISTOGRAMTRAITS_H_
#define HISTOGRAMTRAITS_H_

#include <type_traits>
#include <hydra/Types.h>
#include <hydra/DenseHistogram.h>
#include <hydra/SparseHistogram.h>

namespace hydra {

namespace detail {

//tags to identify hydra histograms
template<class T, class R = void>
struct histogram_type { typedef R type; };

//dense histogram
template<class T, class Enable = void>
struct is_hydra_dense_histogram: std::false_type {};

template<class T>
struct is_hydra_dense_histogram<T, typename tag_type< typename T::hydra_dense_histogram_tag>::type>: std::true_type {};

//sparse histogram
template<class T, class Enable = void>
struct is_hydra_sparse_histogram: std::false_type {};

template<class T>
struct is_hydra_sparse_histogram<T, typename tag_type< typename T::hydra_sparse_histogram_tag>::type>: std::true_type {};


}  // namespace detail
}// namespace hydra


#endif /* HISTOGRAMTRAITS_H_ */
