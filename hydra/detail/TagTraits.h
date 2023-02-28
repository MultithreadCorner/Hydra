/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * FunctorTraits.h
 *
 *  Created on: 25/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef TAGTRAITS_H_
#define TAGTRAITS_H_


#include <type_traits>
#include <hydra/Types.h>


namespace hydra {

namespace detail {

//tags to identify hydra pdf and functors

template<class T, class R = void>
struct tag_type { typedef R type; };


//sum of pdfs
template<class T, class Enable = void>
struct is_hydra_sum_pdf: std::false_type {};

template<class T>
struct is_hydra_sum_pdf<T, typename tag_type< typename T::hydra_sum_pdf_tag>::type>: std::true_type {};


//storable
template<class T, class Enable = void>
struct is_hydra_convertible_to_tuple: std::false_type {};

template<class T>
struct is_hydra_convertible_to_tuple<T, typename tag_type< typename T::hydra_convertible_to_tuple_tag>::type>: std::true_type {};



}  // namespace detail
}// namespace hydra

#endif /* TAGTRAITS_H_ */
