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
 * multivector.h
 *
 *  Created on: 18/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_H_
#define MULTIVECTOR_H_

#include<thrust/iterator/zip_iterator.h>

namespace hydra {

namespace experimental {

template< template<typename T, typename A> class V, typename A, typename ...Ts>
class multivector{

public:

	//allocator
    typedef A  allocator_type;

    //tuples of types
	typedef thrust::tuple<Ts...> 			                        value_tuple_type;
	typedef thrust::tuple<V<Ts, A>...> 		                        storage_tuple_type;
	typedef thrust::tuple<typename V<Ts, A>::pointer...> 	        pointer_tuple_type;
	typedef thrust::tuple<typename V<Ts, A>::const_pointer...> 	    const_pointer_tuple_type;
	typedef thrust::tuple<typename V<Ts, A>::reference...> 	        reference_tuple;
	typedef thrust::tuple<typename V<Ts, A>::const_reference...>    const_reference_tuple;
	typedef thrust::tuple<typename V<Ts, A>::size_type...> 	        size_type_tuple;
	typedef thrust::tuple<typename V<Ts, A>::iterator...> 	        iterator_tuple;
	typedef thrust::tuple<typename V<Ts, A>::const_iterator...>     const_iterator_tuple;
	typedef thrust::tuple<typename V<Ts, A>::reverse_iterator...>   reverse_iterator_tuple;
	typedef thrust::tuple<typename V<Ts, A>::const_reverse_iterator...> const_reverse_iterator_tuple;

	//zipped iterators
	typedef thrust::zip_iterator<iterator_tuple>                 iterator;
	typedef thrust::zip_iterator<const_iterator_tuple>           const_iterator;

	//zipped reverse_iterators
	typedef thrust::zip_iterator<reverse_iterator_tuple>         reverse_iterator;
	typedef thrust::zip_iterator<const_reverse_iterator_tuple>   const_reverse_iterator;




};

}  // namespace experimental

}  // namespace hydra


#endif /* MULTIVECTOR_H_ */
