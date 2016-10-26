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
 * multivector.inl
 *
 *  Created on: Oct 26, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_INL_
#define MULTIVECTOR_INL_

#include <thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>

namespace hydra {

namespace experimental {

//----------------------
template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline	auto multivector<Vector,Allocator,T...>::vbegin(void)
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::iterator_tuple>::type
{
	return	thrust::get<I>(fTBegin);
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline	auto multivector<Vector,Allocator,T...>::vend()
-> typename thrust::tuple_element<I,typename multivector<Vector,Allocator,T...>::iterator_tuple>::type
{
	return	thrust::get<I>(fTBegin)+fSize;
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline	auto multivector<Vector,Allocator,T...>::vcbegin() const
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::const_iterator_tuple>::type
{
	return	thrust::get<I>(fTConstBegin);
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline	auto multivector<Vector,Allocator,T...>::vcend() const
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::const_iterator_tuple>::type
{
	return thrust::get<I>(fTConstBegin)+fSize;
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline	auto multivector<Vector,Allocator,T...>::vrbegin()
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::reverse_iterator_tuple>::type
{
	return	thrust::get<I>(fTReverseBegin);
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline auto multivector<Vector,Allocator,T...>::vrend()
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::reverse_iterator_tuple>::type
{
	return	thrust::get<I>(fTReverseBegin)+fSize;
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline auto multivector<Vector,Allocator,T...>::vcrbegin() const
-> typename thrust::tuple_element<I,typename multivector<Vector,Allocator,T...>::const_reverse_iterator_tuple>::type
{
	return	thrust::get<I>(fTConstReverseBegin);
}

template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
template<unsigned int I>
inline auto multivector<Vector,Allocator,T...>::vcrend() const
-> typename thrust::tuple_element<I,typename  multivector<Vector,Allocator,T...>::const_reverse_iterator_tuple>::type
{
	return	thrust::get<I>(fTConstReverseBegin)+fSize;
}



}  // namespace experimental

}  // namespace hydra


#endif /* MULTIVECTOR_INL_ */
