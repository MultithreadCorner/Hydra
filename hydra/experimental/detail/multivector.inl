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
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/experimental/detail/multivector.inc>

namespace hydra {

namespace experimental {

namespace detail {

_GenerateVoidCallArgs(shrink_to_fit)
_GenerateVoidCallArgs(clear)
_GenerateVoidCallArgs(pop_back )
_GenerateVoidCallArgs(reserve)
_GenerateVoidCallArgs(resize)
_GenerateVoidCallTuple(push_back)

_GenerateNonVoidCallArgsC(size)
_GenerateNonVoidCallArgsC(empty)
_GenerateNonVoidCallArgs(front)
_GenerateNonVoidCallArgsC(front)
_GenerateNonVoidCallArgs(back)
_GenerateNonVoidCallArgs(begin)
_GenerateNonVoidCallArgs(end)
_GenerateNonVoidCallArgsC(cbegin)
_GenerateNonVoidCallArgsC(cend)
_GenerateNonVoidCallArgs(rbegin)
_GenerateNonVoidCallArgs(rend)
_GenerateNonVoidCallArgsC(crbegin)
_GenerateNonVoidCallArgsC(crend)
_GenerateNonVoidCallArgs(data)
_GenerateNonVoidCallArgsC(data)
_GenerateNonVoidCallArgsC(capacity)
_GenerateNonVoidCallArgs(erase)

}
template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...T>
multivector<Vector,Allocator,T...>::multivector():
fStorage(thrust::make_tuple( Vector<T, Allocator<T>>()... ) ),
fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
fTBegin(detail::begin_call_args(fStorage) ),
fTReverseBegin( detail::rbegin_call_args(fStorage) ),
fTConstBegin( detail::cbegin_call_args(fStorage) ),
fTConstReverseBegin( detail::crbegin_call_args(fStorage) ),
fSize( thrust::get<0>(fStorage ).size())
{}

/**
 * constructor size_t n
 */

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 multivector<Vector,Allocator,T...>::multivector(size_t n):
 fStorage(thrust::make_tuple( Vector<T, Allocator<T>>(n)... ) ),
 fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
 fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
 fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
 fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
 fTBegin(detail::begin_call_args(fStorage) ),
 fTReverseBegin( detail::rbegin_call_args(fStorage) ),
 fTConstBegin( detail::cbegin_call_args(fStorage) ),
 fTConstReverseBegin( detail::crbegin_call_args(fStorage) ),
 fSize( thrust::get<0>(fStorage ).size())
 {}

 /**
  * constructor size_t n, ...values
  */

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 multivector<Vector,Allocator,T...>::multivector(size_t n, T... value):
 fStorage( detail::_vctor(n, thrust::make_tuple( value... ) )),
 fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
 fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
 fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
 fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
 fTBegin(detail::begin_call_args(fStorage) ),
 fTReverseBegin( detail::rbegin_call_args(fStorage) ),
 fTConstBegin( detail::cbegin_call_args(fStorage) ),
 fTConstReverseBegin( detail::crbegin_call_args(fStorage) ),
 fSize( thrust::get<0>(fStorage ).size())
 { }

 /**
  * constructor size_t n, ...values
  */
 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 multivector<Vector,Allocator,T...>::multivector(size_t n,
		 typename  multivector<Vector,Allocator,T...>::value_tuple_type value):
 fStorage( detail::_vctor<Vector,Allocator,  T...>(n,  value )),
 fBegin(thrust::make_zip_iterator( detail::begin_call_args(fStorage) )),
 fReverseBegin(thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) )),
 fConstBegin(thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) )),
 fConstReverseBegin(thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) )),
 fTBegin(detail::begin_call_args(fStorage) ),
 fTReverseBegin( detail::rbegin_call_args(fStorage) ),
 fTConstBegin( detail::cbegin_call_args(fStorage) ),
 fTConstReverseBegin( detail::crbegin_call_args(fStorage) ),
 fSize( thrust::get<0>(fStorage ).size())
 { }


 /**
  * copy constructor
  */
 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 template< template<typename...> class Vector2,
 template<typename...> class Allocator2>
 multivector<Vector,Allocator,T...>::multivector( multivector< Vector2, Allocator2, T... > const&  other)
 {
	 this->resize(other.size());

	 thrust::copy(other.begin(), other.end(), this->begin() );

 }

 /**
  * assignment operator=
  */
 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 template< template<typename...> class Vector2,
 template<typename...> class Allocator2>
 multivector< Vector, Allocator, T... >&
 multivector<Vector,Allocator,T...>::operator=( multivector< Vector2, Allocator2, T... > const&  v)
 {
	 this->resize(v.size());

	 thrust::copy(v.begin(), v.end(), this->begin() );

	 return *this;
 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 inline void multivector<Vector,Allocator,T...>::pop_back()
 {
	 detail::pop_back_call_args(fStorage);
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin=thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin=thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 inline void multivector<Vector,Allocator,T...>::push_back(T const&... args)
 {
	 detail::push_back_call_tuple(fStorage, thrust::make_tuple(args...) );
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T> inline
 void multivector<Vector,Allocator,T...>::push_back(thrust::tuple<T...> const& args)
 {
	 detail::push_back_call_tuple( fStorage, args);
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::pointer_tuple_type
 multivector<Vector,Allocator,T...>::data()
 {
	 return detail::data_call_args( fStorage );

 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_pointer_tuple_type
 multivector<Vector,Allocator,T...>::data() const
 {
	 return detail::data_call_args( fStorage );
 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 size_t multivector<Vector,Allocator,T...>::size() const
 {
	 //auto sizes = detail::size_call_args( fStorage );
	 //return thrust::get<0>(fStorage ).size();
	 return fSize;
 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 size_t multivector<Vector,Allocator,T...>::capacity() const
 {
	 //auto sizes = detail::capacity_call_args( fStorage );
	 //return thrust::get<0>(sizes);
	 return thrust::get<0>(fStorage ).capacity();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 bool multivector<Vector,Allocator,T...>::empty() const
 {
	 //auto empties = detail::empty_call_args( fStorage );
	 //return thrust::get<0>(empties);
	 return thrust::get<0>(fStorage ).empty();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 void multivector<Vector,Allocator,T...>::resize(size_t size)
 {
	 detail::resize_call_args( fStorage, size );
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 void multivector<Vector,Allocator,T...>::clear()
 {
	 detail::clear_call_args(fStorage);
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }


 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 void multivector<Vector,Allocator,T...>::shrink_to_fit()
 {
	 detail::shrink_to_fit_call_args(fStorage);
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 void multivector<Vector,Allocator,T...>::reserve(size_t size)
 {
	 detail::reserve_call_args(fStorage, size );
	 this->fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
	 this->fReverseBegin =thrust::make_zip_iterator( detail::rbegin_call_args(fStorage) );
	 this->fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
	 this->fConstReverseBegin =thrust::make_zip_iterator( detail::crbegin_call_args(fStorage) );
	 this->fTBegin =  detail::begin_call_args(fStorage) ;
	 this->fTReverseBegin =  detail::rbegin_call_args(fStorage) ;
	 this->fTConstBegin   =  detail::cbegin_call_args(fStorage) ;
	 this->fTConstReverseBegin =  detail::crbegin_call_args(fStorage) ;
	 this->fSize = thrust::get<0>(fStorage ).size();
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename 	multivector<Vector,Allocator,T...>::reference_tuple
 multivector<Vector,Allocator,T...>::front()
 {
	 return  detail::front_call_args(fStorage);
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_reference_tuple
 multivector<Vector,Allocator,T...>::front() const
 {
	 return  detail::front_call_args(fStorage);
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::reference_tuple
 multivector<Vector,Allocator,T...>::back()
 {
	 return  detail::back_call_args(fStorage);
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_reference_tuple
 multivector<Vector,Allocator,T...>::back() const
 {
	 return  detail::back_call_args(fStorage);
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::iterator
 multivector<Vector,Allocator,T...>::begin()
 {
	 return	fBegin;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::iterator
 multivector<Vector,Allocator,T...>::end()
 {
	 return	fBegin+fSize;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_iterator
 multivector<Vector,Allocator,T...>::cbegin() const
 {
	 return	fConstBegin;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_iterator
 multivector<Vector,Allocator,T...>::cend() const
 {
	 return	fConstBegin+fSize;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::reverse_iterator
 multivector<Vector,Allocator,T...>::rbegin()
 {
	 return	fReverseBegin;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::reverse_iterator
 multivector<Vector,Allocator,T...>::rend()
 {
	 return	fReverseBegin+fSize;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename 	multivector<Vector,Allocator,T...>::const_reverse_iterator
 multivector<Vector,Allocator,T...>::crbegin() const
 {
	 return	fConstReverseBegin;
 }

 template< template<typename...> class Vector,
 template<typename...> class Allocator,
 typename ...T>
 typename multivector<Vector,Allocator,T...>::const_reverse_iterator
 multivector<Vector,Allocator,T...>::crend() const
 {
	 return	fConstReverseBegin+fSize;
 }


 //single container iterators
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
