/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * FunctionArgument.h
 *
 *  Created on: Feb 11, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FUNCTIONARGUMENT_H_
#define FUNCTIONARGUMENT_H_

#include <hydra/detail/Config.h>

namespace hydra {

namespace detail {

template<typename Derived, typename Type>
struct FunctionArgument
{
    typedef Type   value_type;
    typedef Derived name_type;

    FunctionArgument() = default;

    __hydra_host__ __hydra_device__
    FunctionArgument(value_type x) :
     value(x)
     {}

    __hydra_host__ __hydra_device__
    FunctionArgument(FunctionArgument<Derived, Type>const& other):
     value(other)
     {}

    __hydra_host__ __hydra_device__
    FunctionArgument<Derived, Type>&
    operator=(FunctionArgument<Derived, Type>const& other)
    {
        if(this==&other) return *this;
        value = other();
        return *this;
    }

    template<typename Derived2, typename Type2,
     typename = typename std::enable_if<std::is_convertible<Type, Type2>::type
     , FunctionArgument<Derived, Type>& >::type>
    __hydra_host__ __hydra_device__
    FunctionArgument(FunctionArgument<Derived2, Type2>const& other):
     value(other())
     {}

    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value
     , FunctionArgument<Derived, Type>& >::type
    operator=(FunctionArgument<Derived2, Type2>const& other)
    {
        if(this==&other) return *this;
        value = other();
        return *this;
    }

      __hydra_host__ __hydra_device__
      explicit constexpr operator Type() const { return value; }

    __hydra_host__ __hydra_device__
    constexpr Type operator()(void) const { return value; }
    /*
    //=============================================================
    //Arithmetic operators
    //=============================================================

    template<typename Derived2, typename Type2,
      typename ReturnType=decltype(std::declval<Type>() + std::declval<Type2>())>
    __hydra_host__ __hydra_device__
    ReturnType  operator+( FunctionArgument<Derived2, Type2> const & other) const
	{
		return ReturnType(value + Type2(other) );
	}

    template<typename Derived2, typename Type2,
        typename ReturnType=decltype(std::declval<Type>() - std::declval<Type2>())>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value, ReturnType >::type
    operator-( FunctionArgument<Derived2, Type2> const & other) const
	{
        return ReturnType(value -  Type2(other) );
	}

    template<typename Derived2, typename Type2,
        typename ReturnType=decltype(std::declval<Type>() * std::declval<Type2>())>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value, ReturnType >::type
    operator*( FunctionArgument<Derived2, Type2> const & other) const
	{
        return ReturnType(value *  Type2(other) );
	}

    template<typename Derived2, typename Type2,
    typename ReturnType=decltype(std::declval<Type>() / std::declval<Type2>())>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value, ReturnType >::type
    operator/( FunctionArgument<Derived2, Type2> const & other) const
	{
        return ReturnType(value /  Type2(other) );
	}

    template<typename Derived2, typename Type2,
    typename ReturnType=decltype(std::declval<Type>() % std::declval<Type2>())>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value, ReturnType >::type
    operator%( FunctionArgument<Derived2, Type2> const & other) const
	{
        return ReturnType(value %  Type2(other) );
	}
    */
    //=============================================================
    //Compound assignment operators
    //=============================================================

    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value,
    FunctionArgument<Derived, Type>&>::type
    operator+=( FunctionArgument<Derived2, Type2> const & other)
	{
        value+=other();
		return *this;
	}

    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value,
    FunctionArgument<Derived, Type>&>::type
    operator-=( FunctionArgument<Derived2, Type2> const & other)
	{
        value-=other();
		return *this;
	}

    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value,
    FunctionArgument<Derived, Type>&>::type
    operator*=( FunctionArgument<Derived2, Type2> const & other)
	{
        value*=other();
		return *this;
	}

    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value,
    FunctionArgument<Derived, Type>&>::type
    operator/=( FunctionArgument<Derived2, Type2> const & other)
	{
        value/=other();
		return *this;
	}


    template<typename Derived2, typename Type2>
    __hydra_host__ __hydra_device__
    typename std::enable_if<std::is_convertible<Type, Type2>::value,
    FunctionArgument<Derived, Type>&>::type
    operator%=( FunctionArgument<Derived2, Type2> const & other)
	{
        value%=other();
		return *this;
	}

    private:

    value_type value;
};

}  // namespace detail


//arithmetic operators

#define HYDRA_DEFINE_ARGUMENT_OPERATOR(symbol)\
    template<typename Derived1, typename Type1,\
             typename Derived2, typename Type2,\
             typename ReturnType=decltype(std::declval<Type1>() symbol std::declval<Type2>())> \
    __hydra_host__ __hydra_device__\
    ReturnType operator symbol(detail::FunctionArgument<Derived1, Type1> const & a,\
    		                   detail::FunctionArgument<Derived2, Type2> const & b)\
	{\
		return ReturnType(static_cast<Type1>(a) symbol static_cast<Type2>(b) );\
	}\
	\
    template< typename Type1, typename Derived2, typename Type2,\
             typename ReturnType=decltype(std::declval<Type1>() symbol std::declval<Type2>())> \
    __hydra_host__ __hydra_device__\
    ReturnType operator symbol( Type1 a, detail::FunctionArgument<Derived2, Type2> const & b)\
	{\
		return ReturnType(a symbol static_cast<Type2>(b) );\
	}\
	\
	template< typename Type1, typename Derived2, typename Type2,\
			 typename ReturnType=decltype(std::declval<Type1>() symbol std::declval<Type2>())> \
	__hydra_host__ __hydra_device__\
	ReturnType operator symbol( detail::FunctionArgument<Derived2, Type2> const & b, Type1 a)\
	{\
		return ReturnType(a symbol static_cast<Type2>(b) );\
	}\


HYDRA_DEFINE_ARGUMENT_OPERATOR(+)
HYDRA_DEFINE_ARGUMENT_OPERATOR(-)
HYDRA_DEFINE_ARGUMENT_OPERATOR(*)
HYDRA_DEFINE_ARGUMENT_OPERATOR(/)
HYDRA_DEFINE_ARGUMENT_OPERATOR(%)

#ifdef HYDRA_DEFINE_ARGUMENT_OPERATOR
#undef HYDRA_DEFINE_ARGUMENT_OPERATOR
#endif //HYDRA_DEFINE_ARGUMENT_OPERATOR

}  // namespace hydra

#define declarg(NAME, TYPE )                                           \
namespace hydra {	namespace arguments  { 							   \
																	   \
struct NAME : detail::FunctionArgument<NAME, TYPE>                     \
{                                                                      \
 typedef  detail::FunctionArgument<NAME, TYPE>  super_type;            \
                                                                       \
  NAME()=default;                                                      \
                                                                       \
  NAME( TYPE x):                                                       \
     super_type(x)                                                     \
     {}                                                                \
                                                                       \
  NAME( NAME const& other):                                            \
    super_type(other)                                                  \
    {}                                                                 \
                                                                       \
  template<typename T,                                                 \
       typename = typename std::enable_if<                             \
        std::is_base_of< detail::FunctionArgument<T, TYPE>, T>::value, \
        void >::type >                                                 \
  NAME( T const& other):                                               \
    super_type(other)                                                  \
    {}                                                                 \
                                                                       \
  NAME& operator=(NAME const& other)                                   \
  {                                                                    \
        if(this==&other)                                               \
         return *this;                                                 \
                                                                       \
        super_type::operator=(other);                                  \
                                                                       \
        return *this;                                                  \
  }                                                                    \
                                                                       \
  template<typename T>                                                 \
  typename std::enable_if<                                             \
        std::is_base_of< detail::FunctionArgument<T, TYPE>, T>::value, \
        NAME& >::type                                                  \
  operator=(T const& other)                                            \
  {                                                                    \
        if(this==&other)                                               \
         return *this;                                                 \
                                                                       \
        super_type::operator=(other);                                  \
                                                                       \
        return *this;                                                  \
  }                                                                    \
                                                                       \
};                                                                     \
                                                                       \
} /*namespace arguments*/ }/*namespace hydra*/                         \


#endif /* FUNCTIONARGUMENT_H_ */
