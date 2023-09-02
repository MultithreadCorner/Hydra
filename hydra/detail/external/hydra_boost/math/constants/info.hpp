//  Copyright John Maddock 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma once
#endif

#ifndef HYDRA_BOOST_MATH_CONSTANTS_INFO_INCLUDED
#define HYDRA_BOOST_MATH_CONSTANTS_INFO_INCLUDED

#include <hydra/detail/external/hydra_boost/math/constants/constants.hpp>
#include <iostream>
#include <iomanip>
#include <typeinfo>

namespace hydra_boost{ namespace math{ namespace constants{

   namespace detail{

      template <class T>
      const char* nameof(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(T))
      {
         return typeid(T).name();
      }
      template <>
      const char* nameof<float>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(float))
      {
         return "float";
      }
      template <>
      const char* nameof<double>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(double))
      {
         return "double";
      }
      template <>
      const char* nameof<long double>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(long double))
      {
         return "long double";
      }

   }

template <class T, class Policy>
void print_info_on_type(std::ostream& os = std::cout HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(T) HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(Policy))
{
   using detail::nameof;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif
   os <<
      "Information on the Implementation and Handling of \n"
      "Mathematical Constants for Type " << nameof<T>() <<
      "\n\n"
      "Checking for std::numeric_limits<" << nameof<T>() << "> specialisation: " <<
      (std::numeric_limits<T>::is_specialized ? "yes" : "no") << std::endl;
   if(std::numeric_limits<T>::is_specialized)
   {
      os <<
         "std::numeric_limits<" << nameof<T>() << ">::digits reports that the radix is " << std::numeric_limits<T>::radix << ".\n";
      if (std::numeric_limits<T>::radix == 2)
      {
      os <<
         "std::numeric_limits<" << nameof<T>() << ">::digits reports that the precision is \n" << std::numeric_limits<T>::digits << " binary digits.\n";
      }
      else if (std::numeric_limits<T>::radix == 10)
      {
         os <<
         "std::numeric_limits<" << nameof<T>() << ">::digits reports that the precision is \n" << std::numeric_limits<T>::digits10 << " decimal digits.\n";
         os <<
         "std::numeric_limits<" << nameof<T>() << ">::digits reports that the precision is \n"
         << std::numeric_limits<T>::digits * 1000L /301L << " binary digits.\n";  // divide by log2(10) - about 3 bits per decimal digit.
      }
      else
      {
        os << "Unknown radix = " << std::numeric_limits<T>::radix << "\n";
      }
   }
   typedef typename hydra_boost::math::policies::precision<T, Policy>::type precision_type;
   if(precision_type::value)
   {
      if (std::numeric_limits<T>::radix == 2)
      {
       os <<
       "hydra_boost::math::policies::precision<" << nameof<T>() << ", " << nameof<Policy>() << " reports that the compile time precision is \n" << precision_type::value << " binary digits.\n";
      }
      else if (std::numeric_limits<T>::radix == 10)
      {
         os <<
         "hydra_boost::math::policies::precision<" << nameof<T>() << ", " << nameof<Policy>() << " reports that the compile time precision is \n" << precision_type::value << " binary digits.\n";
      }
      else
      {
        os << "Unknown radix = " << std::numeric_limits<T>::radix <<  "\n";
      }
   }
   else
   {
      os <<
         "hydra_boost::math::policies::precision<" << nameof<T>() << ", Policy> \n"
         "reports that there is no compile type precision available.\n"
         "hydra_boost::math::tools::digits<" << nameof<T>() << ">() \n"
         "reports that the current runtime precision is \n" <<
         hydra_boost::math::tools::digits<T>() << " binary digits.\n";
   }

   typedef typename construction_traits<T, Policy>::type construction_type;

   switch(construction_type::value)
   {
   case 0:
      os <<
         "No compile time precision is available, the construction method \n"
         "will be decided at runtime and results will not be cached \n"
         "- this may lead to poor runtime performance.\n"
         "Current runtime precision indicates that\n";
      if(hydra_boost::math::tools::digits<T>() > max_string_digits)
      {
         os << "the constant will be recalculated on each call.\n";
      }
      else
      {
         os << "the constant will be constructed from a string on each call.\n";
      }
      break;
   case 1:
      os <<
         "The constant will be constructed from a float.\n";
      break;
   case 2:
      os <<
         "The constant will be constructed from a double.\n";
      break;
   case 3:
      os <<
         "The constant will be constructed from a long double.\n";
      break;
   case 4:
      os <<
         "The constant will be constructed from a string (and the result cached).\n";
      break;
   default:
      os <<
         "The constant will be calculated (and the result cached).\n";
      break;
   }
   os << std::endl;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

template <class T>
void print_info_on_type(std::ostream& os = std::cout HYDRA_BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(T))
{
   print_info_on_type<T, hydra_boost::math::policies::policy<> >(os);
}

}}} // namespaces

#endif // HYDRA_BOOST_MATH_CONSTANTS_INFO_INCLUDED
