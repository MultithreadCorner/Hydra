//  (C) Copyright John Maddock 2005-2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_STATS_INCLUDED
#define HYDRA_BOOST_MATH_TOOLS_STATS_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <cstdint>
#include <cmath>
#include <hydra/detail/external/hydra_boost/math/tools/precision.hpp>

namespace hydra_boost{ namespace math{ namespace tools{

template <class T>
class stats
{
public:
   stats()
      : m_min(tools::max_value<T>()),
        m_max(-tools::max_value<T>()),
        m_total(0),
        m_squared_total(0)
   {}
   void add(const T& val)
   {
      if(val < m_min)
         m_min = val;
      if(val > m_max)
         m_max = val;
      m_total += val;
      ++m_count;
      m_squared_total += val*val;
   }
   T min HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION()const{ return m_min; }
   T max HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION()const{ return m_max; }
   T total()const{ return m_total; }
   T mean()const{ return m_total / static_cast<T>(m_count); }
   std::uintmax_t count()const{ return m_count; }
   T variance()const
   {
      HYDRA_BOOST_MATH_STD_USING

      T t = m_squared_total - m_total * m_total / m_count;
      t /= m_count;
      return t;
   }
   T variance1()const
   {
      HYDRA_BOOST_MATH_STD_USING

      T t = m_squared_total - m_total * m_total / m_count;
      t /= (m_count-1);
      return t;
   }
   T rms()const
   {
      HYDRA_BOOST_MATH_STD_USING

      return sqrt(m_squared_total / static_cast<T>(m_count));
   }
   stats& operator+=(const stats& s)
   {
      if(s.m_min < m_min)
         m_min = s.m_min;
      if(s.m_max > m_max)
         m_max = s.m_max;
      m_total += s.m_total;
      m_squared_total += s.m_squared_total;
      m_count += s.m_count;
      return *this;
   }
private:
   T m_min, m_max, m_total, m_squared_total;
   std::uintmax_t m_count{0};
};

} // namespace tools
} // namespace math
} // namespace hydra_boost

#endif
