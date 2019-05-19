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
 * Permute.h
 *
 *  Created on: 26/03/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PERMUTE_H_
#define PERMUTE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/sort.h>
#include <hydra/detail/external/thrust/memory.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/swap.h>

namespace hydra {

namespace detail {

template <class Iterator>
__hydra_host__ __hydra_device__
void iter_swap(Iterator first, Iterator second)
{

	HYDRA_EXTERNAL_NS::thrust::swap(*HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(first), *HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(second));
}

template <class Iterator>
__hydra_host__ __hydra_device__
Iterator rotate(Iterator first, Iterator n_first, Iterator last)
{

using hydra::detail::iter_swap;

    if(first == n_first) return last;
    if(n_first == last) return first;

    Iterator next = n_first;

    do {
    	iter_swap(first++, next++);
        if (first == n_first) n_first = next;
    }
    while (next != last);

    Iterator ret = first;

    for(next = n_first; next != last; ) {
    	iter_swap( first++, next++);
        if(first == n_first) n_first = next;
        else if(next == last) next = n_first;
    }

    return ret;
}

template<typename Iterator, typename Integer, typename Comparator>
typename std::enable_if<std::is_integral<Integer>::value, void>::type
__hydra_host__ __hydra_device__
nth_permutation(Iterator begin, Iterator end, Integer n, Comparator comp)
{
  HYDRA_EXTERNAL_NS::thrust::sort(begin, end, comp);

  const Integer k(end - begin);

  Integer radix = Integer(1);
  for(Integer i = Integer(2); i < k; ++i)
    radix *= i;

  for(Integer i = 0; i < k; ++i)
  {

    Integer digit = n / radix;
    rotate(begin, begin + digit, begin + digit + 1);
    ++ begin;
    n %= radix;
    if (i + 1 != k)
      radix /= (k - i - 1);
  }
}



}  // namespace detail

} // namespace hydra


#endif /* PERMUTE_H_ */
