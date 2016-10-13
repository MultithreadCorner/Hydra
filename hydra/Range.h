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
 * Range.h
 *
 *  Created on: 10/09/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

namespace hydra {

/**
 * Range helper class to use with hydra::Eval
 */
template<typename Iterator>
struct Range
{
	typedef typename thrust::iterator_value<Iterator>::type value_type;
	typedef typename thrust::iterator_system<Iterator>::type system;

	/**
	 * ctor taking begin and end iterators
	 */
	Range(Iterator begin, Iterator end ):
		fBegin(begin),
		fEnd(end)
	{}

	/**
	 * ctor taking begin iterator and number of elements
	 * end iterator will be (begin + n)
	 */
	Range(Iterator begin, size_t n ):
			fBegin(begin),
			fEnd( begin+n)
		{}



	value_type& operator()(const size_t idx)
	{
		return *(fBegin+idx);
	}


	value_type& operator[](const size_t idx)
	{
		return *(fBegin+idx);
	}

	size_t size(){
	 return	thrust::distance(fBegin, fEnd);
	}

	const Iterator begin() const {
		return fBegin;
	}


	const Iterator end() const {
		return fEnd;
	}

	Iterator begin()  {
		return fBegin;
	}


	Iterator end()  {
		return fEnd;
	}



private:
	Iterator fBegin;
	Iterator fEnd;
};

template<typename Iterator>
Range<Iterator> make_range( Iterator begin,  Iterator end)
{ return Range<Iterator>(begin, end ); }

template<typename Iterator>
Range<Iterator> make_range( Iterator begin,  size_t n)
{ return Range<Iterator>(begin, begin+n ); }

template<typename T>
Range<typename T::iterator> make_range( T const& container )
{ return Range<typename T::iterator>(container.begin(), container.end() ); }

template<typename T>
Range<typename T::iterator> make_range( T const& container ,  size_t n)
{ return Range<typename T::iterator>(container.begin(), container.begin()+n); }


}  // namespace hydra



#endif /* RANGE_H_ */
