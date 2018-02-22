/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * GenericRange.h
 *
 *  Created on: 29/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENERICRANGE1_INL_
#define GENERICRANGE1_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>


namespace hydra {

template<typename Iterator>
class GenericRange<Iterator>{

public:
	//stl-like typedefs
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::difference_type    difference_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::value_type         value_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::pointer            pointer;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::reference          reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator>::iterator_category  iterator_category;

	GenericRange()=delete;

	GenericRange(Iterator begin, Iterator end):
		fBegin( begin),
		fEnd( end )
		{}

	GenericRange(Iterator begin,  size_t last):
			fBegin( begin),
			fEnd( begin + last )
			{}

	GenericRange(GenericRange<Iterator> const& other):
			fBegin( other.GetBegin()),
			fEnd( other.GetEnd() )
			{}

	GenericRange<Iterator>&
	operator=(GenericRange<Iterator> const& other){

		if(this==&other) return this;

		fBegin = other.GetBegin();
		fEnd = other.GetEnd();
		return this;
	}


	Iterator begin(){ return fBegin;};

	Iterator   end(){ return fEnd;};

	size_t size() { return hydra::distance(fBegin, fEnd);}

	Iterator GetBegin() const {
		return fBegin;
	}

	void SetBegin(Iterator begin) {
		fBegin = begin;
	}

	Iterator GetEnd() const {
		return fEnd;
	}

	void SetEnd(Iterator end) {
		fEnd = end;
	}

	reference  operator[](size_t i)
	{
		return fBegin[i];
	}

	const reference  operator[](size_t i) const
	{
		return fBegin[i];
	}


private:
	Iterator fBegin;
	Iterator fEnd;

};

template<typename Iterator>
GenericRange<Iterator>
make_range(Iterator begin, Iterator end ){
	return GenericRange<Iterator>( begin, end);
}

}  // namespace hydra



#endif /* GENERICRANGE_H_ */
