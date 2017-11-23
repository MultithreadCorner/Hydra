/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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

namespace hydra {

template<typename Iterator>
class GenericRange<Iterator>{

public:

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

	auto operator[](size_t i)
	-> decltype(begin()[0] )
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
