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
 * Weights.h
 *
 *  Created on: 11/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WEIGHTS_H_
#define WEIGHTS_H_

namespace hydra {

template<typename Iterator>
class Weights{

public:

	Weights(Iterator first, Iterator last):
		fSumW(0),
		fSumW2(0),
		fBegin(first),
		fEnd(last)
	{}

	Weights( Weights<Iterator> const& other):
		fSumW(other.GetSumW()),
		fSumW2(other.GetSumW2()),
		fBegin(other.begin()),
		fEnd(other.end())
	{}
	
	Weights<Iterator>& operator=( Weights<Iterator> const& other){
	
		if(this==&other)return *this;
		fSumW  = other.GetSumW();
		fSumW2 = other.GetSumW2();
		fBegin = other.begin();
		fEnd   = other.end();
		return *this;
	}

	Iterator begin(){
		return fBegin;
	}

	Iterator end(){
		return fEnd;
	}

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

	GReal_t GetSumW() const {
		return fSumW;
	}

	void SetSumW(GReal_t sumW) {
		fSumW = sumW;
	}

	GReal_t GetSumW2() const {
		return fSumW2;
	}

	void SetSumW2(GReal_t sumW2) {
		fSumW2 = sumW2;
	}

private:
	
	GReal_t  fSumW;
	GReal_t  fSumW2;
	Iterator fBegin;
	Iterator fEnd;

};


}  // namespace hydra



#endif /* WEIGHTS_H_ */
