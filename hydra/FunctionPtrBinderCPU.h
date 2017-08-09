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
 * FunctionPtrBinderCPU.h
 *
 *  Created on: 01/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FUNCTIONPTRBINDERCPU_H_
#define FUNCTIONPTRBINDERCPU_H_


#include <hydra/Function.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>

namespace hydra {

template< size_t N=0>
struct FunctionPtrBinderCPU:public BaseFunctor<FunctionPtrBinderCPU<N>, GReal_t, N>
{

	typedef double (*FunctionPtr_t)(unsigned int n, GReal_t*, GReal_t*);
	typedef std::function<double(unsigned int n, GReal_t*, GReal_t*)> StdFunction_t;

	FunctionPtrBinderCPU( FunctionPtr_t func ):
	 fStdFunction( func )
	{}

	FunctionPtrBinderCPU( StdFunction_t func ):
		fStdFunction( func )
	{}


	inline FunctionPtrBinderCPU(FunctionPtrBinderCPU<N> const& other):
	BaseFunctor< FunctionPtrBinderCPU<N>, GReal_t,N>(other),
	fStdFunction( other.GetStdFunction() )
	{ }


	inline FunctionPtrBinderCPU<N>& operator=(FunctionPtrBinderCPU<N>  const& other)
	{
		if(this == &other) return *this;

		BaseFunctor< FunctionPtrBinderCPU<N>, GReal_t,N >::operator=(other);
		fStdFunction = other.GetStdFunction();

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x)
	{
		GReal_t pars[N]{};
#if (N>0)
		#pragma unroll (N)
#endif
		for(size_t i=0;i<N;i++)
			pars[i]=this->GetParameter(i);

		return fStdFunction(N, pars, x);

	}

	StdFunction_t GetStdFunction() const {
		return fStdFunction;
	}

	void SetStdFunction(StdFunction_t stdFunction) {
		fStdFunction = stdFunction;
	}



private:
	StdFunction_t fStdFunction;
};

}  // namespace hydra



#endif /* FUNCTIONPTRBINDERCPU_H_ */
