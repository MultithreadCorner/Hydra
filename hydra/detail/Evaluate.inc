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
 * Evaluate.inc
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef EVALUATE_INC_
#define EVALUATE_INC_

namespace hydra {

namespace detail {

template<typename ReturnType,  typename T >
struct process
{

	process(T const& f):
		fTuple(f)
	{}

	template<typename ArgType  >
	__hydra_host__ __hydra_device__ ReturnType operator()(ArgType&& x)
	{
		return detail::invoke<T,ArgType>(x,fTuple);
	}

	template<typename ArgType1, typename ArgType2  >
	__hydra_host__ __hydra_device__ ReturnType operator()(ArgType1&& x, ArgType2&& y )
	{
		return detail::invoke<T,ArgType1,ArgType2>(x,y,fTuple);
	}

	T fTuple;

};



template<typename T, template<typename, typename...> class V,  size_t N>
inline 	size_t get_size(std::array<V<T>*, N>const & Array)
{
	bool same_size=true;

	for ( size_t n=1; n< N; n++ )
	{
		if( Array[n-1]->size() != Array[n]->size() )
		{
			same_size = false;
			break;
		}
	}
	return same_size ? Array[0]->size() : 0;
}

}/* namespace detail */


}// namespace hydra

#endif /* EVALUATE_INC_ */
