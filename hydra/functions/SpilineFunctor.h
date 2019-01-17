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
 * SpilineFunctor.h
 *
 *  Created on: 22/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPILINEFUNCTOR_H_
#define SPILINEFUNCTOR_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Spiline.h>
#include <hydra/Placeholders.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/execution_policy.h>
#include <hydra/detail/external/thrust/binary_search.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <math.h>
#include <algorithm>
#include <memory>

namespace hydra {

/**
 * \class SpilineFunctor
 *
 * A simple method for a one—dimensional
interpolation on a given set of data points (xi, yi). In each interval
(xi,x,-+1) the interpolation function is assumed to be a third-
order polynomial passing through the data points. The slope at
each grid point is determined in such a way as to guarantee
a monotonic behavior of the interpolating function. The result
is a smooth curve with continuous ﬁrst-order derivatives that
passes through any given set of data points without spurious
oscillations. Local extrema can occur only at grid points where
they are given by the data, but not in between two adjacent
grid points. The method gives exact results if the data points
correspond to a second-order polynomial.

Reference: M. Steffen, Astron. Astrophys. 239, 443—450 (1990).
*/
template<typename Iterator1, typename Iterator2, unsigned int ArgIndex=0>
class SpilineFunctor: public BaseFunctor<SpilineFunctor<Iterator1, Iterator2, ArgIndex>,
typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type, 0>
{

	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<Iterator2>::value_type return_type;

public:

	SpilineFunctor() = delete;

	SpilineFunctor( Iterator1 first, Iterator1 last, Iterator2 output ):
		BaseFunctor<SpilineFunctor<Iterator1, Iterator2, ArgIndex>, return_type, 0>(),
		fSize(HYDRA_EXTERNAL_NS::thrust::distance(first, last)),
		fX(first),
		fY(output)
		{}

	__hydra_host__ __hydra_device__
	SpilineFunctor(SpilineFunctor<Iterator1, Iterator2, ArgIndex> const& other ):
	BaseFunctor<SpilineFunctor<Iterator1, Iterator2, ArgIndex>, return_type, 0>(other),
	fSize(other.GetSize()),
	fX(other.GetX()),
	fY(other.GetY())
	{ }

	__hydra_host__ __hydra_device__ inline
	SpilineFunctor<Iterator1, Iterator2, ArgIndex>&
	operator=(SpilineFunctor<Iterator1, Iterator2, ArgIndex> const& other )
	{
		if(this == &other) return *this;

		BaseFunctor<SpilineFunctor<Iterator1, Iterator2, ArgIndex>, return_type, 0>::operator=(other);

		fSize=other.GetSize();
		fX=other.GetX();
		fY=other.GetY();

		return *this;
	}

	__hydra_host__ __hydra_device__
	size_t GetSize() const
	{
		return fSize;
	}

	__hydra_host__ __hydra_device__
	Iterator1 GetX() const
	{
		return fX;
	}

	__hydra_host__ __hydra_device__
	Iterator2 GetY() const
	{
		return fY;
	}

	template<typename Type>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int n, Type*x)  const {

		GReal_t X  = x[ArgIndex];

		Iterator1 fXN = fX + fSize;

		GReal_t r = spiline(fX,  fX + fSize, fY,  X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}

	template<typename Type>
	__hydra_host__ __hydra_device__
	inline double Evaluate(Type x)  const {

		GReal_t X  = hydra::get<ArgIndex>(x); //mass

		GReal_t r = spiline(fX, fX + fSize,	fY,  X);

		return  CHECK_VALUE( r, "r=%f",r) ;
	}


private:


	size_t fSize;
	Iterator1 fX;
	Iterator2 fY;
};


template<typename Iterator1, typename Iterator2, unsigned int I=0>
inline SpilineFunctor<Iterator1, Iterator2, I>
make_spiline(Iterator1 first, Iterator1 last, Iterator2 output,
		placeholders::placeholder<I> arg_index = placeholders::_0  ){

	return SpilineFunctor<Iterator1, Iterator2, I>( first, last, output);
}

template<typename Iterable1, typename Iterable2, unsigned int I=0,
         typename Iterator1=decltype(std::declval<Iterable1>().begin()),
         typename Iterator2=decltype(std::declval<Iterable2>().begin())>
inline typename std::enable_if<
          hydra::detail::is_iterable<Iterable1>::value &&
          hydra::detail::is_iterable<Iterable2>::value,
          SpilineFunctor<Iterator1, Iterator2, I> >::type
make_spiline(Iterable1&& x, Iterable2&& y,	placeholders::placeholder<I> arg_index = placeholders::_0  ){

	return SpilineFunctor<Iterator1, Iterator2, I>( std::forward<Iterable1>(x).begin(),
			std::forward<Iterable1>(x).end(), std::forward<Iterable2>(y).begin());
}



}  // namespace hydra



#endif /* SPILINEFUNCTOR_H_ */
