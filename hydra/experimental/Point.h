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
 * Point.h
 *
 *  Created on: 14/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef _POINT_H_
#define _POINT_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <ostream>
//std
#include <array>
#include <cmath>
//thrust
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


namespace hydra {
namespace experimental {

template<typename T, size_t DIM, bool VALUE_ERROR=false, bool CO0RDINATE_ERROR=false>
struct Point;

//operators
template<typename T, size_t DIM, bool VALUE_ERROR, bool CO0RDINATE_ERROR>
__host__ __device__
inline bool operator==(Point<T,DIM,VALUE_ERROR,CO0RDINATE_ERROR> const& lhs,
		Point<T,DIM,VALUE_ERROR,CO0RDINATE_ERROR> const& rhs)
{ return lhs.GetData()==rhs.GetData(); }


//operator+
template<typename T , size_t N>
__host__ __device__ inline
Point<T,N,false,false>
operator+(Point<T,N,false,false> const& point1,
		Point<T,N,false,false> const& point2)
{
	// typedef typename detail::tuple_type<N, T>::type type;

	 Point<T,N,false,false> point;//(type(), 0);

	 point.SetWeight( point1.GetWeight() + point2.GetWeight());
	 point.SetWeight2( point1.GetWeight2() + point2.GetWeight2());

	 point.SetCoordinate(hydra::operator+(point1.GetCoordinates() , point2.GetCoordinates())  );

	 return point ;

}



//output stream operators
template<typename T , size_t N>
__host__ __device__ inline
Point<T,N,true,false>
operator+(Point<T,N,true,false> const& point1,
		Point<T,N,true,false> const& point2)
{

	 Point<T,N,true,false> point;

	 point.SetWeight( point1.GetWeight() + point2.GetWeight());
	 point.SetWeight2( point1.GetWeight2() + point2.GetWeight2());
	 point.SetError( sqrt(  point1.GetError()*point1.GetError() + point2.GetError()*point2.GetError() ));
	 point.SetCoordinate( hydra::operator+(point1.GetCoordinates() , point2.GetCoordinates()) );

	 return point ;

}


//output stream operators
template<typename T , size_t N>
__host__ __device__ inline
Point<T,N,true,true>
operator+(Point<T,N,true,true> const& point1,
		Point<T,N,true,true> const& point2)
{
	 //typedef typename detail::tuple_type<N, T>::type type;

	 Point<T,N,true,true> point;//(type(), 0);

	 point.SetWeight( point1.GetWeight() + point2.GetWeight());
	 point.SetWeight2( point1.GetWeight2() + point2.GetWeight2());
	 point.SetError( sqrt(  point1.GetError()*point1.GetError() + point2.GetError()*point2.GetError() ));
     point.SetCoordinates( hydra::operator+(point1.GetCoordinates() , point2.GetCoordinates())  );

     auto errors1Sq =  hydra::operator*(point1.GetCoordinateErrors() , point1.GetCoordinateErrors());
	 auto errors2Sq =  hydra::operator*(point2.GetCoordinateErrors() , point2.GetCoordinateErrors());

	 auto errorSq   = errors1Sq + errors2Sq;

	 auto Sqrt = [ = ] __host__ __device__ ( T x ){ return sqrt(x); };

	 point.SetCoordinatesErrors(  hydra::detail::callOnTuple( [ = ] __host__ __device__
			 ( T x ){ return sqrt(x); }  ,  errorSq));

	 return point ;

}


//output stream operators
template<typename T, size_t DIM, bool VALUE_ERROR, bool CO0RDINATE_ERROR>
__host__
inline std::ostream& operator<<(std::ostream& os, Point<T,DIM,VALUE_ERROR,CO0RDINATE_ERROR > const& point){

	return os<<point.GetData() ;
}

}  // namespace experimental

} // namespace hydra

#include <hydra/experimental/detail/Point1.inl>
#include <hydra/experimental/detail/Point2.inl>
#include <hydra/experimental/detail/Point3.inl>


#endif /* POINT_H_ */
