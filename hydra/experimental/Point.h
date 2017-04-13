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

//thrust
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


namespace hydra {
namespace experimental {

template<typename T, size_t DIM, bool VALUE_ERROR=false, bool CO0RDINATE_ERROR=false>
struct Point;

template<typename T, size_t DIM, bool VALUE_ERROR, bool CO0RDINATE_ERROR>
__host__ __device__
inline bool operator==(Point<T,DIM,VALUE_ERROR,CO0RDINATE_ERROR> const& lhs,
		Point<T,DIM,VALUE_ERROR,CO0RDINATE_ERROR> const& rhs)
{ return lhs.GetData()==rhs.GetData(); }





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
