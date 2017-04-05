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
 * PointVector.h
 *
 *  Created on: 15/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef POINTVECTOR_H_
#define POINTVECTOR_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Point.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/TypeTraits.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace hydra
{
/**
 * PointVector wraps a thrust::vector<Points<T,N>> and provide methods
 *
 */


template<  unsigned int BACKEND=0,typename T=GReal_t, size_t N=1, bool V_ERROR=false, bool C_ERROR=false >
class PointVector{

public:

	typedef detail::BackendTraits<BACKEND> system_t;


	typedef typename system_t::template container< Point<T, N, V_ERROR, C_ERROR> > type;
	typedef Point<T, N, V_ERROR, C_ERROR> value_type;
	typedef typename type::const_iterator const_iterator;
	typedef typename type::iterator iterator;

	__host__
	PointVector():
		fPoints(type()){}

	__host__
	PointVector(size_t n):
		fPoints(type(n)) {}
	

	template<unsigned int BACKEND2 >
	__host__
	PointVector( PointVector<BACKEND2, T, N, V_ERROR, C_ERROR > const& other):
	fPoints(other.GetPoints()){}
	
	
	~PointVector(){};

	/**
	 * get access to the underlying container
	 */
	__host__
	const type& GetPoints() const { return fPoints; }

	__host__
	type& GetPoints() { return fPoints; }


	/**Todo
	 * add simple point
	 */

	/**
	 * Add a new point
	 */
	__host__
	void AddPoint( value_type const& point)
	{
		fPoints.push_back(point);
	}

	__host__
	value_type const& GetPoint(size_t i) const
	{
		return fPoints[i];
	}


	/**
	 *  constant iterator access
	 */
	__host__
	const_iterator begin() const { return fPoints.begin(); }
	__host__
	const_iterator end() const { return fPoints.begin()+fPoints.size(); }

	/**
	 *   non-const iterator access
	 */
	__host__
	iterator begin() { return fPoints.begin(); }
	__host__
	iterator end()   { return fPoints.end(); }
	
	/**
	 *   access to the point
	 */
	__host__
	const value_type& operator[] (size_t i)  const { return fPoints[i]; }
	__host__
	value_type& operator[] (size_t i) { return fPoints[i]; }


	/**
	 * size
	 */
	__host__
	size_t Size(){ return fPoints.size(); }

private:
	
	type fPoints;
	

};

}//namespace hydra

#endif /* POINTVECTOR_H_ */
