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

#ifndef _POINTVECTOR_H_
#define _POINTVECTOR_H_

//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/experimental/Point.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/experimental/multivector.h>
//thrust


#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace hydra
{

namespace experimental {
/**
 * PointVector wraps a thrust::vector<Points<T,N>> and provide methods
 *
 */
template<typename T, hydra::detail::Backend BACKEND>
class PointVector;

template<hydra::detail::Backend BACKEND, typename T, size_t N, bool V_ERROR, bool C_ERROR >
class PointVector< Point<T, N, V_ERROR, C_ERROR>,  BACKEND >
{

public:

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;

    typedef Point<T, N, V_ERROR, C_ERROR> point_t;
    typedef typename point_t::type super_t;
    typedef T cell_type;
    typedef typename system_t::template container< cell_type > column_type;

	typedef typename system_t::template container< super_t > prototype_t;

	typedef hydra::experimental::multivector<prototype_t> data_t;

	typedef typename column_type::iterator column_iterator;
	typedef typename column_type::const_iterator const_column_iterator;


	typedef typename data_t::const_iterator const_iterator;
	typedef typename data_t::iterator iterator;
	typedef typename data_t::reference_tuple reference;
	typedef typename data_t::const_reference_tuple const_reference;

	__host__
	PointVector():
		fData(data_t()){}

	__host__
	PointVector(size_t n):
		fData(data_t(n, point_t().GetData())) {}
	

	__host__
	PointVector( PointVector<Point<T, N, V_ERROR, C_ERROR>, BACKEND> const& other):
		fData(other.GetData()){}

	template<hydra::detail::Backend  BACKEND2 >
	__host__
	PointVector( PointVector<Point<T, N, V_ERROR, C_ERROR>, BACKEND2> const& other):
	fData(other.GetData()){}
	
	

	/**
	 * get access to the underlying container
	 */
	__host__
	const data_t& GetData() const { return fData; }

	__host__
	data_t& GetData(){ return fData; }


	/**Todo
	 * add simple point
	 */

	/**
	 * Add a new point
	 */
	__host__
	void AddPoint( point_t const& point)
	{
		fData.push_back(point);
	}


	__host__
	point_t GetPoint(size_t i) const
	{
		return point_t(fData[i]);
	}

	__host__
	size_t Size(){ return fData.size(); }

	/*
	 * stl like inteface
	 */

	__host__
	size_t size() const { return fData.size(); }

	/**
	 *  constant iterator access
	 */
	__host__
	const_iterator begin() const { return fData.cbegin(); }
	__host__
	const_iterator end() const { return fData.cend(); }

	__host__
	const_iterator cbegin() const { return fData.cbegin(); }
	__host__
	const_iterator cend() const { return fData.cend(); }



	/**
	 *   non-const iterator access
	 */
	__host__
	iterator begin() { return fData.begin(); }
	__host__
	iterator end()   { return fData.end(); }
	
	/**
	 *  subscript operator
	 */
	__host__
    const_reference operator[] (size_t i)  const { return (fData.begin()[i]); }

	__host__
	reference operator[] (size_t i) { return (fData.begin()[i]); }



	/**
	 * size
	 */


private:
	
	data_t fData;
	

};


template<size_t I, typename T, size_t N, hydra::detail::Backend   BACKEND>
auto CoordBegin(PointVector<Point<T,N,false,false>,BACKEND >& container )
-> typename PointVector<Point<T,N,false,false>, BACKEND>::column_iterator
{
	auto begin= container.GetData().template vbegin<I+2>();
	return begin;
}
/*
template<size_t I, typename T, size_t N, unsigned int BACKEND>
auto GetCoordinateBegin(PointVector<Point<T,N,false,false>,
		BACKEND >& container )
-> decltype( container.GetData().template vbegin<I+2>())
{
	auto begin= container.GetData().template vbegin<I+2>();
	return begin;
}
*/

template<size_t I, typename T, size_t N, hydra::detail::Backend   BACKEND>
auto CoordEnd(PointVector<Point<T,N,false,false>,  BACKEND >& container )
-> typename PointVector<Point<T,N,false,false>, BACKEND>::column_iterator
{
	auto begin= container.GetData().template vend<I+2>();
	return begin;
}

template<size_t I, typename T, size_t N, bool CERROR,hydra::detail::Backend  BACKEND>
auto CoordBegin(PointVector<Point<T,N,true,CERROR>,  BACKEND >& container )
-> decltype(container.GetData().template vbegin<I+3>())
{
	auto begin= container.GetData().template vbegin<I+3>();
	return begin;
}

template<size_t I, typename T, size_t N, bool CERROR, hydra::detail::Backend  BACKEND>
auto CoordEnd(PointVector<Point<T,N,true,CERROR>,  BACKEND >& container )
-> decltype( container.GetData().template vend<I+3>())
{
	auto begin= container.GetData().template vend<I+3>();
	return begin;
}

}  // namespace experimental

}//namespace hydra

#endif /* POINTVECTOR_H_ */
