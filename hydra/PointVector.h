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
#include <hydra/Point.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/multivector.h>
//thrust


#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace hydra {


template<typename T, hydra::detail::Backend BACKEND>
class PointVector;

/**
 * @ingroup generic
 * @brief PointVector implements storage for multidimensional datasets used in fits etc.
 *
 * This is an iterable container and data is stored using an SoA layout, but the access
 * mimics a AoS. A N-dimensional PointVector instance can be understood as a table with N columns
 * and many (millions) rows. The user can iterates over the table accessing the whole row, which is returned as
 * a tuple, or accessing entries in a given column. Column-based access is away faster than
 * access a row and then get the desired element from the tuple.
 *
 * @tparam    Point<T, N, V_ERROR, C_ERROR> is the type of point the PointVector will store.
 * @tparam    Backend specifies the memory space where the information will be stored.
 */
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

	typedef multivector<prototype_t> data_t;

	typedef typename column_type::iterator column_iterator;
	typedef typename column_type::const_iterator const_column_iterator;


	typedef typename data_t::const_iterator const_iterator;
	typedef typename data_t::iterator iterator;
	typedef typename data_t::reference_tuple reference;
	typedef typename data_t::const_reference_tuple const_reference;

	PointVector():
		fData(data_t()){}

	PointVector(size_t n):
		fData(data_t(n, point_t().GetData())) {}
	

	PointVector( PointVector<Point<T, N, V_ERROR, C_ERROR>, BACKEND> const& other):
		fData(other.GetData()){}

	template<hydra::detail::Backend  BACKEND2 >
	PointVector( PointVector<Point<T, N, V_ERROR, C_ERROR>, BACKEND2> const& other):
	fData(other.GetData()){}
	
	

	/**
	 *@brief  get access to the underlying container
	 */
	const data_t& GetData() const { return fData; }

	data_t& GetData(){ return fData; }


	/**
	 * @brief Add a new point
	 */
	void AddPoint( point_t const& point)
	{
		fData.push_back(point);
	}


	/**
	 *
	 */
	point_t GetPoint(size_t i) const
	{
		return point_t(fData[i]);
	}


	size_t Size(){ return fData.size(); }

	/*
	 * stl like inteface
	 */

	__host__
	size_t size() const { return fData.size(); }

	/**
	 *  constant iterator access
	 */
	const_iterator begin() const { return fData.cbegin(); }
	const_iterator end() const { return fData.cend(); }
	const_iterator cbegin() const { return fData.cbegin(); }
	const_iterator cend() const { return fData.cend(); }



	/**
	 *   non-const iterator access
	 */
	iterator begin() { return fData.begin(); }
	iterator end()   { return fData.end(); }
	
	/**
	 *  subscript operator
	 */
	const_reference operator[] (size_t i)  const { return (fData.begin()[i]); }
		  reference operator[] (size_t i) { return (fData.begin()[i]); }

private:
	data_t fData;
};


template<size_t I, typename T, size_t N, hydra::detail::Backend   BACKEND>
hydra::pair<typename PointVector<Point<T,N,false,false>, BACKEND>::column_iterator,
typename PointVector<Point<T,N,false,false>, BACKEND>::column_iterator>
get_column(PointVector<Point<T,N,false,false>,BACKEND >& container )
{
	return hydra::make_pair(
			container.GetData().template vbegin<I>(),
			container.GetData().template vend<I>());
}


template<size_t I, typename T, size_t N, hydra::detail::Backend   BACKEND>
hydra::pair<typename PointVector<Point<T,N,false,false>, BACKEND>::const_column_iterator,
typename PointVector<Point<T,N,false,false>, BACKEND>::const_column_iterator>
get_column(PointVector<Point<T,N,false,false>,BACKEND > const& container )
{
	return hydra::make_pair(
			container.GetData().template vcbegin<I>(),
			container.GetData().template vcend<I>());
}





}//namespace hydra

#endif /* POINTVECTOR_H_ */
