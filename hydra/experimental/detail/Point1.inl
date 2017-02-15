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
 * Point1.inl
 *
 *  Created on: 15/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef POINT1_INL_
#define POINT1_INL_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>

#include <hydra/detail/utility/Arithmetic_Tuple.h>
//std
#include <array>

//thrust
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace hydra {

namespace experimental {


template<typename T, size_t DIM=1>
struct Point<T, DIM, false, false>
{
	constexpr static size_t N = DIM +2;
	constexpr static const size_t Dimension=DIM;

	typedef typename detail::tuple_type<N, T>::type type;
	typedef typename detail::tuple_type<DIM, T>::type coordinate_type;
	typedef  T value_type;

	/**
	 * Default constructor for points
	 * without errors in coordinates and value
	 */
	__host__ __device__
	Point():
	fData()
	{ }

	/**
	 * Constructor for points without errors in coordinates and value
	 * @param coordinates:  std::array<value_type,DIM> with the coordinates
	 * @param weight: weight of this point
	 */
	__host__
	Point(std::array<value_type,DIM> const& coordinates, value_type weight=1.0 )
	{
		auto weights = thrust::make_tuple(weight, weight*weight );
		auto coords  = hydra::detail::arrayToTuple<value_type,DIM>(const_cast<value_type*>(coordinates.data() ));
		fData = thrust::tuple_cat(weights, coords  );
	}

	/**
	 * Constructor for points without errors in coordinates and value
	 * @param coordinates: value_type[DIM] with the coordinates
	 * @param weight: weight of this point
	 */
	__host__ __device__
	Point(value_type (&coordinates)[DIM],  value_type weight=1.0 )
	{
		auto weights = thrust::make_tuple(weight, weight*weight );
		auto coords  = hydra::detail::arrayToTuple<value_type,DIM>(const_cast<value_type*>( &coordinates[0] ));
		fData = thrust::tuple_cat(weights, coords  );
	}

	/**
	 * Constructor for points without errors in coordinates and value
	 * @param coordinates: std::initializer_list
	 * @param weight: weight of this point
	 */
	__host__
	Point(std::initializer_list<value_type> coordinates, value_type weight=1.0 )
	{
		std::vector<value_type> v(coordinates);
		auto weights = thrust::make_tuple(weight, weight*weight );
		auto coords  = hydra::detail::arrayToTuple<value_type,DIM>(const_cast<value_type*>(v.data() ));
		fData = thrust::tuple_cat(weights, coords  );
	}


	/**
	 * Constructor for points without errors in coordinates and value
	 * @param coordinates: std::initializer_list
	 * @param weight: weight of this point
	 * @return
	 */
	__host__  __device__
	Point(  coordinate_type coordinates,value_type weight=1.0)
	{
		auto weights = thrust::make_tuple(weight, weight*weight );
		fData = thrust::tuple_cat(weights, coordinates  );
	}

	/**
	 * Constructor for points without errors in coordinates and value
	 * @param coordinates: pointer to array dimension DIM
	 * @param weight: weight of this point
	 * @return
	 */
	__host__  __device__
	explicit Point(value_type* coordinates,	value_type weight=1.0 )
	{
		auto weights = thrust::make_tuple(weight, weight*weight );
		auto coords  = hydra::detail::arrayToTuple<value_type,DIM>(const_cast<value_type*>( coordinates));
		fData = thrust::tuple_cat(weights, coords  );
	}


	__host__  __device__
	Point( Point<value_type,DIM,false,false> const& other):
	fData(other.GetData() )
		{}

	__host__  __device__ inline
	Point<value_type,DIM,false,false>& operator=(type const& value)
	{
		if( !HasValueError && !HasPointError)
		{
			this->SetCoordinates( value);
			this->SetWeight(1.0);
			this->SetWeight2(1.0);
			return *this;
		}
		else{
			return *this;
		}

	}

	__host__  __device__ inline
	Point<value_type,DIM,false,false>& operator=(value_type value)
	{
		if(N==1 && !HasValueError && !HasPointError)
		{
			this->SetCoordinates( thrust::make_tuple(value));
			this->SetWeight(1.0);
			this->SetWeight2(1.0);
			return *this;
		}
		else{
			return *this;
		}

	}


	__host__  __device__
	inline type& GetCoordinates() {
		return fCoordinates;
	}

	__host__  __device__
	inline type const& GetCoordinates() const{
		return fCoordinates;
	}

	__host__  __device__
	inline value_type GetCoordinate(const int i) const{
		return detail::extract<value_type, type>(i, fCoordinates);
	}

	__host__  __device__
	inline void SetCoordinates(type coordinates) {
		fCoordinates = coordinates;
	}


	__host__  __device__
	inline value_type GetWeight() {
		return fWeight;
	}

	__host__  __device__
	inline const value_type GetWeight() const {
		return fWeight;
	}

	__host__  __device__
	inline void SetWeight(value_type weight) {
		fWeight = weight;
	}

	__host__  __device__
	inline value_type GetWeight2()  {
		return fWeight2;
	}

	__host__  __device__
	inline value_type GetWeight2() const {
		return fWeight2;
	}

	__host__  __device__
	inline void SetWeight2(value_type weight2) {
		fWeight2 = weight2;
	}

private:
	type fData;
}

}  // namespace experimental

} // namespace hydra
#endif /* POINT1_INL_ */
