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

#ifndef POINT_H_
#define POINT_H_


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


/**
 * Value error to be interpreted as error on bin content or point weight depending on the case.
 * ValueError is base class for Point and provides conditional methods and members
 * to save space in device memory
 */
template<typename T, bool>
struct ValueError{
public:

	typedef T value_type;
	typedef ValueError<T, true> type;
};

template<typename T>
struct ValueError<T, true> {

public:

	typedef T value_type;
	typedef ValueError<T, true> type;

	__host__  __device__
	ValueError():fValueError(0){}

	__host__  __device__
	ValueError(value_type const error ):
	fValueError(error)
       {}

	__host__  __device__
	inline ValueError(ValueError<value_type, true> const& other ):
	fValueError(other.GetValueError())
	{}


	__host__  __device__
	inline value_type GetValueError() const {
		return fValueError;
	}

	__host__  __device__
	inline void SetValueError(value_type weighError) {
		fValueError = weighError;
	}

private:
	value_type fValueError;
};

/**
 * PointError add symmetric error to data points, just in case someone comes to need of it
 * PointError is base class for Point and provides conditional methods and members
 * to save space in device memory
 */

template<typename T, size_t N, bool>
struct PointError{

	typedef typename detail::tuple_type<N, T>::type type;
	typedef T value_type;
};


template<typename T, size_t N>
struct PointError<T, N, true>{

	typedef typename detail::tuple_type<N, T>::type type;
	typedef T value_type;

	__host__  __device__
	PointError():fErrors(){}

	PointError(std::array<value_type,N> coordinates):
		fErrors( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates.data() ) ))
	{ }

	PointError(std::initializer_list<value_type> coordinates)
	{
		std::vector<value_type> v(coordinates);
		fErrors( detail::arrayToTuple<value_type,N>(const_cast<value_type*>( v.data() ) ));
	}

	__host__  __device__
	PointError(type coordinates):
	fErrors( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) ))
	{ }

	__host__  __device__
	explicit PointError(value_type* coordinates):
	fErrors( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) ))
	{ }

	__host__  __device__
	PointError( PointError<value_type,N,true> const& other):
		fErrors(other.GetErrors() )
	{}

	__host__  __device__
	type const& GetErrors() const {
			return fErrors;
		}

	__host__  __device__
	type& GetErrors() {
		return fErrors;
	}

	__host__  __device__
	inline value_type GetError(const int i) const{
			return detail::extract<value_type, type>(i, fErrors);
		}


	__host__  __device__
	void SetErrors(const type errors) {
		fErrors = errors;
	}

private:
    type fErrors;

};


template<typename T, size_t N=1, bool HasValueError=false, bool HasPointError=false>
struct Point: public ValueError<T, HasValueError>, public PointError<T, N, HasPointError>
{

	typedef typename detail::tuple_type<N, T>::type type;
	typedef  T value_type;
	static const size_t Dimension=N;

	/*
	 * no errors
	 */

	template<bool U = HasValueError, bool V=HasPointError>
	__host__  __device__
	Point(const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
	fCoordinates(),
	fWeight(0),
	fWeight2(0)
	{ }

   /*
	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(value_type coord, value_type weight=0,
			const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
			fCoordinates( detail::make_tuple<N>(coord)),
			fWeight(weight),
			fWeight2(weight*weight)
	{ }
	*/

	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(std::array<value_type,N> coordinates, value_type weight,
			const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates.data() ) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(std::initializer_list<value_type> coordinates, value_type weight,
			const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
	fWeight(weight),
	fWeight2(weight*weight)
	{
		std::vector<value_type> v(coordinates);
		fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>( v.data() ) ));
	}

	template<bool U = HasValueError, bool V=HasPointError>
	__host__  __device__
	Point(type coordinates, value_type weight=1.0,
			const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
	fCoordinates( coordinates ),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U = HasValueError, bool V=HasPointError>
	__host__  __device__
	explicit Point(value_type* coordinates, value_type weight,
			const typename std::enable_if< !U && !V , void>::type* dummy=0 ):
		fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) )),
		fWeight(weight),
		fWeight2(weight*weight)
	{ }




	/*
	 * value errors
	 */
	template<bool U = HasValueError, bool V=HasPointError>
	__host__ __device__
	Point(const typename std::enable_if< U && !V , void>::type* dummy=0 ):
	ValueError<T, HasValueError>(),
	fCoordinates(),
	fWeight(0)
	{ }

	/*
	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(value_type coord, value_type weight=0, value_type error=0,
			const typename std::enable_if< U && !V , void>::type* dummy=0 ):
			ValueError<T, HasValueError>(error),
			fCoordinates( detail::make_tuple<N>(coord)),
			fWeight(weight),
			fWeight2(weight*weight)
			{ }
    */

	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(std::array<value_type,N> coordinates, const value_type weight, value_type error,
			const typename std::enable_if< U && !V , void>::type* dummy=0 ):
	ValueError<T, HasValueError>(error),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates.data() ) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U = HasValueError, bool V=HasPointError>
	__host__
	Point(std::initializer_list<value_type> coordinates, const value_type weight, value_type error,
			const typename std::enable_if< U && !V , void>::type* dummy=0):
	ValueError<T, HasValueError>(error),
	fWeight(weight),
	fWeight2(weight*weight)
	{
		std::vector<value_type> v(coordinates);
		fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<T*>( v.data() ) ));
	}

	template<bool U = HasValueError, bool V=HasPointError>
	__host__  __device__
	Point(type coordinates, const value_type weight, value_type error,
			const typename std::enable_if< U && !V , void>::type* dummy=0):
	ValueError<T, HasValueError>(error),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U = HasValueError, bool V=HasPointError>
	__host__  __device__
	explicit Point(value_type* coordinates, const value_type weight, value_type error,
			const typename std::enable_if< U && !V , void>::type* dummy=0):
	ValueError<T, HasValueError>(error),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }


	/*
	 * value errors + point errors
	 */
	template<bool U=HasValueError, bool V=HasPointError>
	__host__  __device__
	Point(const typename std::enable_if< U&&V, void >::type* dummy=0 ):
			ValueError<T, HasValueError>(),
			PointError<T, N, HasPointError>(),
			fCoordinates(),
			fWeight(0),
			fWeight2(0)
			{ }

	/*
	template<bool U=HasValueError, bool V=HasPointError>
		__host__
		Point(value_type coord, value_type coord_errors, value_type weight, value_type error,
				const typename std::enable_if< U&&V, void >::type* dummy=0 ):
		ValueError<T, HasValueError>(error),
		PointError<T, N, HasPointError>(detail::make_tuple<N>(coord_errors)),
		fCoordinates( detail::make_tuple<N>(coord)),
		fWeight(weight),
		fWeight2(weight*weight)
		{ }
    */
	template<bool U=HasValueError, bool V=HasPointError>
	__host__
	Point(std::array<value_type,N> coordinates, std::array<value_type,N> coordinates_errors, value_type weight, value_type error,
			const typename std::enable_if< U&&V, void >::type* dummy=0 ):
	ValueError<T, HasValueError>(error),
	PointError<T, N, HasPointError>(coordinates_errors),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<T*>(coordinates.data() ) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U=HasValueError, bool V=HasPointError>
	__host__
	Point(std::initializer_list<value_type> coordinates, std::initializer_list<value_type> coordinates_errors, value_type weight, value_type error,
			const typename std::enable_if< U&&V, void >::type* dummy=0 ):
	ValueError<T, HasValueError>(error),
	PointError<T, N, HasPointError>(coordinates_errors),
	fWeight(weight),
	fWeight2(weight*weight)
	{
		std::vector<value_type> v(coordinates);
		fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>( v.data() ) ));
	}

	template<bool U=HasValueError, bool V=HasPointError>
	__host__  __device__
	Point(type coordinates, type coordinates_errors, value_type weight, value_type error,
			const typename std::enable_if< U&&V, void >::type* dummy=0 ):
	ValueError<T, HasValueError>(error),
	PointError<T, N, HasPointError>(coordinates_errors),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	template<bool U=HasValueError, bool V=HasPointError>
	__host__  __device__
	explicit Point(value_type* coordinates, value_type* coordinates_errors, value_type weight, value_type error,
			const typename std::enable_if< U&&V, void >::type* dummy=0 ):
	ValueError<T, HasValueError>(error),
	PointError<T, N, HasPointError>(coordinates_errors),
	fCoordinates( detail::arrayToTuple<value_type,N>(const_cast<value_type*>(coordinates) )),
	fWeight(weight),
	fWeight2(weight*weight)
	{ }

	//copy is always enabled
	__host__  __device__
	Point( Point<value_type,N,HasValueError,HasPointError > const& other):
		ValueError<T, HasValueError>(other),
		PointError<T, N, HasPointError>(other),
		fCoordinates(other.GetCoordinates() ),
		fWeight(other.GetWeight()),
		fWeight2(other.GetWeight2())
	{}

	__host__  __device__ inline
	Point<value_type,N,HasValueError,HasPointError>& operator=(type const& value)
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
	Point<value_type,N,HasValueError,HasPointError>& operator=(value_type value)
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
	type fCoordinates;
	value_type fWeight;
	value_type fWeight2;

};


//output stream operators
template<typename T , size_t N>
__host__ __device__ inline
Point<T,N,false,false> operator+(Point<T,N,false,false> const& point1,
		Point<T,N,false,false> const& point2)
{
	 typedef typename detail::tuple_type<N, T>::type type;

	 Point<T,N,false,false> point(type(), 0);

	 point.SetWeight( point1.GetWeight() + point2.GetWeight());
	 point.SetWeight2( point1.GetWeight2() + point2.GetWeight2());

	 for(size_t i = 0; i<N; i++)
	 point.SetCoordinates( point1.GetCoordinates() + point2.GetCoordinates() );

	 return point ;

}

//output stream operators
template<typename T , size_t N>
__host__
std::ostream& operator<<(std::ostream& os, Point<T,N,false,false> const& point)
{
	 return os <<"X["<< point.Dimension <<"]-> Coord: " << point.GetCoordinates() <<
			                            " Value "       << point.GetWeight() ;

}

template<typename T , size_t N>
__host__
std::ostream& operator<<(std::ostream& os, Point<T,N,true,false> const& point)
{
	 return os <<"X["<< point.Dimension <<"]-> Coord: " << point.GetCoordinates() <<
			                            " Value "       << point.GetWeight() <<
			                            " ValueError "  << point.GetValueError();

}

template<typename T , size_t N>
__host__
std::ostream& operator<<(std::ostream& os, Point<T,N,true,true> const& point)
{
	 return os <<"X["<< point.Dimension <<"]-> Coord: " << point.GetCoordinates() <<
			                            " CoordError: " << point.GetErrors() <<
			                            " Value "       << point.GetWeight() <<
			                            " ValueError "  << point.GetValueError();

}



} // namespace hydra

#endif /* POINT_H_ */
