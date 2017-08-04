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
 * Vector3R.h
 *
 * obs.: inspired on the corresponding EvtGen classes.
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef _VECTOR3R_H_
#define _VECTOR3R_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <iosfwd>
#include <iostream>
#include <math.h>
#include <hydra/Collection.h>

using std::ostream;

namespace hydra {

/**
 * @ingroup phsp
 * @brief This class represents three-dimensional Euclidian vectors and implements common operation performed on it.
 * This class is inspired on the corresponding EvtGen classes.
 *
 */
class Vector3R
{

	__host__      __device__      friend Vector3R rotateEuler(const Vector3R& v,
			GReal_t phi, GReal_t theta, GReal_t ksi);

	__host__      __device__      inline friend Vector3R operator*(GReal_t c,
			const Vector3R& v2);
	__host__      __device__      inline friend GReal_t operator*(const Vector3R& v1,
			const Vector3R& v2);
	__host__      __device__      inline friend Vector3R operator+(const Vector3R& v1,
			const Vector3R& v2);
	__host__      __device__      inline friend Vector3R operator-(const Vector3R& v1,
			const Vector3R& v2);
	__host__      __device__      inline friend Vector3R operator*(const Vector3R& v1,
			GReal_t c);
	__host__      __device__      inline friend Vector3R operator/(const Vector3R& v1,
			GReal_t c);
	__host__      __device__      friend Vector3R cross(const Vector3R& v1,
			const Vector3R& v2);

public:
	__host__ __device__ inline Vector3R();
	__host__ __device__ inline Vector3R(GReal_t x, GReal_t y, GReal_t z);
	__host__ __device__ inline Vector3R(const Vector3R& other);
	__host__ __device__ inline Vector3R(Vector3R&& other);
	__host__ __device__ inline Vector3R& operator=(const Vector3R& other);
	__host__ __device__ inline Vector3R& operator=(Vector3R&& other);
	__host__ __device__ inline void swap(Vector3R& other);
	__host__      __device__      inline Vector3R& operator*=(const GReal_t c);
	__host__      __device__      inline Vector3R& operator/=(const GReal_t c);
	__host__      __device__      inline Vector3R& operator+=(const Vector3R& v2);
	__host__      __device__      inline Vector3R& operator-=(const Vector3R& v2);
	__host__ __device__ inline void set(GInt_t i, GReal_t d);
	__host__ __device__ inline void set(GReal_t x, GReal_t y, GReal_t z);
	__host__ __device__ inline void applyRotateEuler(GReal_t phi, GReal_t theta,
			GReal_t ksi);
	__host__      __device__      inline GReal_t get(GInt_t i) const;
	__host__       inline friend std::ostream& operator<<(std::ostream& s,
			const Vector3R& v);
	__host__      __device__      inline GReal_t dot(const Vector3R& v2);
	__host__      __device__      inline GReal_t d3mag() const;

private:

	GReal_t v[3];

	_DeclareStorable(Vector3R, v[0], v[1] , v[2])
};

__host__ __device__
inline void swap(Vector3R& v1, Vector3R& v2)
{
	return v1.swap(v2);
}


__host__ __device__
inline Vector3R operator*(GReal_t c, const Vector3R& v2)
{

	return Vector3R(v2) *= c;
}
__host__ __device__
inline Vector3R operator*(const Vector3R& v1, GReal_t c)
{

	return Vector3R(v1) *= c;
}
__host__ __device__
inline Vector3R operator/(const Vector3R& v1, GReal_t c)
{

	return Vector3R(v1) /= c;
}
__host__ __device__
inline GReal_t operator*(const Vector3R& v1, const Vector3R& v2)
{

	return v1.v[0] * v2.v[0] + v1.v[1] * v2.v[1] + v1.v[2] * v2.v[2];
}
__host__ __device__
inline Vector3R operator+(const Vector3R& v1, const Vector3R& v2)
{

	return Vector3R(v1) += v2;
}
__host__ __device__
inline Vector3R operator-(const Vector3R& v1, const Vector3R& v2)
{

	return Vector3R(v1) -= v2;

}


inline ostream& operator<<(ostream& s, const Vector3R& v)
{

	s << "(" << v.v[0] << "," << v.v[1] << "," << v.v[2] << ")";

	return s;

}
__host__ __device__
inline Vector3R cross(const Vector3R& p1, const Vector3R& p2)
{

	//Calcs the cross product.  Added by djl on July 27, 1995.
	//Modified for real vectros by ryd Aug 28-96

	return Vector3R(p1.v[1] * p2.v[2] - p1.v[2] * p2.v[1],
			p1.v[2] * p2.v[0] - p1.v[0] * p2.v[2],
			p1.v[0] * p2.v[1] - p1.v[1] * p2.v[0]);

}

__host__ __device__
inline Vector3R rotateEuler(const Vector3R& v,	GReal_t phi, GReal_t theta, GReal_t ksi)
{
	Vector3R vect(v);
	vect.applyRotateEuler(phi, theta, ksi);
	return vect;
}

}// namespace hydra
#endif /* _VECTOR3R_H_ */

#include <hydra/detail/Vector3R.inl>
