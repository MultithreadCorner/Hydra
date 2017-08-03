
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
 * Vector4R.h
 *
 * obs.: inspired on the corresponding EvtGen class.
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */

#ifndef _VECTOR4R_H_
#define _VECTOR4R_H_

#include <math.h>
#include <iostream>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Vector3R.h>
#include <hydra/Collection.h>

using std::ostream;

namespace hydra {

/**
 * @ingroup phsp
 * @brief This class represents four-dimensional relativistic vectors and implements common operation performed on it.
 * This class is inspired on the corresponding EvtGen classes.
 *
 */
class Vector4R
{

	__host__      __device__      inline friend Vector4R operator*(GReal_t d,
			const Vector4R& v2);
	__host__      __device__      inline friend Vector4R operator*(const Vector4R& v2,
			GReal_t d);
	__host__      __device__      inline friend Vector4R operator/(const Vector4R& v2,
			GReal_t d);
	__host__      __device__      inline friend GReal_t operator*(const Vector4R& v1,
			const Vector4R& v2);
	__host__      __device__      inline friend Vector4R operator+(const Vector4R& v1,
			const Vector4R& v2);
	__host__      __device__      inline friend Vector4R operator-(const Vector4R& v1,
			const Vector4R& v2);

public:
	__host__ __device__ inline Vector4R();
	__host__ __device__ inline Vector4R(GReal_t e, GReal_t px, GReal_t py,
			GReal_t pz);
	__host__ __device__ inline Vector4R(const Vector4R& other);
	__host__ __device__ inline Vector4R(Vector4R&& other);

	__host__ __device__ inline Vector4R& operator=(const Vector4R& other);
	__host__ __device__ inline Vector4R& operator=(Vector4R&& other);

	__host__ __device__ inline void swap( Vector4R& other);

	__host__ __device__ inline void set(GInt_t i, GReal_t d);
	__host__ __device__ inline void set(GReal_t e, GReal_t px, GReal_t py,
			GReal_t pz);
	__host__      __device__      inline Vector4R& operator*=(GReal_t c);
	__host__      __device__      inline Vector4R& operator/=(GReal_t c);
	__host__      __device__      inline Vector4R& operator+=(const Vector4R& v2);
	__host__      __device__      inline Vector4R& operator-=(const Vector4R& v2);
	__host__      __device__      inline GReal_t get(GInt_t i) const;
	__host__      __device__      inline GReal_t cont(const Vector4R& v4) const;
	__host__      inline friend std::ostream& operator<<(std::ostream& s,
			const Vector4R& v);
	__host__      __device__       inline GReal_t mass2() const;
	__host__      __device__       inline GReal_t mass() const;
	__host__ __device__ inline void applyRotateEuler(GReal_t alpha,
			GReal_t beta, GReal_t gamma);
	__host__ __device__ inline void applyBoostTo(const Vector4R& p4,
			bool inverse = false);
	__host__ __device__ inline void applyBoostTo(const Vector3R& boost,
			bool inverse = false);
	__host__ __device__ inline void applyBoostTo(const GReal_t bx,
			const GReal_t by, const GReal_t bz, bool inverse = false);
	__host__      __device__       inline Vector4R cross(const Vector4R& v2);
	__host__      __device__       inline GReal_t dot(const Vector4R& v2) const;
	__host__      __device__       inline GReal_t d3mag() const;

	// Added by AJB - calculate scalars in the rest frame of the current object
	__host__      __device__       inline GReal_t scalartripler3( Vector4R p1,
			Vector4R p2, Vector4R p3) const;
	__host__      __device__       inline GReal_t dotr3(const Vector4R& p1,
			const Vector4R& p2) const;
	__host__      __device__       inline GReal_t mag2r3(const Vector4R& p1) const;
	__host__      __device__       inline GReal_t magr3(const Vector4R& p1) const;

private:

	GReal_t v[4];

	__host__      __device__      inline GReal_t Square(GReal_t x) const
	{
		return x * x;
	}


	_DeclareStorable(Vector4R, v[0], v[1] , v[2], v[3])
};

__host__ __device__
Vector4R rotateEuler(const Vector4R& rs, GReal_t alpha, GReal_t beta,
		GReal_t gamma);
__host__ __device__
Vector4R boostTo(const Vector4R& rs, const Vector4R& p4, bool inverse = false);

__host__ __device__
Vector4R boostTo(const Vector4R& rs, const Vector3R& boost,
		bool inverse = false);

__host__ __device__
inline Vector4R operator*(GReal_t c, const Vector4R& v2)
{

	return Vector4R(v2) *= c;
}
__host__ __device__
inline Vector4R operator*(const Vector4R& v2, GReal_t c)
{

	return Vector4R(v2) *= c;
}
__host__ __device__
inline Vector4R operator/(const Vector4R& v2, GReal_t c)
{

	return Vector4R(v2) /= c;
}


__host__ __device__
inline GReal_t operator*(const Vector4R& v1, const Vector4R& v2)
{

	return v1.v[0] * v2.v[0] - v1.v[1] * v2.v[1] - v1.v[2] * v2.v[2]
			- v1.v[3] * v2.v[3];
}

__host__ __device__
inline void swap(Vector4R& v1, Vector4R& v2)
{
	return v1.swap(v2);
}


/*
inline GReal_t Vector4R::cont(const Vector4R& v4) const
{

	return v[0] * v4.v[0] - v[1] * v4.v[1] - v[2] * v4.v[2] - v[3] * v4.v[3];
}

*/
__host__ __device__
inline Vector4R operator-(const Vector4R& v1, const Vector4R& v2)
{

	return Vector4R(v1) -= v2;
}
__host__ __device__
inline Vector4R operator+(const Vector4R& v1, const Vector4R& v2)
{

	return Vector4R(v1) += v2;
}


__host__ __device__
inline Vector4R rotateEuler(const Vector4R& rs, GReal_t alpha, GReal_t beta,
		GReal_t gamma)
{

	Vector4R tmp(rs);
	tmp.applyRotateEuler(alpha, beta, gamma);
	return tmp;

}
__host__ __device__
inline Vector4R boostTo(const Vector4R& rs, const Vector4R& p4, bool inverse)
{

	Vector4R tmp(rs);
	tmp.applyBoostTo(p4, inverse);
	return tmp;

}
__host__ __device__
inline Vector4R boostTo(const Vector4R& rs, const Vector3R& boost, bool inverse)
{

	Vector4R tmp(rs);
	tmp.applyBoostTo(boost, inverse);
	return tmp;

}



inline ostream& operator<<(ostream& s, const Vector4R& v)
{

	s << "(" << v.v[0] << "," << v.v[1] << "," << v.v[2] << "," << v.v[3]
			<< ")";

	return s;

}


}// namespace hydra

#include<hydra/detail/Vector4R.inl>
#endif /* VECTOR4R_H_ */
