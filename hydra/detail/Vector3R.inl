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
 * Vector3R.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef _VECTOR3R_INL_
#define _VECTOR3R_INL_

namespace hydra {


__host__ __device__
inline Vector3R& Vector3R::operator*=(const GReal_t c)
{

	v[0] *= c;
	v[1] *= c;
	v[2] *= c;
	return *this;
}
__host__ __device__
inline Vector3R& Vector3R::operator/=(const GReal_t c)
{

	v[0] /= c;
	v[1] /= c;
	v[2] /= c;
	return *this;
}
__host__ __device__
inline Vector3R& Vector3R::operator+=(const Vector3R& v2)
{

	v[0] += v2.v[0];
	v[1] += v2.v[1];
	v[2] += v2.v[2];
	return *this;
}
__host__ __device__
inline Vector3R& Vector3R::operator-=(const Vector3R& v2)
{

	v[0] -= v2.v[0];
	v[1] -= v2.v[1];
	v[2] -= v2.v[2];
	return *this;
}

__host__ __device__
inline void Vector3R::swap(Vector3R& other)
{
	if(this==&other) return;

	Vector3R temp(*this);
	*this= other;
	other = temp;
	return ;
}

__host__ __device__
inline GReal_t Vector3R::get(GInt_t i) const
{
	return v[i];
}
__host__ __device__
inline void Vector3R::set(GInt_t i, GReal_t d)
{

	v[i] = d;
}
__host__ __device__
inline void Vector3R::set(GReal_t x, GReal_t y, GReal_t z)
{

	v[0] = x;
	v[1] = y;
	v[2] = z;
}
__host__ __device__
inline Vector3R::Vector3R()
{

	v[0] = v[1] = v[2] = 0.0;
}
__host__ __device__
inline Vector3R::Vector3R(GReal_t x, GReal_t y, GReal_t z)
{

	v[0] = x;
	v[1] = y;
	v[2] = z;
}
__host__ __device__
inline Vector3R::Vector3R(const Vector3R& other)
{

	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
}

__host__ __device__
inline Vector3R::Vector3R(Vector3R&& other)
{

	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
}


__host__ __device__
inline Vector3R& Vector3R::operator=(const Vector3R& other)
{
	if(this==&other) return *this;

	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);

	return *this;
}


__host__ __device__
inline Vector3R& Vector3R::operator=(Vector3R&& other)
{
	if(this==&other) return *this;

	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);

	return *this;
}


__host__ __device__
inline void Vector3R::applyRotateEuler(GReal_t phi, GReal_t theta, GReal_t ksi)
{

	GReal_t temp[3];
	GReal_t sp, st, sk, cp, ct, ck;

	sp = sin(phi);
	st = sin(theta);
	sk = sin(ksi);
	cp = cos(phi);
	ct = cos(theta);
	ck = cos(ksi);

	temp[0] = (ck * ct * cp - sk * sp) * v[0] + (-sk * ct * cp - ck * sp) * v[1]
			+ st * cp * v[2];
	temp[1] = (ck * ct * sp + sk * cp) * v[0] + (-sk * ct * sp + ck * cp) * v[1]
			+ st * sp * v[2];
	temp[2] = -ck * st * v[0] + sk * st * v[1] + ct * v[2];

	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
__host__ __device__
inline GReal_t Vector3R::d3mag() const

// returns the 3 momentum mag.
{
	GReal_t temp;

	temp = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	temp = sqrt(temp);

	return temp;
} // r3mag
__host__ __device__
inline GReal_t Vector3R::dot(const Vector3R& p2)
{

	GReal_t temp;

	temp = v[0] * p2.v[0];
	temp += v[0] * p2.v[0];
	temp += v[0] * p2.v[0];

	return temp;
} //dot



}// namespace hydra
#endif /* VECTOR3R_INL_ */
