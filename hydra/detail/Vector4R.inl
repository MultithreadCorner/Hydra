/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * Vector4R.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef _VECTOR4R_INL_
#define _VECTOR4R_INL_

namespace hydra {
__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator+=(const Vector4R& v2)
{

	v[0] += v2.v[0];
	v[1] += v2.v[1];
	v[2] += v2.v[2];
	v[3] += v2.v[3];

	return *this;
}
__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator-=(const Vector4R& v2)
{

	v[0] -= v2.v[0];
	v[1] -= v2.v[1];
	v[2] -= v2.v[2];
	v[3] -= v2.v[3];

	return *this;
}
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::mass2() const
{

	return v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3];
}


__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator*=(GReal_t c)
{

	v[0] *= c;
	v[1] *= c;
	v[2] *= c;
	v[3] *= c;

	return *this;
}

__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator/=(GReal_t c)
{

	GReal_t cinv = 1.0 / c;
	v[0] *= cinv;
	v[1] *= cinv;
	v[2] *= cinv;
	v[3] *= cinv;

	return *this;
}


__hydra_host__ __hydra_device__
inline GReal_t Vector4R::cont(const Vector4R& v4) const
{

	return v[0] * v4.v[0] - v[1] * v4.v[1] - v[2] * v4.v[2] - v[3] * v4.v[3];
}


__hydra_host__ __hydra_device__
inline GReal_t Vector4R::get(GInt_t i) const
{
	return v[i];
}
__hydra_host__ __hydra_device__
inline void Vector4R::set(GInt_t i, GReal_t d)
{

	v[i] = d;
}

__hydra_host__ __hydra_device__ inline void Vector4R::set(GReal_t e, GReal_t p1, GReal_t p2,
		GReal_t p3)
{

	v[0] = e;
	v[1] = p1;
	v[2] = p2;
	v[3] = p3;
}



__hydra_host__ __hydra_device__
inline Vector4R::Vector4R(GReal_t e, GReal_t p1, GReal_t p2, GReal_t p3)
{

	v[0] = e;
	v[1] = p1;
	v[2] = p2;
	v[3] = p3;
}

__hydra_host__ __hydra_device__
inline Vector4R::Vector4R(GReal_t e, const Vector3R& p)
{

	v[0] = e;
	v[1] = p.get(0);
	v[2] = p.get(1);
	v[3] = p.get(2);
}

__hydra_host__ __hydra_device__
inline Vector4R::Vector4R(const Vector4R& other)
{
	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
	v[3] = other.get(3);
}

__hydra_host__ __hydra_device__
inline Vector4R::Vector4R(Vector4R&& other)
{
	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
	v[3] = other.get(3);
}


__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator=(Vector4R&& other)
{
	if(this==&other) return *this;
	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
	v[3] = other.get(3);
	 return *this;
}

__hydra_host__ __hydra_device__
inline Vector4R& Vector4R::operator=(Vector4R const& other)
{
	if(this==&other) return *this;
	v[0] = other.get(0);
	v[1] = other.get(1);
	v[2] = other.get(2);
	v[3] = other.get(3);
	 return *this;
}


__hydra_host__ __hydra_device__
inline void Vector4R::swap(Vector4R& other)
{
	if(this==&other) return;

	Vector4R temp(*this);
	*this= other;
	other = temp;
	return ;
}


__hydra_host__ __hydra_device__
inline GReal_t Vector4R::mass() const
{

	GReal_t m2 = v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3];

	if (m2 > 0.0)
	{
		return ::sqrt(m2);
	}
	else
	{
		return 0.0;
	}
}


__hydra_host__ __hydra_device__
inline void Vector4R::applyRotateEuler(GReal_t phi, GReal_t theta, GReal_t ksi)
{

	GReal_t sp = ::sin(phi);
	GReal_t st = ::sin(theta);
	GReal_t sk = ::sin(ksi);
	GReal_t cp = ::cos(phi);
	GReal_t ct = ::cos(theta);
	GReal_t ck = ::cos(ksi);

	GReal_t x = (ck * ct * cp - sk * sp) * v[1]
			+ (-sk * ct * cp - ck * sp) * v[2] + st * cp * v[3];
	GReal_t y = (ck * ct * sp + sk * cp) * v[1]
			+ (-sk * ct * sp + ck * cp) * v[2] + st * sp * v[3];
	GReal_t z = -ck * st * v[1] + sk * st * v[2] + ct * v[3];

	v[1] = x;
	v[2] = y;
	v[3] = z;

}


__hydra_host__ __hydra_device__
inline void Vector4R::applyBoostTo(const Vector4R& p4, bool inverse)
{

	GReal_t e = p4.get(0);

	Vector3R boost(p4.get(1) / e, p4.get(2) / e, p4.get(3) / e);

	applyBoostTo(boost, inverse);

	return;

}
__hydra_host__ __hydra_device__
inline void Vector4R::applyBoostTo(const Vector3R& boost, bool inverse)
{

	GReal_t bx, by, bz, gamma, b2;

	bx = boost.get(0);
	by = boost.get(1);
	bz = boost.get(2);

	GReal_t bxx = bx * bx;
	GReal_t byy = by * by;
	GReal_t bzz = bz * bz;

	b2 = bxx + byy + bzz;

	if (b2 > 0.0 && b2 < 1.0)
	{

		gamma = 1.0 / ::sqrt(1.0 - b2);

		GReal_t gb2 = (gamma - 1.0) / b2;

		GReal_t gb2xy = gb2 * bx * by;
		GReal_t gb2xz = gb2 * bx * bz;
		GReal_t gb2yz = gb2 * by * bz;

		GReal_t gbx = gamma * bx;
		GReal_t gby = gamma * by;
		GReal_t gbz = gamma * bz;

		GReal_t e2 = v[0];
		GReal_t px2 = v[1];
		GReal_t py2 = v[2];
		GReal_t pz2 = v[3];

		if (inverse)
		{
			v[0] = gamma * e2 - gbx * px2 - gby * py2 - gbz * pz2;

			v[1] = -gbx * e2 + gb2 * bxx * px2 + px2 + gb2xy * py2
					+ gb2xz * pz2;

			v[2] = -gby * e2 + gb2 * byy * py2 + py2 + gb2xy * px2
					+ gb2yz * pz2;

			v[3] = -gbz * e2 + gb2 * bzz * pz2 + pz2 + gb2yz * py2
					+ gb2xz * px2;
		}
		else
		{
			v[0] = gamma * e2 + gbx * px2 + gby * py2 + gbz * pz2;

			v[1] = gbx * e2 + gb2 * bxx * px2 + px2 + gb2xy * py2 + gb2xz * pz2;

			v[2] = gby * e2 + gb2 * byy * py2 + py2 + gb2xy * px2 + gb2yz * pz2;

			v[3] = gbz * e2 + gb2 * bzz * pz2 + pz2 + gb2yz * py2 + gb2xz * px2;
		}
	}

}
__hydra_host__ __hydra_device__
inline void Vector4R::applyBoostTo(const GReal_t bx, const GReal_t by,
		const GReal_t bz, bool inverse)
{

	GReal_t gamma, b2;

	GReal_t bxx = bx * bx;
	GReal_t byy = by * by;
	GReal_t bzz = bz * bz;

	b2 = bxx + byy + bzz;

	if (b2 > 0.0 && b2 < 1.0)
	{

		gamma = 1.0 / ::sqrt(1.0 - b2);

		GReal_t gb2 = (gamma - 1.0) / b2;

		GReal_t gb2xy = gb2 * bx * by;
		GReal_t gb2xz = gb2 * bx * bz;
		GReal_t gb2yz = gb2 * by * bz;

		GReal_t gbx = gamma * bx;
		GReal_t gby = gamma * by;
		GReal_t gbz = gamma * bz;

		GReal_t e2 = v[0];
		GReal_t px2 = v[1];
		GReal_t py2 = v[2];
		GReal_t pz2 = v[3];

		if (inverse)
		{
			v[0] = gamma * e2 - gbx * px2 - gby * py2 - gbz * pz2;

			v[1] = -gbx * e2 + gb2 * bxx * px2 + px2 + gb2xy * py2
					+ gb2xz * pz2;

			v[2] = -gby * e2 + gb2 * byy * py2 + py2 + gb2xy * px2
					+ gb2yz * pz2;

			v[3] = -gbz * e2 + gb2 * bzz * pz2 + pz2 + gb2yz * py2
					+ gb2xz * px2;
		}
		else
		{
			v[0] = gamma * e2 + gbx * px2 + gby * py2 + gbz * pz2;

			v[1] = gbx * e2 + gb2 * bxx * px2 + px2 + gb2xy * py2 + gb2xz * pz2;

			v[2] = gby * e2 + gb2 * byy * py2 + py2 + gb2xy * px2 + gb2yz * pz2;

			v[3] = gbz * e2 + gb2 * bzz * pz2 + pz2 + gb2yz * py2 + gb2xz * px2;
		}
	}

}

__hydra_host__ __hydra_device__
inline Vector4R Vector4R::cross(const Vector4R& p2)
{

	//Calcs the cross product.  Added by djl on July 27, 1995.
	//Modified for real vectros by ryd Aug 28-96

	Vector4R temp;

	temp.v[0] = 0.0;
	temp.v[1] = v[2] * p2.v[3] - v[3] * p2.v[2];
	temp.v[2] = v[3] * p2.v[1] - v[1] * p2.v[3];
	temp.v[3] = v[1] * p2.v[2] - v[2] * p2.v[1];

	return temp;
}

__hydra_host__      __hydra_device__
inline Vector3R Vector4R::vector3()
{
	Vector3R temp;


	temp.set(v[1], v[2] , v[3] );

	return temp;
}
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::p() const

// returns the 3 momentum mag.
{
	GReal_t temp;

	temp = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];

	temp = ::sqrt(temp);

	return temp;
} //

__hydra_host__ __hydra_device__
inline GReal_t Vector4R::p2() const

// returns the 3 momentum mag.
{
	GReal_t temp;

	temp = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];


	return temp;
} //

__hydra_host__ __hydra_device__
inline GReal_t Vector4R::d3mag() const

// returns the 3 momentum mag.
{
	GReal_t temp;

	temp = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];

	temp = ::sqrt(temp);

	return temp;
} // r3mag


__hydra_host__ __hydra_device__
inline GReal_t Vector4R::dot(const Vector4R& p2) const
{

	//Returns the dot product of the 3 momentum.  Added by
	//djl on July 27, 1995.  for real!!!

	GReal_t temp;

	temp  = v[1] * p2.v[1];
	temp += v[2] * p2.v[2];
	temp += v[3] * p2.v[3];

	return temp;

} //dot

// calculate (p1xp2)*p3 in the rest frame of
// 4-vector *this (sub-optimal implementation)
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::scalartripler3(Vector4R p1,
		Vector4R p2, Vector4R p3) const
{
	p1.applyBoostTo(*this);
	p2.applyBoostTo(*this);
	p3.applyBoostTo(*this);

	GReal_t r = (p1.cross(p2)).dot(p3);
	return r;
}

// Calculate the 3-d dot product of 4-vectors p1 and p2 in the rest frame of
// 4-vector p0
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::dotr3(const Vector4R& p1, const Vector4R& p2) const
{
	return 1 / mass2() * ((*this) * p1) * ((*this) * p2) - p1 * p2;
}

// Calculate the 3-d magnitude squared of 4-vector p1 in the rest frame of
// 4-vector p0
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::mag2r3(const Vector4R& p1) const
{
	return Square((*this) * p1) / mass2() - p1.mass2();
}

// Calculate the 3-d magnitude 4-vector p1 in the rest frame of 4-vector p0.
__hydra_host__ __hydra_device__
inline GReal_t Vector4R::magr3(const Vector4R& p1) const
{
	return ::sqrt(mag2r3(p1));
}


}// namespace hydra

#endif /* VECTOR4R_INL_ */
