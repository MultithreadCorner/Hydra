/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * Types.h
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * @file
 * @ingroup generic
 * @brief Common and useful typedefs
 */

#ifndef TYPES_H_
#define TYPES_H_

namespace hydra
{
//---- POD types -------------------------------------------------------------------

typedef char GChar_t;      ///< Signed Character 1 byte (char)
typedef unsigned char GUChar_t;  ///< Unsigned Character 1 byte (unsigned char)
typedef short GShort_t;     ///< Signed Short integer 2 bytes (short)
typedef unsigned short GUShort_t; ///<Unsigned Short integer 2 bytes (unsigned short)
typedef int GInt_t;       ///< Signed integer 4 bytes (int)
typedef unsigned int GUInt_t;      ///< Unsigned integer 4 bytes (unsigned int)
typedef long GLong_t;      ///< Signed long integer 4 bytes (long)
typedef unsigned long GULong_t; //Unsigned long integer 4 bytes (unsigned long)
typedef float GFloat_t;     ///< Float 4 bytes (float)
typedef double GDouble_t;    ///< Double 8 bytes
typedef long double GLongDouble_t;    ///< Long Double
typedef char GText_t;      ///< General string (char)
typedef bool GBool_t;      ///< Boolean (0=false, 1=true) (bool)
typedef unsigned char GByte_t;      ///< Byte (8 bits) (unsigned char)
typedef long long GLong64_t; ///< Portable signed long integer 8 bytes
typedef unsigned long long GULong64_t; ///< Portable unsigned long integer 8 bytes

#ifdef FP_SINGLE
typedef float GReal_t;///< Double 8 bytes or float 4 bytes
#else
typedef double GReal_t;///< Double 16 bytes or float 4 bytes
#endif

//---- constants ---------------------------------------------------------------

#ifndef NULL
#define NULL 0
#endif

const GBool_t kTrue = true;
const GBool_t kFalse = false;

#define PI     3.1415926535897932384626422832795028841971

namespace math_constants {

const double sqrt6           = 2.449489742783178098197;  //sqrt(6)
const double sqrt3_2         = 1.224744871391589049099;  //sqrt(3/2)
const double sqrt2           = 1.41421356237309504880;   //sqrt(2)
const double sqrt3           = 1.73205080756887729353;   //sqrt(3)
const double sqrt5           = 2.23606797749978969641;   //sqrt(5)
const double sqrt10          = 3.162277660168379331999;  //sqrt(10)
const double inverse_sqrt2   = 0.707106781186547524401;  //1/sqrt(2)
const double inverse_sqrt2Pi = 0.398942280401432677939;  //1/sqrt(2pi)

}//math_constants

struct null_type
{};


}
#endif /* TYPES_H_ */
