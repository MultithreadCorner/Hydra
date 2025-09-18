/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * StaticAssert.h
 *
 *  Created on: 09/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef STATICASSERT_H_
#define STATICASSERT_H_


#define HYDRA_STATIC_ASSERT(condition, message)\
static_assert(condition,\
"\033[1;34m"\
"\n\n"\
"|~~~~~~~~~~~~~~< HYDRA STATIC ASSERTION FAILED >~~~~~~~~~~~~~~|\n"\
"> Error : " message"\n\n"\
"> Please inspect the error messages issued above to find the line generating the error.\n"\
"|~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~|\n"\
"\n\n"\
"\033[0m");


#endif /* STATICASSERT_H_ */
