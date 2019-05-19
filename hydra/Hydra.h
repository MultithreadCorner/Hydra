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
 * Hydra.h
 *
 *  Created on: 21/02/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HYDRA_H_
#define HYDRA_H_

//  This is the only Hydra header that is guaranteed to
//  change with every Hydra release.
//
//  HYDRA_VERSION % 100 is the sub-minor version
//  HYDRA_VERSION / 100 % 1000 is the minor version
//  HYDRA_VERSION / 100000 is the major version
//
//  Because this header does not #include <hydra/detail/Config.h>,
//  it is the only Hydra header that does not cause
//  HYDRA_HOST_SYSTEM and HYDRA_DEVICE_SYSTEM to be defined.

/*! \def HYDRA_VERSION
 *  \brief The preprocessor macro \p HYDRA_VERSION encodes the version
 *         number of the Hydra.
 *
 *         <tt>HYDRA_VERSION % 100</tt> is the patch version.
 *         <tt>HYDRA_VERSION / 100 % 1000</tt> is the feature version.
 *         <tt>HYDRA_VERSION / 100000</tt> is the major version.
 */
#define HYDRA_VERSION 200302

/*! \def HYDRA_MAJOR_VERSION
 *  \brief The preprocessor macro \p HYDRA_MAJOR_VERSION encodes the
 *         major version number of Hydra.
 */
#define HYDRA_MAJOR_VERSION     (HYDRA_VERSION / 100000)

/*! \def HYDRA_MINOR_VERSION
 *  \brief The preprocessor macro \p HYDRA_MINOR_VERSION encodes the
 *         minor version number of Hydra.
 */
#define HYDRA_MINOR_VERSION     (HYDRA_VERSION / 100 % 1000)

/*! \def HYDRA_PATCH_NUMBER
 *  \brief The preprocessor macro \p HYDRA_PATCH_NUMBER encodes the
 *         patch number of the Hydra library.
 */
#define HYDRA_PATCH_NUMBER  1



// Declare these namespaces here for the purpose of Doxygenating them

/*! \HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
 *  \brief \p thrust is the top-level namespace which contains all Hydra
 *         functions and types.
 */
namespace hydra{ }





#endif /* HYDRA_H_ */
