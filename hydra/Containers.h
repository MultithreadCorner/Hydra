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
 * Containers.h
 *
 * Copyright 2016 Antonio Augusto Alves Junior
 *
 * Created on : Feb 25, 2016
 *      Author: Antonio Augusto Alves Junior
 */



/**
 * @file
 * @ingroup generic
 *  Typedefs for useful container classes used in MCBooster.
 *  Containers defined here should be used in users application also.
 */
#ifndef CONTAINERS_H_
#define CONTAINERS_H_

#include <vector>
#include <array>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>

#include <hydra/detail/external/thrust/complex.h>

#include <hydra/detail/Containers.inl>

namespace hydra
{


//-----------------------------------------------------------------------
//complex number container
typedef HYDRA_EXTERNAL_NS::thrust::complex<GReal_t> GComplex_t; /*! Typedef for complex number.*/

//-----------------------------------------------------------------------

typedef mc_host_vector<Vector4R> FourVectors_h; /*! Vector4R host vector. Use it to store four-vectors at __hydra_host__.*/
typedef mc_host_vector<Vector3R> ThreeVectors_h; /*! Vector3R host vector. Use it to store four-vectors at __hydra_host__.*/

//-----------------------------------------------------------------------
//basic containers on host

typedef mc_host_vector<GBool_t>    BoolVector_h; /*! Typedef for a GBool_t host vector.*/
typedef mc_host_vector<GReal_t>    RealVector_h; /*! Typedef for a GReal_t host vector.*/
typedef mc_host_vector<GComplex_t> ComplexVector_h; /*! Typedef for a GComplex_t host vector.*/
typedef mc_host_vector<Vector4R>   Particles_h;/*! Typedef for a  Vector4R host vector.*/


//-----------------------------------------------------------------------
//basic containers on device
typedef mc_device_vector<GBool_t>    BoolVector_d; /*! Typedef for a GBool_t device vector.*/
typedef mc_device_vector<GReal_t>    RealVector_d; /*! Typedef for a GReal_t device vector.*/
typedef mc_device_vector<GComplex_t> ComplexVector_d; /*! Typedef for a GComplex_t device vector.*/
typedef mc_device_vector<Vector4R>   Particles_d; /*! Typedef for a  Vector4R device vector.*/


}
#endif /* CONTAINERS_H_ */
