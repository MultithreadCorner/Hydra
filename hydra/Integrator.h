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
 * Integrator.h
 *
 *  Created on: 31/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <utility>

namespace hydra {

template<typename Functor, size_t N=1>
class IntegrationFormula;

template<typename Functor, size_t N=1>
class AnalyticalIntegral;

template<typename Functor, size_t N=1>
class NumericalIntegral;

template<typename Algorithm, size_t N=1>
struct Integral;


}  // namespace hydra

#include <hydra/detail/AnalyticalIntegral.inl>
#include <hydra/detail/NumericalIntegral.inl>
#include <hydra/detail/Integrator.inl>

#endif /* INTEGRATOR_H_ */
