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
 * Range.h
 *
 *  Created on: 29/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Distance.h>
#include <hydra/detail/Iterable_traits.h>
#include <utility>

namespace hydra {

template<typename ...T>
class Range;

}// namespace hydra

#include <hydra/detail/Range1.inl>
#include <hydra/detail/Range2.inl>
#include <hydra/detail/CountingRange.inl>
#include <hydra/detail/ConstantRange.inl>
#include <hydra/detail/RandomRange.inl>
#include <hydra/detail/PhaseSpaceRange.inl>
#include <hydra/detail/BooststrappedRange.inl>

#endif /* RANGE_H_ */
