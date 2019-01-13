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
 * MachineEpsilon.h
 *
 *  Created on: 16/08/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MACHINEEPSILON_H_
#define MACHINEEPSILON_H_

#include <hydra/detail/Config.h>

namespace hydra {

namespace detail {

__hydra_host__ __hydra_device__
inline double machine_eps_f64(double value=1.0) {

    typedef union {long long i64; double f64; } f64_t;

    f64_t s;

    s.f64 = value;
    s.i64++;

    return (s.f64 - value);
}

__hydra_host__ __hydra_device__
inline float machine_eps_f32(float value=1.0) {

    typedef union {int i32; float f32;} f32_t;

    f32_t s;

    s.f32 = value;
    s.i32++;
    return (s.f32 - value);
}


}  // namespace detail

}  // namespace hydra


#endif /* MACHINEEPSILON_H_ */
