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
 * SafeCompare.h
 *
 *  Created on: 14/08/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SAFECOMPARE_H_
#define SAFECOMPARE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/MachineEpsilon.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace hydra {

namespace detail {


//implements relative method - do not use for comparing with zero
//use this most of the time, tolerance needs to be meaningful in your context
template<typename T>
__hydra_host__ __hydra_device__
inline bool ApproximatelyEqual(T a, T b, T tolerance = machine_eps_f64())
{
    T diff = ::fabs(a - b);
    if (diff <= tolerance)
        return true;

    if (diff < ::fmax(::fabs(a), ::fabs(b)) * tolerance)
        return true;

    return false;
}

//supply tolerance that is meaningful in your context
//for example, default tolerance may not work if you are comparing double with float
template<typename T>
__hydra_host__ __hydra_device__
inline bool ApproximatelyZero(T a, T tolerance = machine_eps_f64())
{
    if (::fabs(a) <= tolerance)
        return true;
    return false;
}


//use this when you want to be on safe side
//for example, don't start rover unless signal is above 1
template<typename T>
__hydra_host__ __hydra_device__
inline bool SafeLessThan(T a, T b, T tolerance = machine_eps_f64())
{
    T diff = a - b;
    if (diff < tolerance)
        return true;

    if (diff < ::fmax(::fabs(a), ::fabs(b)) * tolerance)
        return true;

    return false;
}
template<typename T>
__hydra_host__ __hydra_device__
inline bool SafeGreaterThan(T a, T b, T tolerance = machine_eps_f64())
{
    T diff = a - b;
    if (diff > tolerance)
        return true;

    if (diff > ::fmax(::fabs(a), ::fabs(b)) * tolerance)
        return true;

    return false;
}

//implements ULP method
//use this when you are only concerned about floating point precision issue
//for example, if you want to see if a is 1.0 by checking if its within
//10 closest representable floating point numbers around 1.0.
template<typename T>
__hydra_host__ __hydra_device__
inline bool WithinPrecisionInterval( T b, T a, unsigned int interval_size = 1)
{
    T min_a = a - (a - ::nextafter(a, std::numeric_limits<T>::lowest())) * interval_size;
    T max_a = a + (::nextafter(a, std::numeric_limits<T>::max()) - a) * interval_size;

    return min_a <= b && max_a >= b;
}


}  // namespace detail

}  // namespace hydra




#endif /* SAFECOMPARE_H_ */
