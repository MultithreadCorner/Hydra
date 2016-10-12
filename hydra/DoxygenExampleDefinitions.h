
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
 * DoxygenExampleDefinitions.h
 *
 *  Created on: 12/10/2016
 *      Author: augalves
 */

#ifndef DOXYGENEXAMPLEDEFINITIONS_H_
#define DOXYGENEXAMPLEDEFINITIONS_H_


/// @file
/// @brief Pseudo header file that does not contain code but serves as
/// a container for Doxygen example documentation blocks.

/**
 * @example HydraEvaluateExample.cu
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas
 * using the CUDA backend.
 * The usage and the expected output is something like this:
 ```
 ./Hydra_Example_NVCC_DEVICE_CUDA_HOST_OMP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 0.268075
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 0.213697
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.252592278 -0.967572809) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.932195527 -0.361955107) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (0.585050502 0.810996862) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.851355528 0.524589139) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (0.921620144 0.388093171) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.989926331 0.141583398) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.775938693 -0.630808327) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.879555884 0.475795593) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (0.964678017 0.263431819) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (-0.999776981 -0.0211184451) ............... [sin(x)^2 + cos(x)^2] = 1

 ```
*/


/**
 * @example HydraEvaluateExample.cpp
 * @brief This is an example of how to use hydra::Eval to evaluate C++11 lambdas using the OpenMP backend.
 * The usage and the expected output is something like this:
```
./Hydra_Example_GCC_DEVICE_OMP_HOST_CPP_Eval -n=10000000

--------------------------------------------------------------
| Evaluation of [sin(x), cos(x)]
| Time (ms) = 534.884
--------------------------------------------------------------
--------------------------------------------------------------
| Evaluation of [sin(x)^2 + cos(x)^2]
| Time (ms) = 21.6937
--------------------------------------------------------------
|>   0 [sin(x), cos(x)] = (-0.303346204 -0.952880412) ............... [sin(x)^2 + cos(x)^2] = 1
|>   1 [sin(x), cos(x)] = (0.974836209 -0.222922333) ............... [sin(x)^2 + cos(x)^2] = 1
|>   2 [sin(x), cos(x)] = (-0.69576933 0.718265299) ............... [sin(x)^2 + cos(x)^2] = 1
|>   3 [sin(x), cos(x)] = (0.853703285 -0.520759734) ............... [sin(x)^2 + cos(x)^2] = 1
|>   4 [sin(x), cos(x)] = (-0.941210424 0.337820866) ............... [sin(x)^2 + cos(x)^2] = 1
|>   5 [sin(x), cos(x)] = (-0.8711111 -0.491085992) ............... [sin(x)^2 + cos(x)^2] = 1
|>   6 [sin(x), cos(x)] = (-0.0704802093 -0.997513178) ............... [sin(x)^2 + cos(x)^2] = 1
|>   7 [sin(x), cos(x)] = (0.490961891 -0.87118105) ............... [sin(x)^2 + cos(x)^2] = 1
|>   8 [sin(x), cos(x)] = (-0.78756139 0.616236202) ............... [sin(x)^2 + cos(x)^2] = 1
|>   9 [sin(x), cos(x)] = (0.715995247 0.698105154) ............... [sin(x)^2 + cos(x)^2] = 1

```
 */




#endif /* DOXYGENEXAMPLEDEFINITIONS_H_ */
