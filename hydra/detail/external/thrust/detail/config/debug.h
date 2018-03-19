/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#ifndef HYDRA_THRUST_DEBUG
#  ifndef NDEBUG
#    if defined(DEBUG) || defined(_DEBUG)
#      define HYDRA_THRUST_DEBUG 1
#    endif // (DEBUG || _DEBUG)
#  endif // NDEBUG
#endif // HYDRA_THRUST_DEBUG

#if HYDRA_THRUST_DEBUG
#  ifndef __HYDRA_THRUST_SYNCHRONOUS
#    define __HYDRA_THRUST_SYNCHRONOUS 1
#  endif // __HYDRA_THRUST_SYNCHRONOUS
#endif // HYDRA_THRUST_DEBUG

