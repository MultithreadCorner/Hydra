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

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

// XXX the order of these #includes matters

#include <hydra/detail/external/thrust/detail/config/simple_defines.h>
#include <hydra/detail/external/thrust/detail/config/compiler.h>
// host_system.h & device_system.h must be #included as early as possible
// because other config headers depend on it
#include <hydra/detail/external/thrust/detail/config/host_system.h>
#include <hydra/detail/external/thrust/detail/config/device_system.h>
#include <hydra/detail/external/thrust/detail/config/host_device.h>
#include <hydra/detail/external/thrust/detail/config/debug.h>
#include <hydra/detail/external/thrust/detail/config/compiler_fence.h>
#include <hydra/detail/external/thrust/detail/config/forceinline.h>
#include <hydra/detail/external/thrust/detail/config/exec_check_disable.h>
#include <hydra/detail/external/thrust/detail/config/global_workarounds.h>

