/*
 *  Copyright 2021 NVIDIA Corporation
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

/**
 * \file namespace.h
 * \brief Utilities that allow `hydra_thrust::` to be placed inside an
 * application-specific namespace.
 */

/**
 * \def HYDRA_THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `hydra_thrust::` and `cub::` namespaces.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef HYDRA_THRUST_CUB_WRAPPED_NAMESPACE
#define HYDRA_THRUST_WRAPPED_NAMESPACE HYDRA_THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def HYDRA_THRUST_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `hydra_thrust::` namespace.
 * If HYDRA_THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef HYDRA_THRUST_WRAPPED_NAMESPACE
#define HYDRA_THRUST_NS_PREFIX                                                       \
  namespace HYDRA_THRUST_WRAPPED_NAMESPACE                                           \
  {

#define HYDRA_THRUST_NS_POSTFIX }

#define HYDRA_THRUST_NS_QUALIFIER ::HYDRA_THRUST_WRAPPED_NAMESPACE::hydra_thrust
#endif

/**
 * \def HYDRA_THRUST_NS_PREFIX
 * This macro is inserted prior to all `namespace hydra_thrust { ... }` blocks. It is
 * derived from HYDRA_THRUST_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case HYDRA_THRUST_NS_PREFIX,
 * HYDRA_THRUST_NS_POSTFIX, and HYDRA_THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef HYDRA_THRUST_NS_PREFIX
#define HYDRA_THRUST_NS_PREFIX
#endif

/**
 * \def HYDRA_THRUST_NS_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace hydra_thrust { ... }` block. It is defined appropriately when
 * HYDRA_THRUST_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case HYDRA_THRUST_NS_PREFIX, HYDRA_THRUST_NS_POSTFIX, and
 * HYDRA_THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef HYDRA_THRUST_NS_POSTFIX
#define HYDRA_THRUST_NS_POSTFIX
#endif

/**
 * \def HYDRA_THRUST_NS_QUALIFIER
 * This macro is used to qualify members of hydra_thrust:: when accessing them from
 * outside of their namespace. By default, this is just `::hydra_thrust`, and will be
 * set appropriately when HYDRA_THRUST_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case HYDRA_THRUST_NS_PREFIX, HYDRA_THRUST_NS_POSTFIX, and
 * HYDRA_THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef HYDRA_THRUST_NS_QUALIFIER
#define HYDRA_THRUST_NS_QUALIFIER ::hydra_thrust
#endif

/**
 * \def HYDRA_THRUST_NAMESPACE_BEGIN
 * This macro is used to open a `hydra_thrust::` namespace block, along with any
 * enclosing namespaces requested by HYDRA_THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define HYDRA_THRUST_NAMESPACE_BEGIN                                                 \
  HYDRA_THRUST_NS_PREFIX                                                             \
  namespace hydra_thrust                                                             \
  {

/**
 * \def HYDRA_THRUST_NAMESPACE_END
 * This macro is used to close a `hydra_thrust::` namespace block, along with any
 * enclosing namespaces requested by HYDRA_THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define HYDRA_THRUST_NAMESPACE_END                                                   \
  } /* end namespace hydra_thrust */                                                 \
  HYDRA_THRUST_NS_POSTFIX

// The following is just here to add docs for the hydra_thrust namespace:

HYDRA_THRUST_NS_PREFIX


//#define CUB_NS_PREFIX namespace hydra_thrust {   namespace cuda_cub {

//#define CUB_NS_POSTFIX }  }

//#define CUB_NS_QUALIFIER ::hydra_thrust::cuda_cub::cub
/*! \namespace hydra_thrust
 *  \brief \p hydra_thrust is the top-level namespace which contains all Thrust
 *         functions and types.
 */
namespace hydra_thrust
{
}

HYDRA_THRUST_NS_POSTFIX
