/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#include <utility> // for std::declval

#define __HYDRA_THRUST_DEFINE_HAS_MEMBER_FUNCTION(trait_name, member_function_name)  \
  template <typename T, typename Signature, typename = void>                   \
  struct trait_name : hydra_thrust::false_type                                       \
  {};                                                                          \
                                                                               \
  template <typename T, typename ResultT, typename... Args>                    \
  struct trait_name<T,                                                         \
                    ResultT(Args...),                                          \
                    typename hydra_thrust::detail::enable_if<                        \
                      hydra_thrust::detail::is_same<ResultT, void>::value ||         \
                      hydra_thrust::detail::is_convertible<                          \
                        ResultT,                                               \
                        decltype(std::declval<T>().member_function_name(       \
                          std::declval<Args>()...))>::value>::type>            \
      : hydra_thrust::true_type                                                      \
  {};
