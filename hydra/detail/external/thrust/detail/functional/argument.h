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

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/tuple.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{
namespace functional
{

template<unsigned int i, typename Env>
  struct argument_helper
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::tuple_element<i,Env>::type type;
};

template<unsigned int i>
  struct argument_helper<i,HYDRA_EXTERNAL_NS::thrust::null_type>
{
  typedef HYDRA_EXTERNAL_NS::thrust::null_type type;
};


template<unsigned int i>
  class argument
{
  public:
    template<typename Env>
      struct result
        : argument_helper<i,Env>
    {
    };

    __hydra_host__ __hydra_device__
    argument(void){}

    template<typename Env>
    __hydra_host__ __hydra_device__
    typename result<Env>::type eval(const Env &e) const
    {
      return HYDRA_EXTERNAL_NS::thrust::get<i>(e);
    } // end eval()
}; // end argument

} // end functional
} // end detail
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END

