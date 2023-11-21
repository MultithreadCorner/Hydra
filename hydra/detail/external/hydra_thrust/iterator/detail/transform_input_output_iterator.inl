/*
 *  Copyright 2020-2021 NVIDIA Corporation
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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

HYDRA_THRUST_NAMESPACE_BEGIN

template <typename InputFunction, typename OutputFunction, typename Iterator>
  class transform_input_output_iterator;

namespace detail
{

// Proxy reference that invokes InputFunction when reading from and
// OutputFunction when writing to the dereferenced iterator
template <typename InputFunction, typename OutputFunction, typename Iterator>
  class transform_input_output_iterator_proxy
{
  using iterator_value_type = typename hydra_thrust::iterator_value<Iterator>::type;

  using Value = invoke_result_t<InputFunction, iterator_value_type>;

  public:
    __host__ __device__
    transform_input_output_iterator_proxy(const Iterator& io, InputFunction input_function, OutputFunction output_function)
      : io(io), input_function(input_function), output_function(output_function)
    {
    }

    transform_input_output_iterator_proxy(const transform_input_output_iterator_proxy&) = default;

    __hydra_thrust_exec_check_disable__
    __host__ __device__
    operator Value const() const
    {
      return input_function(*io);
    }

    __hydra_thrust_exec_check_disable__
    template <typename T>
    __host__ __device__
    transform_input_output_iterator_proxy operator=(const T& x)
    {
      *io = output_function(x);
      return *this;
    }

    __hydra_thrust_exec_check_disable__
    __host__ __device__
    transform_input_output_iterator_proxy operator=(const transform_input_output_iterator_proxy& x)
    {
      *io = output_function(x);
      return *this;
    }

  private:
    Iterator io;
    InputFunction input_function;
    OutputFunction output_function;
};

// Compute the iterator_adaptor instantiation to be used for transform_input_output_iterator
template <typename InputFunction, typename OutputFunction, typename Iterator>
struct transform_input_output_iterator_base
{
private:
  using iterator_value_type = typename hydra_thrust::iterator_value<Iterator>::type;

public:
    typedef hydra_thrust::iterator_adaptor
    <
        transform_input_output_iterator<InputFunction, OutputFunction, Iterator>
      , Iterator
      , detail::invoke_result_t<InputFunction, iterator_value_type>
      , hydra_thrust::use_default
      , hydra_thrust::use_default
      , transform_input_output_iterator_proxy<InputFunction, OutputFunction, Iterator>
    > type;
};

// Register transform_input_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <typename InputFunction, typename OutputFunction, typename Iterator>
struct is_proxy_reference<
    transform_input_output_iterator_proxy<InputFunction, OutputFunction, Iterator> >
    : public hydra_thrust::detail::true_type {};

} // end detail
HYDRA_THRUST_NAMESPACE_END

