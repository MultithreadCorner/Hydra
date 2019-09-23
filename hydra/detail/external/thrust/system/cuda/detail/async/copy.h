/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// TODO: Move into system::cuda

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/cpp11_required.h>
#include <hydra/detail/external/thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC

#include <hydra/detail/external/thrust/system/cuda/config.h>

#include <hydra/detail/external/thrust/system/cuda/detail/async/customization.h>
#include <hydra/detail/external/thrust/system/cuda/detail/async/transform.h>
#include <hydra/detail/external/thrust/system/cuda/detail/cross_system.h>
#include <hydra/detail/external/thrust/system/cuda/future.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/type_traits/logical_metafunctions.h>
#include <hydra/detail/external/thrust/detail/static_assert.h>
#include <hydra/detail/external/thrust/type_traits/is_trivially_relocatable.h>
#include <hydra/detail/external/thrust/type_traits/is_contiguous_iterator.h>
#include <hydra/detail/external/thrust/distance.h>
#include <hydra/detail/external/thrust/advance.h>
#include <hydra/detail/external/thrust/uninitialized_copy.h>

#include <type_traits>
HYDRA_EXTERNAL_NAMESPACE_BEGIN
HYDRA_THRUST_BEGIN_NS

namespace system { namespace cuda { namespace detail
{

// ContiguousIterator input and output iterators
// TriviallyCopyable elements
// Host to device, device to host, device to device
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  FromPolicy& from_exec
, ToPolicy&   to_exec
, ForwardIt   first
, Size        n
, OutputIt    output
) ->
  typename std::enable_if<
    is_indirectly_trivially_relocatable_to<ForwardIt, OutputIt>::value
  , unique_eager_event
  >::type
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  auto const device_alloc = get_async_device_allocator(
    select_device_system(from_exec, to_exec)
  );

  using pointer
    = typename HYDRA_EXTERNAL_NS::thrust::detail::allocator_traits<decltype(device_alloc)>::
      template rebind_traits<void>::pointer;

  unique_eager_event e;

  // Set up stream with dependencies.

  cudaStream_t const user_raw_stream = HYDRA_EXTERNAL_NS::thrust::cuda_cub::stream(
    select_device_system(from_exec, to_exec)
  );

  if (HYDRA_EXTERNAL_NS::thrust::cuda_cub::default_stream() != user_raw_stream)
  {
    e = make_dependent_event(
      std::tuple_cat(
        std::make_tuple(
          unique_stream(nonowning, user_raw_stream)
        )
      , extract_dependencies(
          std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(from_exec))
        )
      , extract_dependencies(
          std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(to_exec))
        )
      )
    );
  }
  else
  {
    e = make_dependent_event(
      std::tuple_cat(
        extract_dependencies(
          std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(from_exec))
        )
      , extract_dependencies(
          std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(to_exec))
        )
      )
    );
  }

  // Run copy.

  HYDRA_EXTERNAL_NS::thrust::cuda_cub::throw_on_error(
    cudaMemcpyAsync(
      HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(&*output)
    , HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(&*first)
    , sizeof(T) * n
    , direction_of_copy(from_exec, to_exec)
    , e.stream().native_handle()
    )
  , "after copy launch"
  );

  return e;
}

// Non-ContiguousIterator input or output, or non-TriviallyRelocatable value type
// Device to device
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<FromPolicy>& from_exec
, HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<ToPolicy>&   to_exec
, ForwardIt                                   first
, Size                                        n
, OutputIt                                    output
) ->
  typename std::enable_if<
    conjunction<
      negation<
        is_indirectly_trivially_relocatable_to<ForwardIt, OutputIt>
      >
    , decltype(is_device_to_device_copy(from_exec, to_exec))
    >::value
  , unique_eager_event
  >::type
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  return async_transform_n(
    select_device_system(from_exec, to_exec)
  , first, n, output, HYDRA_EXTERNAL_NS::thrust::identity<T>()
  );
}

template <typename OutputIt>
void async_copy_n_compile_failure_no_cuda_to_non_contiguous_output()
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (negation<is_contiguous_iterator<OutputIt>>::value)
  , "copying to non-ContiguousIterators in another system from the CUDA system "
    "is not supported; use `HYDRA_THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)` to "
    "indicate that an iterator points to elements that are contiguous in memory."
  );
}

// Non-ContiguousIterator output iterator
// TriviallyRelocatable value type
// Device to host, host to device
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  FromPolicy& from_exec
, ToPolicy&   to_exec
, ForwardIt   first
, Size        n
, OutputIt    output
) ->
  typename std::enable_if<
    conjunction<
      negation<is_contiguous_iterator<OutputIt>>
    , is_trivially_relocatable_to<
        typename iterator_traits<ForwardIt>::value_type
      , typename iterator_traits<OutputIt>::value_type
      >
    , disjunction<
        decltype(is_host_to_device_copy(from_exec, to_exec))
      , decltype(is_device_to_host_copy(from_exec, to_exec))
      >
    >::value
  , unique_eager_event
  >::type
{
  async_copy_n_compile_failure_no_cuda_to_non_contiguous_output<OutputIt>();

  return {};
}

// Workaround for MSVC's lack of expression SFINAE and also for an NVCC bug.
// In NVCC, when two SFINAE-enabled overloads are only distinguishable by a
// part of a SFINAE condition that is in a `decltype`, NVCC thinks they are the
// same overload and emits an error.
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt
>
struct is_buffered_trivially_relocatable_host_to_device_copy
  : HYDRA_EXTERNAL_NS::thrust::integral_constant<
      bool
    ,    !is_contiguous_iterator<ForwardIt>::value
      && is_contiguous_iterator<OutputIt>::value
      && is_trivially_relocatable_to<
            typename iterator_traits<ForwardIt>::value_type
          , typename iterator_traits<OutputIt>::value_type
          >::value
      && decltype(
           is_host_to_device_copy(
             std::declval<FromPolicy const&>()
           , std::declval<ToPolicy const&>()
           )
         )::value
    >
{};

// Non-ContiguousIterator input iterator, ContiguousIterator output iterator
// TriviallyRelocatable value type
// Host to device
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  FromPolicy&                               from_exec
, HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<ToPolicy>& to_exec
, ForwardIt                                 first
, Size                                      n
, OutputIt                                  output
) ->
  typename std::enable_if<
    is_buffered_trivially_relocatable_host_to_device_copy<
      FromPolicy
    , HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<ToPolicy>
    , ForwardIt, OutputIt
    >::value
  , unique_eager_event
  >::type
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  auto const host_alloc = get_async_host_allocator(
    from_exec
  );

  // Create host-side buffer.

  auto buffer = uninitialized_allocate_unique_n<T>(host_alloc, n);

  auto const buffer_ptr = buffer.get();

  // Copy into host-side buffer.

  // TODO: Switch to an async call once we have async interfaces for host
  // systems and support for cross system dependencies.
  uninitialized_copy_n(from_exec, first, n, buffer_ptr);

  // Run device-side copy.

  auto new_to_exec = HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(to_exec).rebind_after(
    std::tuple_cat(
      std::make_tuple(
        std::move(buffer)
      )
    , extract_dependencies(
        std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(from_exec))
      )
    , extract_dependencies(
        std::move(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(to_exec))
      )
    )
  );

  HYDRA_THRUST_STATIC_ASSERT((
    std::tuple_size<decltype(
      extract_dependencies(to_exec)
    )>::value + 1
    <=
    std::tuple_size<decltype(
      extract_dependencies(new_to_exec)
    )>::value
  ));

  return async_copy_n(
    from_exec
    // TODO: We have to cast back to the right execution_policy class. Ideally,
    // we should be moving here.
  , new_to_exec
  , buffer_ptr
  , n
  , output
  );
}

// Workaround for MSVC's lack of expression SFINAE and also for an NVCC bug.
// In NVCC, when two SFINAE-enabled overloads are only distinguishable by a
// part of a SFINAE condition that is in a `decltype`, NVCC thinks they are the
// same overload and emits an error.
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt
>
struct is_buffered_trivially_relocatable_device_to_host_copy
  : HYDRA_EXTERNAL_NS::thrust::integral_constant<
      bool
    ,    !is_contiguous_iterator<ForwardIt>::value
      && is_contiguous_iterator<OutputIt>::value
      && is_trivially_relocatable_to<
            typename iterator_traits<ForwardIt>::value_type
          , typename iterator_traits<OutputIt>::value_type
          >::value
      && decltype(
           is_device_to_host_copy(
             std::declval<FromPolicy const&>()
           , std::declval<ToPolicy const&>()
           )
         )::value
    >
{};

// Non-ContiguousIterator input iterator, ContiguousIterator output iterator
// TriviallyRelocatable value type
// Device to host
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<FromPolicy>& from_exec
, ToPolicy&                                   to_exec
, ForwardIt                                   first
, Size                                        n
, OutputIt                                    output
) ->
  typename std::enable_if<
    is_buffered_trivially_relocatable_device_to_host_copy<
      HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<FromPolicy>
    , ToPolicy
    , ForwardIt, OutputIt
    >::value
  , unique_eager_event
  >::type
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  auto const device_alloc = get_async_device_allocator(
    from_exec
  );

  // Create device-side buffer.

  auto buffer = uninitialized_allocate_unique_n<T>(device_alloc, n);

  auto const buffer_ptr = buffer.get();

  // Run device-side copy.

  auto f0 = async_copy_n(
    from_exec
  , from_exec
  , first
  , n
  , buffer_ptr
  );

  // Run copy back to host.

  auto new_from_exec = HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(from_exec).rebind_after(
    std::move(buffer)
  , std::move(f0)
  );

  HYDRA_THRUST_STATIC_ASSERT((
    std::tuple_size<decltype(
      extract_dependencies(from_exec)
    )>::value + 1
    <=
    std::tuple_size<decltype(
      extract_dependencies(new_from_exec)
    )>::value
  ));

  return async_copy_n(
    new_from_exec
  , to_exec
  , buffer_ptr
  , n
  , output
  );
}

template <typename InputType, typename OutputType>
void async_copy_n_compile_failure_non_trivially_relocatable_elements()
{
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (is_trivially_relocatable_to<OutputType, InputType>::value)
  , "only sequences of TriviallyRelocatable elements can be copied to and from "
    "the CUDA system; use `HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(T)` to "
    "indicate that a type can be copied by bitwise (e.g. by `memcpy`)"
  );
}

// Non-TriviallyRelocatable value type
// Host to device, device to host
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename OutputIt, typename Size
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy_n(
  FromPolicy& from_exec
, ToPolicy&   to_exec
, ForwardIt   first
, Size        n
, OutputIt    output
) ->
  typename std::enable_if<
    conjunction<
      negation<
        is_trivially_relocatable_to<
          typename iterator_traits<ForwardIt>::value_type
        , typename iterator_traits<OutputIt>::value_type
        >
      >
    , disjunction<
        decltype(is_host_to_device_copy(from_exec, to_exec))
      , decltype(is_device_to_host_copy(from_exec, to_exec))
      >
    >::value
  , unique_eager_event
  >::type
{
  // TODO: We could do more here with cudaHostRegister.

  async_copy_n_compile_failure_non_trivially_relocatable_elements<
    typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIt>::value_type
  , typename std::add_lvalue_reference<
      typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIt>::value_type
    >::type
  >();

  return {};
}

}}} // namespace system::cuda::detail

namespace cuda_cub
{

// ADL entry point.
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy(
  HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<FromPolicy>&         from_exec
, HYDRA_EXTERNAL_NS::thrust::cpp::execution_policy<ToPolicy>&            to_exec
, ForwardIt                                           first
, Sentinel                                            last
, OutputIt                                            output
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  HYDRA_EXTERNAL_NS::thrust::system::cuda::detail::async_copy_n(
    from_exec, to_exec, first, distance(first, last), output
  )
)

// ADL entry point.
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy(
  HYDRA_EXTERNAL_NS::thrust::cpp::execution_policy<FromPolicy>& from_exec
, HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<ToPolicy>&  to_exec
, ForwardIt                                  first
, Sentinel                                   last
, OutputIt                                   output
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  HYDRA_EXTERNAL_NS::thrust::system::cuda::detail::async_copy_n(
    from_exec, to_exec, first, distance(first, last), output
  )
)

// ADL entry point.
template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
>
HYDRA_THRUST_RUNTIME_FUNCTION
auto async_copy(
  HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<FromPolicy>& from_exec
, HYDRA_EXTERNAL_NS::thrust::cuda::execution_policy<ToPolicy>&   to_exec
, ForwardIt                                   first
, Sentinel                                    last
, OutputIt                                    output
)
HYDRA_THRUST_DECLTYPE_RETURNS(
  HYDRA_EXTERNAL_NS::thrust::system::cuda::detail::async_copy_n(
    from_exec, to_exec, first, distance(first, last), output
  )
)

} // cuda_cub

HYDRA_THRUST_END_NS
HYDRA_EXTERNAL_NAMESPACE_END
#endif // HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC

#endif

