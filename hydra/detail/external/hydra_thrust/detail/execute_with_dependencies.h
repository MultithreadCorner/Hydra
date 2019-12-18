/*
 *  Copyright 2018 NVIDIA Corporation
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
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>

#include <tuple>
#include <type_traits>

namespace hydra_thrust
{
namespace detail
{

struct capture_as_dependency_fn
{
  template<typename Dependency>
  auto operator()(Dependency&& dependency) const
  HYDRA_THRUST_DECLTYPE_RETURNS(capture_as_dependency(HYDRA_THRUST_FWD(dependency)))
};

// Default implementation: universal forwarding.
template<typename Dependency>
auto capture_as_dependency(Dependency&& dependency)
HYDRA_THRUST_DECLTYPE_RETURNS(HYDRA_THRUST_FWD(dependency))

template<typename... Dependencies>
auto capture_as_dependency(std::tuple<Dependencies...>& dependencies)
HYDRA_THRUST_DECLTYPE_RETURNS(
  tuple_for_each(HYDRA_THRUST_FWD(dependencies), capture_as_dependency_fn{})
)

template<template<typename> class BaseSystem, typename... Dependencies>
struct execute_with_dependencies
    : BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>
{
private:
    using super_t = BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>;

    std::tuple<remove_cvref_t<Dependencies>...> dependencies;

public:
    __host__
    execute_with_dependencies(super_t const &super, Dependencies && ...dependencies)
        : super_t(super), dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(super_t const &super, UDependencies && ...deps)
        : super_t(super), dependencies(HYDRA_THRUST_FWD(deps)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(UDependencies && ...deps)
        : dependencies(HYDRA_THRUST_FWD(deps)...)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(super_t const &super, std::tuple<UDependencies...>&& deps)
        : super_t(super), dependencies(std::move(deps))
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_dependencies(std::tuple<UDependencies...>&& deps)
        : dependencies(std::move(deps))
    {
    }

    std::tuple<remove_cvref_t<Dependencies>...>
    __host__
    extract_dependencies() 
    {
        return std::move(dependencies);
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(UDependencies&& ...udependencies) const
    {
        return { capture_as_dependency(HYDRA_THRUST_FWD(udependencies))... };
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>& udependencies) const
    {
        return { capture_as_dependency(udependencies) };
    }
    template<typename ...UDependencies>
    __host__
    execute_with_dependencies<BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>&& udependencies) const
    {
        return { capture_as_dependency(std::move(udependencies)) };
    }
};

template<
    typename Allocator,
    template<typename> class BaseSystem,
    typename... Dependencies
>
struct execute_with_allocator_and_dependencies
    : BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >
{
private:
    using super_t = BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >;

    std::tuple<remove_cvref_t<Dependencies>...> dependencies;
    Allocator alloc;

public:
    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(super_t const &super, Allocator a, UDependencies && ...deps)
        : super_t(super), dependencies(HYDRA_THRUST_FWD(deps)...), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(Allocator a, UDependencies && ...deps)
        : dependencies(HYDRA_THRUST_FWD(deps)...), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(super_t const &super, Allocator a, std::tuple<UDependencies...>&& deps)
        : super_t(super), dependencies(std::move(deps)), alloc(a)
    {
    }

    template <typename... UDependencies>
    __host__
    execute_with_allocator_and_dependencies(Allocator a, std::tuple<UDependencies...>&& deps)
        : dependencies(std::move(deps)), alloc(a)
    {
    }

    std::tuple<remove_cvref_t<Dependencies>...>
    __host__
    extract_dependencies() 
    {
        return std::move(dependencies);
    }

    typename std::remove_reference<Allocator>::type&
    __host__
    get_allocator()
    {
        return alloc;
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(UDependencies&& ...udependencies) const
    {
        return { alloc, capture_as_dependency(HYDRA_THRUST_FWD(udependencies))... };
    }

    // Rebinding.
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>& udependencies) const
    {
        return { alloc, capture_as_dependency(udependencies) };
    }
    template<typename ...UDependencies>
    __host__
    execute_with_allocator_and_dependencies<Allocator, BaseSystem, UDependencies...>
    rebind_after(std::tuple<UDependencies...>&& udependencies) const
    {
        return { alloc, capture_as_dependency(std::move(udependencies)) };
    }
};

template<template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(hydra_thrust::detail::execute_with_dependencies<BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}
template<template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(hydra_thrust::detail::execute_with_dependencies<BaseSystem, Dependencies...>& system)
{
    return std::move(system).extract_dependencies();
}

template<typename Allocator, template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(hydra_thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}
template<typename Allocator, template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<remove_cvref_t<Dependencies>...>
extract_dependencies(hydra_thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system)
{
    return std::move(system).extract_dependencies();
}

template<typename System>
__host__
std::tuple<>
extract_dependencies(System &&)
{
    return std::tuple<>{};
}

} // end detail
} // end hydra_thrust

#endif // HYDRA_THRUST_CPP_DIALECT >= 2011

