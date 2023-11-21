#
# find_package(Thrust) config file.
#
# Provided by NVIDIA under the same license as the associated Thrust library.
#
# Reply-To: Allison Vacanti <alliepiper16@gmail.com>
#
# *****************************************************************************
# **     The following is a short reference to using Thrust from CMake.      **
# ** For more details, see the README.md in the same directory as this file. **
# *****************************************************************************
#
# # General Usage:
# find_package(Thrust REQUIRED CONFIG)
# hydra_thrust_create_target(Thrust [options])
# target_link_libraries(some_project_lib Thrust)
#
# # Create default target with: HOST=CPP DEVICE=CUDA
# hydra_thrust_create_target(TargetName)
#
# # Create target with: HOST=CPP DEVICE=TBB
# hydra_thrust_create_target(TargetName DEVICE TBB)
#
# # Create target with: HOST=TBB DEVICE=OMP
# hydra_thrust_create_target(TargetName HOST TBB DEVICE OMP)
#
# # Create CMake cache options HYDRA_THRUST_[HOST|DEVICE]_SYSTEM and configure a
# # target from them. This allows these systems to be changed by developers at
# # configure time, per build.
# hydra_thrust_create_target(TargetName FROM_OPTIONS
#   [HOST_OPTION <option_name>]      # Optionally rename the host system option
#   [DEVICE_OPTION <option_name>]    # Optionally rename the device system option
#   [HOST_OPTION_DOC <doc_string>]   # Optionally change the cache label
#   [DEVICE_OPTION_DOC <doc_string>] # Optionally change the cache label
#   [HOST <default system>]          # Optionally change the default backend
#   [DEVICE <default system>]        # Optionally change the default backend
#   [ADVANCED]                       # Optionally mark options as advanced
# )
#
# # Use a custom TBB, CUB, and/or OMP
# # (Note that once set, these cannot be changed. This includes COMPONENT
# # preloading and lazy lookups in hydra_thrust_create_target)
# find_package(Thrust REQUIRED)
# hydra_thrust_set_CUB_target(MyCUBTarget)  # MyXXXTarget contains an existing
# hydra_thrust_set_TBB_target(MyTBBTarget)  # interface to XXX for Thrust to use.
# hydra_thrust_set_OMP_target(MyOMPTarget)
# hydra_thrust_create_target(ThrustWithMyCUB DEVICE CUDA)
# hydra_thrust_create_target(ThrustWithMyTBB DEVICE TBB)
# hydra_thrust_create_target(ThrustWithMyOMP DEVICE OMP)
#
# # Create target with HOST=CPP DEVICE=CUDA and some advanced flags set
# hydra_thrust_create_target(TargetName
#   IGNORE_DEPRECATED_API         # Silence build warnings about deprecated APIs
#   IGNORE_DEPRECATED_CPP_DIALECT # Silence build warnings about deprecated compilers and C++ standards
#   IGNORE_DEPRECATED_CPP_11      # Only silence deprecation warnings for C++11
#   IGNORE_DEPRECATED_COMPILER    # Only silence deprecation warnings for old compilers
#   IGNORE_CUB_VERSION            # Skip configure-time and compile-time CUB version checks
# )
#
# # Test if a particular system has been loaded. ${var_name} is set to TRUE or
# # FALSE to indicate if "system" is found.
# hydra_thrust_is_system_found(<system> <var_name>)
# hydra_thrust_is_cuda_system_found(<var_name>)
# hydra_thrust_is_tbb_system_found(<var_name>)
# hydra_thrust_is_omp_system_found(<var_name>)
# hydra_thrust_is_cpp_system_found(<var_name>)
#
# # Define / update HYDRA_THRUST_${system}_FOUND flags in current scope
# hydra_thrust_update_system_found_flags()
#
# # View verbose log with target and dependency information:
# $ cmake . --log-level=VERBOSE (CMake 3.15.7 and above)
#
# # Print debugging output to status channel:
# hydra_thrust_debug_internal_targets()
# hydra_thrust_debug_target(TargetName "${HYDRA_THRUST_VERSION}")

cmake_minimum_required(VERSION 3.15)

# Minimum supported libcudacxx version:
set(hydra_thrust_libcudacxx_version 1.8.0)

################################################################################
# User variables and APIs. Users can rely on these:
#

# Advertise system options:
set(HYDRA_THRUST_HOST_SYSTEM_OPTIONS
  CPP OMP TBB
  CACHE INTERNAL "Valid Thrust host systems."
  FORCE
)
set(HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS
  CUDA CPP OMP TBB
  CACHE INTERNAL "Valid Thrust device systems"
  FORCE
)

# Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
set(HYDRA_THRUST_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "" FORCE)
set(HYDRA_THRUST_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(HYDRA_THRUST_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(HYDRA_THRUST_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(HYDRA_THRUST_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "" FORCE)
set(HYDRA_THRUST_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "" FORCE)

function(hydra_thrust_create_target target_name)
  hydra_thrust_debug("Assembling target ${target_name}. Options: ${ARGN}" internal)
  set(options
    ADVANCED
    FROM_OPTIONS
    IGNORE_CUB_VERSION_CHECK
    IGNORE_DEPRECATED_API
    IGNORE_DEPRECATED_COMPILER
    IGNORE_DEPRECATED_CPP_11
    IGNORE_DEPRECATED_CPP_DIALECT
  )
  set(keys
    DEVICE
    DEVICE_OPTION
    DEVICE_OPTION_DOC
    HOST
    HOST_OPTION
    HOST_OPTION_DOC
  )
  cmake_parse_arguments(TCT "${options}" "${keys}" "" ${ARGN})
  if (TCT_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Unrecognized arguments passed to hydra_thrust_create_target: "
      ${TCT_UNPARSED_ARGUMENTS}
    )
  endif()

  # Check that the main Thrust internal target is available
  # (functions have global scope, targets have directory scope, so this
  # might happen)
  if (NOT TARGET Thrust::Thrust)
    message(AUTHOR_WARNING
      "The `hydra_thrust_create_target` function was called outside the scope of the "
      "hydra_thrust targets. Call find_package again to recreate targets."
    )
  endif()

  _hydra_thrust_set_if_undefined(TCT_HOST CPP)
  _hydra_thrust_set_if_undefined(TCT_DEVICE CUDA)
  _hydra_thrust_set_if_undefined(TCT_HOST_OPTION HYDRA_THRUST_HOST_SYSTEM)
  _hydra_thrust_set_if_undefined(TCT_DEVICE_OPTION HYDRA_THRUST_DEVICE_SYSTEM)
  _hydra_thrust_set_if_undefined(TCT_HOST_OPTION_DOC "Thrust host system.")
  _hydra_thrust_set_if_undefined(TCT_DEVICE_OPTION_DOC "Thrust device system.")

  if (NOT TCT_HOST IN_LIST HYDRA_THRUST_HOST_SYSTEM_OPTIONS)
    message(FATAL_ERROR
      "Requested HOST=${TCT_HOST}; must be one of ${HYDRA_THRUST_HOST_SYSTEM_OPTIONS}"
    )
  endif()

  if (NOT TCT_DEVICE IN_LIST HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS)
    message(FATAL_ERROR
      "Requested DEVICE=${TCT_DEVICE}; must be one of ${HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS}"
    )
  endif()

  if (TCT_FROM_OPTIONS)
    _hydra_thrust_create_cache_options(
      ${TCT_HOST} ${TCT_DEVICE}
      ${TCT_HOST_OPTION} ${TCT_DEVICE_OPTION}
      ${TCT_HOST_OPTION_DOC} ${TCT_DEVICE_OPTION_DOC}
      ${TCT_ADVANCED}
    )
    set(TCT_HOST ${${TCT_HOST_OPTION}})
    set(TCT_DEVICE ${${TCT_DEVICE_OPTION}})
    hydra_thrust_debug("Current option settings:" internal)
    hydra_thrust_debug("  - ${TCT_HOST_OPTION}=${TCT_HOST}" internal)
    hydra_thrust_debug("  - ${TCT_DEVICE_OPTION}=${TCT_DEVICE}" internal)
  endif()

  _hydra_thrust_find_backend(${TCT_HOST} REQUIRED)
  _hydra_thrust_find_backend(${TCT_DEVICE} REQUIRED)

  # We can just create an INTERFACE IMPORTED target here instead of going
  # through _hydra_thrust_declare_interface_alias as long as we aren't hanging any
  # Thrust/CUB include paths directly on ${target_name}.
  add_library(${target_name} INTERFACE IMPORTED)
  target_link_libraries(${target_name}
    INTERFACE
    Thrust::${TCT_HOST}::Host
    Thrust::${TCT_DEVICE}::Device
  )

  # This would be nice to enforce, but breaks when using old cmake + new
  # compiler, since cmake doesn't know what features the new compiler version
  # supports.
  # Leaving this here as a reminder not to add it back. Just let the
  # compile-time checks in hydra_thrust/detail/config/cpp_dialect.h handle it.
  #
  #  if (NOT TCT_IGNORE_DEPRECATED_CPP_DIALECT)
  #    if (TCT_IGNORE_DEPRECATED_CPP_11)
  #      target_compile_features(${target_name} INTERFACE cxx_std_11)
  #    else()
  #      target_compile_features(${target_name} INTERFACE cxx_std_14)
  #    endif()
  #  endif()

  if (TCT_IGNORE_DEPRECATED_CPP_DIALECT)
    target_compile_definitions(${target_name} INTERFACE "HYDRA_THRUST_IGNORE_DEPRECATED_CPP_DIALECT")
  endif()

  if (TCT_IGNORE_DEPRECATED_API)
    target_compile_definitions(${target_name} INTERFACE "HYDRA_THRUST_IGNORE_DEPRECATED_API")
  endif()

  if (TCT_IGNORE_DEPRECATED_CPP_11)
    target_compile_definitions(${target_name} INTERFACE "HYDRA_THRUST_IGNORE_DEPRECATED_CPP_11")
  endif()

  if (TCT_IGNORE_DEPRECATED_COMPILER)
    target_compile_definitions(${target_name} INTERFACE "HYDRA_THRUST_IGNORE_DEPRECATED_COMPILER")
  endif()

  if (TCT_IGNORE_CUB_VERSION_CHECK)
    target_compile_definitions(${target_name} INTERFACE "HYDRA_THRUST_IGNORE_CUB_VERSION_CHECK")
  else()
    if (("${TCT_HOST}" STREQUAL "CUDA" OR "${TCT_DEVICE}" STREQUAL "CUDA") AND
    (NOT HYDRA_THRUST_VERSION VERSION_EQUAL HYDRA_THRUST_CUB_VERSION))
      message(FATAL_ERROR
        "The version of CUB found by CMake is not compatible with this release of Thrust. "
        "CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. "
        "Pass IGNORE_CUB_VERSION_CHECK to hydra_thrust_create_target to ignore. "
        "(CUB ${HYDRA_THRUST_CUB_VERSION}, Thrust ${HYDRA_THRUST_VERSION})."
        )
    endif()
  endif()

  hydra_thrust_debug_target(${target_name} "Thrust ${HYDRA_THRUST_VERSION}"  internal)
endfunction()

function(hydra_thrust_is_system_found system var_name)
  if (TARGET Thrust::${system})
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(hydra_thrust_is_cpp_system_found var_name)
  hydra_thrust_is_system_found(CPP ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(hydra_thrust_is_cuda_system_found var_name)
  hydra_thrust_is_system_found(CUDA ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(hydra_thrust_is_tbb_system_found var_name)
  hydra_thrust_is_system_found(TBB ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(hydra_thrust_is_omp_system_found var_name)
  hydra_thrust_is_system_found(OMP ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

# Since components are loaded lazily, this will refresh the
# HYDRA_THRUST_${component}_FOUND flags in the current scope.
# Alternatively, check system states individually using the
# hydra_thrust_is_system_found functions.
macro(hydra_thrust_update_system_found_flags)
  set(HYDRA_THRUST_FOUND TRUE)
  hydra_thrust_is_system_found(CPP  HYDRA_THRUST_CPP_FOUND)
  hydra_thrust_is_system_found(CUDA HYDRA_THRUST_CUDA_FOUND)
  hydra_thrust_is_system_found(TBB  HYDRA_THRUST_TBB_FOUND)
  hydra_thrust_is_system_found(OMP  HYDRA_THRUST_OMP_FOUND)
endmacro()

function(hydra_thrust_debug msg)
  # Use the VERBOSE channel when called internally
  # Run `cmake . --log-level=VERBOSE` to view.
  if ("${ARGN}" STREQUAL "internal")
    # If CMake is too old to know about the VERBOSE channel, just be silent.
    # Users reproduce much the same output on the STATUS channel by using:
    # hydra_thrust_create_target(Thrust [...])
    # hydra_thrust_debug_internal_targets()
    # hydra_thrust_debug_target(Thrust)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.15.7")
      set(channel VERBOSE)
    else()
      return()
    endif()
  else()
    set(channel STATUS)
  endif()

  message(${channel} "Thrust: ${msg}")
endfunction()

# Print details of the specified target.
function(hydra_thrust_debug_target target_name version)
  if (NOT TARGET ${target_name})
    return()
  endif()

  set(is_internal "${ARGN}")

  if (version)
    set(version "(${version})")
  endif()

  hydra_thrust_debug("TargetInfo: ${target_name}: ${version}" ${is_internal})

  function(_hydra_thrust_print_prop_if_set target_name prop)
    get_target_property(value ${target_name} ${prop})
    if (value)
      hydra_thrust_debug("TargetInfo: ${target_name} > ${prop}: ${value}" ${is_internal})
    endif()
  endfunction()

  function(_hydra_thrust_print_imported_prop_if_set target_name prop)
    get_target_property(imported ${target_name} IMPORTED)
    get_target_property(type ${target_name} TYPE)
    if (imported AND NOT ${type} STREQUAL "INTERFACE_LIBRARY")
      _hydra_thrust_print_prop_if_set(${target_name} ${prop})
    endif()
  endfunction()

  _hydra_thrust_print_prop_if_set(${target_name} ALIASED_TARGET)
  _hydra_thrust_print_prop_if_set(${target_name} IMPORTED)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_DEFINITIONS)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_FEATURES)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_OPTIONS)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_LINK_DEPENDS)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_LINK_DIRECTORIES)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_LINK_LIBRARIES)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_LINK_OPTIONS)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_HYDRA_THRUST_HOST)
  _hydra_thrust_print_prop_if_set(${target_name} INTERFACE_HYDRA_THRUST_DEVICE)
  _hydra_thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION)
  _hydra_thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION_DEBUG)
  _hydra_thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION_RELEASE)
endfunction()

function(hydra_thrust_debug_internal_targets)
  function(_hydra_thrust_debug_backend_targets backend version)
    hydra_thrust_debug_target(Thrust::${backend} "${version}")
    hydra_thrust_debug_target(Thrust::${backend}::Host "${version}")
    hydra_thrust_debug_target(Thrust::${backend}::Device "${version}")
  endfunction()

  hydra_thrust_debug_target(Thrust::Thrust "${HYDRA_THRUST_VERSION}")

  _hydra_thrust_debug_backend_targets(CPP "Thrust ${HYDRA_THRUST_VERSION}")

  _hydra_thrust_debug_backend_targets(OMP "${HYDRA_THRUST_OMP_VERSION}")
  hydra_thrust_debug_target(OpenMP::OpenMP_CXX "${HYDRA_THRUST_OMP_VERSION}")

  _hydra_thrust_debug_backend_targets(TBB "${HYDRA_THRUST_TBB_VERSION}")
  hydra_thrust_debug_target(TBB:tbb "${HYDRA_THRUST_TBB_VERSION}")

  _hydra_thrust_debug_backend_targets(CUDA "CUB ${HYDRA_THRUST_CUB_VERSION}")
  hydra_thrust_debug_target(CUB::CUB "${HYDRA_THRUST_CUB_VERSION}")
  hydra_thrust_debug_target(libcudacxx::libcudacxx "${HYDRA_THRUST_libcudacxx_VERSION}")
endfunction()

################################################################################
# Internal utilities. Subject to change.
#

function(_hydra_thrust_set_if_undefined var)
  if (NOT DEFINED ${var})
    set(${var} ${ARGN} PARENT_SCOPE)
  endif()
endfunction()

function(_hydra_thrust_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) nvcc will automatically check the CUDA Toolkit include path *before* the
  #    system includes. This means that the Toolkit Thrust will *always* be used
  #    during compilation, and the include paths of an IMPORTED Thrust::Thrust
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to Thrust::Thrust. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong hydra_thrust!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)
  add_library(${alias_name} ALIAS ${ugly_name})
endfunction()

# Create cache options for selecting the user/device systems with ccmake/cmake-gui.
function(_hydra_thrust_create_cache_options host device host_option device_option host_doc device_doc advanced)
  hydra_thrust_debug("Creating system cache options: (advanced=${advanced})" internal)
  hydra_thrust_debug("  - Host Option=${host_option} Default=${host} Doc='${host_doc}'" internal)
  hydra_thrust_debug("  - Device Option=${device_option} Default=${device} Doc='${device_doc}'" internal)
  set(${host_option} ${host} CACHE STRING "${host_doc}")
  set_property(CACHE ${host_option} PROPERTY STRINGS ${HYDRA_THRUST_HOST_SYSTEM_OPTIONS})
  set(${device_option} ${device} CACHE STRING "${device_doc}")
  set_property(CACHE ${device_option} PROPERTY STRINGS ${HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS})
  if (advanced)
    mark_as_advanced(${host_option} ${device_option})
  endif()
endfunction()

# Create Thrust::${backend}::Host and Thrust::${backend}::Device targets.
# Assumes that `Thrust::${backend}` and `_Thrust_${backend}` have been created
# by _hydra_thrust_declare_interface_alias and configured to bring in system
# dependency interfaces (including Thrust::Thrust).
function(_hydra_thrust_setup_system backend)
  set(backend_target_alias "Thrust::${backend}")

  if (backend IN_LIST HYDRA_THRUST_HOST_SYSTEM_OPTIONS)
    set(host_target "_Thrust_${backend}_Host")
    set(host_target_alias "Thrust::${backend}::Host")
    if (NOT TARGET ${host_target_alias})
      _hydra_thrust_declare_interface_alias(${host_target_alias} ${host_target})
      target_compile_definitions(${host_target} INTERFACE
        "HYDRA_THRUST_HOST_SYSTEM=HYDRA_THRUST_HOST_SYSTEM_${backend}")
      target_link_libraries(${host_target} INTERFACE ${backend_target_alias})
      set_property(TARGET ${host_target} PROPERTY INTERFACE_HYDRA_THRUST_HOST ${backend})
      set_property(TARGET ${host_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING HYDRA_THRUST_HOST)
      hydra_thrust_debug_target(${host_target_alias} "" internal)
    endif()
  endif()

  if (backend IN_LIST HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS)
    set(device_target "_Thrust_${backend}_Device")
    set(device_target_alias "Thrust::${backend}::Device")
    if (NOT TARGET ${device_target_alias})
      _hydra_thrust_declare_interface_alias(${device_target_alias} ${device_target})
      target_compile_definitions(${device_target} INTERFACE
        "HYDRA_THRUST_DEVICE_SYSTEM=HYDRA_THRUST_DEVICE_SYSTEM_${backend}")
      target_link_libraries(${device_target} INTERFACE ${backend_target_alias})
      set_property(TARGET ${device_target} PROPERTY INTERFACE_HYDRA_THRUST_DEVICE ${backend})
      set_property(TARGET ${device_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING HYDRA_THRUST_DEVICE)
      hydra_thrust_debug_target(${device_target_alias} "" internal)
    endif()
  endif()
endfunction()

# Use the provided cub_target for the CUDA backend. If Thrust::CUB already
# exists, this call has no effect.
function(hydra_thrust_set_CUB_target cub_target)
  if (NOT TARGET Thrust::CUB)
    hydra_thrust_debug("Setting CUB target to ${cub_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(HYDRA_THRUST_CUB_VERSION ${CUB_VERSION} CACHE INTERNAL
      "CUB version used by Thrust"
      FORCE
    )
    _hydra_thrust_declare_interface_alias(Thrust::CUB _Thrust_CUB)
    target_link_libraries(_Thrust_CUB INTERFACE ${cub_target})
    hydra_thrust_debug_target(${cub_target} "${HYDRA_THRUST_CUB_VERSION}" internal)
    hydra_thrust_debug_target(Thrust::CUB "CUB ${HYDRA_THRUST_CUB_VERSION}" internal)
  endif()
endfunction()

# Internal use only -- libcudacxx must be found during the initial
# `find_package(Thrust)` call and cannot be set afterwards. See README.md in
# this directory for details on using a specific libcudacxx target.
function(_hydra_thrust_set_libcudacxx_target libcudacxx_target)
  if (NOT TARGET Thrust::libcudacxx)
    hydra_thrust_debug("Setting libcudacxx target to ${libcudacxx_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(HYDRA_THRUST_libcudacxx_VERSION ${libcudacxx_VERSION} CACHE INTERNAL
      "libcudacxx version used by Thrust"
      FORCE
    )
    _hydra_thrust_declare_interface_alias(Thrust::libcudacxx _Thrust_libcudacxx)
    target_link_libraries(_Thrust_libcudacxx INTERFACE ${libcudacxx_target})
    hydra_thrust_debug_target(${libcudacxx_target} "${HYDRA_THRUST_libcudacxx_VERSION}" internal)
    hydra_thrust_debug_target(Thrust::libcudacxx "libcudacxx ${HYDRA_THRUST_libcudacxx_VERSION}" internal)
  endif()
endfunction()

# Use the provided tbb_target for the TBB backend. If Thrust::TBB already
# exists, this call has no effect.
function(hydra_thrust_set_TBB_target tbb_target)
  if (NOT TARGET Thrust::TBB)
    hydra_thrust_debug("Setting TBB target to ${tbb_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(HYDRA_THRUST_TBB_VERSION ${TBB_VERSION} CACHE INTERNAL
      "TBB version used by Thrust"
      FORCE
    )
    _hydra_thrust_declare_interface_alias(Thrust::TBB _Thrust_TBB)
    target_link_libraries(_Thrust_TBB INTERFACE Thrust::Thrust ${tbb_target})
    hydra_thrust_debug_target(${tbb_target} "${HYDRA_THRUST_TBB_VERSION}" internal)
    hydra_thrust_debug_target(Thrust::TBB "${HYDRA_THRUST_TBB_VERSION}" internal)
    _hydra_thrust_setup_system(TBB)
  endif()
endfunction()

# Use the provided omp_target for the OMP backend. If Thrust::OMP already
# exists, this call has no effect.
function(hydra_thrust_set_OMP_target omp_target)
  if (NOT TARGET Thrust::OMP)
    hydra_thrust_debug("Setting OMP target to ${omp_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(HYDRA_THRUST_OMP_VERSION ${OpenMP_CXX_VERSION} CACHE INTERNAL
      "OpenMP version used by Thrust"
      FORCE
    )
    _hydra_thrust_declare_interface_alias(Thrust::OMP _Thrust_OMP)
    target_link_libraries(_Thrust_OMP INTERFACE Thrust::Thrust ${omp_target})
    hydra_thrust_debug_target(${omp_target} "${HYDRA_THRUST_OMP_VERSION}" internal)
    hydra_thrust_debug_target(Thrust::OMP "${HYDRA_THRUST_OMP_VERSION}" internal)
    _hydra_thrust_setup_system(OMP)
  endif()
endfunction()

function(_hydra_thrust_find_CPP required)
  if (NOT TARGET Thrust::CPP)
    hydra_thrust_debug("Generating CPP targets." internal)
    _hydra_thrust_declare_interface_alias(Thrust::CPP _Thrust_CPP)
    target_link_libraries(_Thrust_CPP INTERFACE Thrust::Thrust)
    hydra_thrust_debug_target(Thrust::CPP "Thrust ${HYDRA_THRUST_VERSION}" internal)
    _hydra_thrust_setup_system(CPP)
  endif()
endfunction()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like CUB_VERSION, etc won't be in the caller's
# scope.
macro(_hydra_thrust_find_CUDA required)
  if (NOT TARGET Thrust::CUB)
    hydra_thrust_debug("Searching for CUB ${required}" internal)
    find_package(CUB ${HYDRA_THRUST_VERSION} CONFIG
      ${_HYDRA_THRUST_QUIET_FLAG}
      ${required}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${_HYDRA_THRUST_INCLUDE_DIR}/dependencies/cub" # Source layout (GitHub)
        "${_HYDRA_THRUST_INCLUDE_DIR}/../cub/cub/cmake" # Source layout (Perforce)
        "${_HYDRA_THRUST_CMAKE_DIR}/.."                 # Install layout
    )

    if (TARGET CUB::CUB)
      hydra_thrust_set_CUB_target(CUB::CUB)
    else()
      hydra_thrust_debug("CUB not found!" internal)
    endif()
  endif()

  if (NOT TARGET Thrust::CUDA)
    _hydra_thrust_declare_interface_alias(Thrust::CUDA _Thrust_CUDA)
    _hydra_thrust_setup_system(CUDA)
    target_link_libraries(_Thrust_CUDA INTERFACE
      Thrust::Thrust
      Thrust::CUB
    )
    hydra_thrust_debug_target(Thrust::CUDA "" internal)
  endif()
endmacro()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like TBB_VERSION, etc won't be in the caller's
# scope.
macro(_hydra_thrust_find_TBB required)
  if(NOT TARGET Thrust::TBB)
    hydra_thrust_debug("Searching for TBB ${required}" internal)
    # Swap in a temporary module path to make sure we use our FindTBB.cmake
    set(_HYDRA_THRUST_STASH_MODULE_PATH "${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "${_HYDRA_THRUST_CMAKE_DIR}")

    # Push policy CMP0074 to silence warnings about TBB_ROOT being set. This
    # var is used unconventionally in this FindTBB.cmake module.
    # Someday we'll have a suitable TBB cmake configuration and can avoid this.
    cmake_policy(PUSH)
    cmake_policy(SET CMP0074 OLD)
    set(HYDRA_THRUST_TBB_ROOT "" CACHE PATH "Path to the root of the TBB installation.")
    if (TBB_ROOT AND NOT HYDRA_THRUST_TBB_ROOT)
      message(
        "Warning: TBB_ROOT is set. "
        "Thrust uses HYDRA_THRUST_TBB_ROOT to avoid issues with CMake Policy CMP0074. "
        "Please set this variable instead when using Thrust with TBB."
      )
    endif()
    set(TBB_ROOT "${HYDRA_THRUST_TBB_ROOT}")
    set(_HYDRA_THRUST_STASH_TBB_ROOT "${TBB_ROOT}")

    find_package(TBB
      ${_HYDRA_THRUST_QUIET_FLAG}
      ${required}
    )

    cmake_policy(POP)
    set(TBB_ROOT "${_HYDRA_THRUST_STASH_TBB_ROOT}")
    set(CMAKE_MODULE_PATH "${_HYDRA_THRUST_STASH_MODULE_PATH}")

    if (TARGET TBB::tbb)
      hydra_thrust_set_TBB_target(TBB::tbb)
    else()
      hydra_thrust_debug("TBB not found!" internal)
    endif()
  endif()
endmacro()

# Wrap the OpenMP flags for CUDA targets
function(hydra_thrust_fixup_omp_target omp_target)
  get_target_property(opts ${omp_target} INTERFACE_COMPILE_OPTIONS)
  if (opts MATCHES "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^>]*)>")
    target_compile_options(${omp_target} INTERFACE
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=${CMAKE_MATCH_1}>
    )
  endif()
endfunction()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like OpenMP_CXX_VERSION, etc won't be in the caller's
# scope.
macro(_hydra_thrust_find_OMP required)
  if (NOT TARGET Thrust::OMP)
    hydra_thrust_debug("Searching for OMP ${required}" internal)
    find_package(OpenMP
      ${_HYDRA_THRUST_QUIET_FLAG}
      ${_HYDRA_THRUST_REQUIRED_FLAG_OMP}
      COMPONENTS CXX
    )

    if (TARGET OpenMP::OpenMP_CXX)
      hydra_thrust_fixup_omp_target(OpenMP::OpenMP_CXX)
      hydra_thrust_set_OMP_target(OpenMP::OpenMP_CXX)
    else()
      hydra_thrust_debug("OpenMP::OpenMP_CXX not found!" internal)
    endif()
  endif()
endmacro()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like CUB_VERSION, etc won't be in the caller's
# scope.
macro(_hydra_thrust_find_backend backend required)
  # Unfortunately, _hydra_thrust_find_${backend}(req) is not valid CMake syntax. Hence
  # why this function exists.
  if ("${backend}" STREQUAL "CPP")
    _hydra_thrust_find_CPP("${required}")
  elseif ("${backend}" STREQUAL "CUDA")
    _hydra_thrust_find_CUDA("${required}")
  elseif ("${backend}" STREQUAL "TBB")
    _hydra_thrust_find_TBB("${required}")
  elseif ("${backend}" STREQUAL "OMP")
    _hydra_thrust_find_OMP("${required}")
  else()
    message(FATAL_ERROR "_hydra_thrust_find_backend: Invalid system: ${backend}")
  endif()
endmacro()

################################################################################
# Initialization. Executed inside find_package(Thrust) call.
#

if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_HYDRA_THRUST_QUIET ON CACHE INTERNAL "Quiet mode enabled for Thrust find_package calls." FORCE)
  set(_HYDRA_THRUST_QUIET_FLAG "QUIET" CACHE INTERNAL "" FORCE)
else()
  unset(_HYDRA_THRUST_QUIET CACHE)
  unset(_HYDRA_THRUST_QUIET_FLAG CACHE)
endif()

set(_HYDRA_THRUST_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL
  "Location of hydra_thrust-config.cmake"
  FORCE
)

# Internal target that actually holds the Thrust interface. Used by all other Thrust targets.
if (NOT TARGET Thrust::Thrust)
  _hydra_thrust_declare_interface_alias(Thrust::Thrust _Thrust_Thrust)
  # Pull in the include dir detected by hydra_thrust-config-version.cmake
  set(_HYDRA_THRUST_INCLUDE_DIR "${_HYDRA_THRUST_VERSION_INCLUDE_DIR}"
    CACHE INTERNAL "Location of Thrust headers."
    FORCE
  )
  unset(_HYDRA_THRUST_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache
  target_include_directories(_Thrust_Thrust INTERFACE "${_HYDRA_THRUST_INCLUDE_DIR}")
  hydra_thrust_debug_target(Thrust::Thrust "${HYDRA_THRUST_VERSION}" internal)
endif()

# Find libcudacxx prior to locating backend-specific deps. This ensures that CUB
# finds the same package.
if (NOT TARGET Thrust::libcudacxx)
  hydra_thrust_debug("Searching for libcudacxx REQUIRED" internal)

  # First do a non-required search for any co-packaged versions.
  # These are preferred.
  find_package(libcudacxx ${hydra_thrust_libcudacxx_version} CONFIG
    ${_HYDRA_THRUST_QUIET_FLAG}
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS
      "${_HYDRA_THRUST_INCLUDE_DIR}/dependencies/libcudacxx" # Source layout (GitHub)
      "${_HYDRA_THRUST_INCLUDE_DIR}/../libcudacxx"           # Source layout (Perforce)
      "${_HYDRA_THRUST_CMAKE_DIR}/.."                        # Install layout
  )

  # A second required search allows externally packaged to be used and fails if
  # no suitable package exists.
  find_package(libcudacxx ${hydra_thrust_libcudacxx_version} CONFIG
    REQUIRED
    ${_HYDRA_THRUST_QUIET_FLAG}
  )

  if (TARGET libcudacxx::libcudacxx)
    _hydra_thrust_set_libcudacxx_target(libcudacxx::libcudacxx)
  else()
    hydra_thrust_debug("Expected libcudacxx::libcudacxx target not found!" internal)
  endif()

  target_link_libraries(_Thrust_Thrust INTERFACE Thrust::libcudacxx)
endif()

# Handle find_package COMPONENT requests:
foreach(component ${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})
  if (NOT component IN_LIST HYDRA_THRUST_HOST_SYSTEM_OPTIONS AND
      NOT component IN_LIST HYDRA_THRUST_DEVICE_SYSTEM_OPTIONS)
    message(FATAL_ERROR "Invalid component requested: '${component}'")
  endif()

  unset(req)
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
    set(req "REQUIRED")
  endif()

  hydra_thrust_debug("Preloading COMPONENT '${component}' ${req}" internal)
  _hydra_thrust_find_backend(${component} "${req}")
endforeach()

hydra_thrust_update_system_found_flags()

include(FindPackageHandleStandardArgs)
if (NOT Thrust_CONFIG)
  set(Thrust_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(Thrust CONFIG_MODE)
