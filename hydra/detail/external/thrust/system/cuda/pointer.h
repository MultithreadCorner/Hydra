/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ccudaliance with the License.
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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/detail/pointer.h>
#include <hydra/detail/external/thrust/detail/reference.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace cuda_cub
{

template <typename>
class pointer;

} // end cuda_cub
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END


// specialize HYDRA_EXTERNAL_NS::thrust::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

template <typename Element>
struct iterator_traits<HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<Element> >
{
private:
  typedef HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<Element> ptr;

public:
  typedef typename ptr::iterator_category iterator_category;
  typedef typename ptr::value_type        value_type;
  typedef typename ptr::difference_type   difference_type;
  typedef ptr                             pointer;
  typedef typename ptr::reference         reference;
};    // end iterator_traits

namespace cuda_cub {

// forward declaration of reference for pointer
template <typename Element>
class reference;

// XXX nvcc + msvc have trouble instantiating reference below
//     this is a workaround
template <typename Element>
struct reference_msvc_workaround
{
  typedef HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference<Element> type;
};    // end reference_msvc_workaround


/*! \p pointer stores a pointer to an object allocated in memory available to the cuda system.
 *  This type provides type safety when dispatching standard algorithms on ranges resident
 *  in cuda memory.
 *
 *  \p pointer has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.
 *
 *  \p pointer can be created with the function \p cuda::malloc, or by explicitly calling its constructor
 *  with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained by eiter its <tt>get</tt> member function
 *  or the \p raw_pointer_cast function.
 *
 *  \note \p pointer is not a "smart" pointer; it is the programmer's responsibility to deallocate memory
 *  pointed to by \p pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see cuda::malloc
 *  \see cuda::free
 *  \see raw_pointer_cast
 */
template <typename T>
class pointer
    : public HYDRA_EXTERNAL_NS::thrust::pointer<
          T,
          HYDRA_EXTERNAL_NS::thrust::cuda_cub::tag,
          HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference<T>,
          HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<T> >
{

private:
  typedef HYDRA_EXTERNAL_NS::thrust::pointer<
      T,
      HYDRA_EXTERNAL_NS::thrust::cuda_cub::tag,
      typename reference_msvc_workaround<T>::type,
      HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<T> >
      super_t;

public:
  /*! \p pointer's no-argument constructor initializes its encapsulated pointer to \c 0.
   */
  __hydra_host__ __hydra_device__
  pointer() : super_t() {}

  #if HYDRA_THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __hydra_host__ __hydra_device__
  pointer(decltype(nullptr)) : super_t(nullptr) {}
  #endif

  /*! This constructor allows construction of a <tt>pointer<const T></tt> from a <tt>T*</tt>.
   *
   *  \param ptr A raw pointer to copy from, presumed to point to a location in memory
   *         accessible by the \p cuda system.
   *  \tparam OtherT \p OtherT shall be convertible to \p T.
   */
  template <typename OtherT>
  __hydra_host__ __hydra_device__ explicit pointer(OtherT *ptr) : super_t(ptr)
  {
  }

  /*! This constructor allows construction from another pointer-like object with related type.
   *
   *  \param other The \p OtherPointer to copy.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer shall be convertible
   *          to \p HYDRA_EXTERNAL_NS::thrust::system::cuda::tag and its element type shall be convertible to \p T.
   */
  template <typename OtherPointer>
  __hydra_host__ __hydra_device__
  pointer(const OtherPointer &other,
          typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer>::type * = 0) : super_t(other)
  {
  }

  /*! This constructor allows construction from another pointer-like object with \p void type.
   *
   *  \param other The \p OtherPointer to copy.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer shall be convertible
   *          to \p HYDRA_EXTERNAL_NS::thrust::system::cuda::tag and its element type shall be \p void.
   */
  template <typename OtherPointer>
  __hydra_host__ __hydra_device__
  explicit
  pointer(const OtherPointer &other,
          typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if_void_pointer_is_system_convertible<
              OtherPointer,
              pointer>::type * = 0) : super_t(other)
  {
  }

  /*! Assignment operator allows assigning from another pointer-like object with related type.
   *
   *  \param other The other pointer-like object to assign from.
   *  \tparam OtherPointer The system tag associated with \p OtherPointer shall be convertible
   *          to \p HYDRA_EXTERNAL_NS::thrust::system::cuda::tag and its element type shall be convertible to \p T.
   */
  template <typename OtherPointer>
  __hydra_host__ __hydra_device__
      typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if_pointer_is_convertible<
          OtherPointer,
          pointer,
          pointer &>::type
      operator=(const OtherPointer &other)
  {
    return super_t::operator=(other);
  }

  #if HYDRA_THRUST_CPP_DIALECT >= 2011
  // NOTE: This is needed so that Thrust smart pointers can be used in
  // `std::unique_ptr`.
  __hydra_host__ __hydra_device__
  pointer& operator=(decltype(nullptr))
  {
    super_t::operator=(nullptr);
    return *this;
  }
  #endif
};    // struct pointer

/*! \p reference is a wrapped reference to an object stored in memory available to the \p cuda system.
 *  \p reference is the type of the result of dereferencing a \p cuda::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
class reference
    : public HYDRA_EXTERNAL_NS::thrust::reference<
          T,
          HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<T>,
          HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference<T> >
{

private:
  typedef HYDRA_EXTERNAL_NS::thrust::reference<
      T,
      HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer<T>,
      HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference<T> >
      super_t;

public:
  /*! \cond
   */

  typedef typename super_t::value_type value_type;
  typedef typename super_t::pointer    pointer;

  /*! \endcond
   */

  /*! This constructor initializes this \p reference to refer to an object
   *  pointed to by the given \p pointer. After this \p reference is constructed,
   *  it shall refer to the object pointed to by \p ptr.
   *
   *  \param ptr A \p pointer to copy from.
   */
  __hydra_host__ __hydra_device__ explicit reference(const pointer &ptr)
      : super_t(ptr)
  {
  }

  /*! This constructor accepts a const reference to another \p reference of related type.
   *  After this \p reference is constructed, it shall refer to the same object as \p other.
   *
   *  \param other A \p reference to copy from.
   *  \tparam OtherT The element type of the other \p reference.
   *
   *  \note This constructor is templated primarily to allow initialization of <tt>reference<const T></tt>
   *        from <tt>reference<T></tt>.
   */
  template <typename OtherT>
  __hydra_host__ __hydra_device__
  reference(const reference<OtherT> &other,
            typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if_convertible<
                typename reference<OtherT>::pointer,
                pointer>::type * = 0)
      : super_t(other)
  {
  }

  /*! Copy assignment operator copy assigns from another \p reference of related type.
   *
   *  \param other The other \p reference to assign from.
   *  \return <tt>*this</tt>
   *  \tparam OtherT The element type of the other \p reference.
   */
  template <typename OtherT>
  __hydra_host__ __hydra_device__
      reference &
      operator=(const reference<OtherT> &other);

  /*! Assignment operator assigns from a \p value_type.
   *
   *  \param x The \p value_type to assign from.
   *  \return <tt>*this</tt>
   */
  __hydra_host__ __hydra_device__
      reference &
      operator=(const value_type &x);
};    // struct reference

/*! Exchanges the values of two objects referred to by \p reference.
 *  \p x The first \p reference of interest.
 *  \p y The second \p reference of interest.
 */
template <typename T>
__hydra_host__ __hydra_device__ void swap(reference<T> x, reference<T> y);

} // end cuda_cub

namespace system {


/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace HYDRA_EXTERNAL_NS::thrust::system::cuda
 *  \brief \p HYDRA_EXTERNAL_NS::thrust::system::cuda is the namespace containing functionality for allocating, manipulating,
 *         and deallocating memory available to Thrust's CUDA backend system.
 *         The identifiers are provided in a separate namespace underneath <tt>HYDRA_EXTERNAL_NS::thrust::system</tt>
 *         for import convenience but are also aliased in the top-level <tt>HYDRA_EXTERNAL_NS::thrust::cuda</tt>
 *         namespace for easy access.
 *
 */

namespace cuda {
using HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer;
using HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference;
} // end cuda

/*! \}
 */

} // end system

/*! \HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace HYDRA_EXTERNAL_NS::thrust::cuda
 *  \brief \p HYDRA_EXTERNAL_NS::thrust::cuda is a top-level alias for \p HYDRA_EXTERNAL_NS::thrust::system::cuda. */
namespace cuda {
using HYDRA_EXTERNAL_NS::thrust::cuda_cub::pointer;
using HYDRA_EXTERNAL_NS::thrust::cuda_cub::reference;
} // end cuda

} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END


#include <hydra/detail/external/thrust/system/cuda/detail/pointer.inl>
