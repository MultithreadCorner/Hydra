
/*! \file hydra_thrust/zip_function.h
 *  \brief Adaptor type that turns an N-ary function object into one that takes
 *         a tuple of size N so it can easily be used with algorithms taking zip
 *         iterators
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>
#include <hydra/detail/external/hydra_thrust/detail/modern_gcc_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011 && !defined(HYDRA_THRUST_LEGACY_GCC)

#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/type_traits/integer_sequence.h>
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>

HYDRA_THRUST_NAMESPACE_BEGIN

/*! \addtogroup function_objects Function Objects
 *  \{
 */

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

namespace detail {
namespace zip_detail {

// Add workaround for decltype(auto) on C++11-only compilers:
#if HYDRA_THRUST_CPP_DIALECT >= 2014

__hydra_thrust_exec_check_disable__
template <typename Function, typename Tuple, std::size_t... Is>
__host__ __device__
decltype(auto) apply_impl(Function&& func, Tuple&& args, index_sequence<Is...>)
{
  return func(hydra_thrust::get<Is>(HYDRA_THRUST_FWD(args))...);
}

template <typename Function, typename Tuple>
__host__ __device__
decltype(auto) apply(Function&& func, Tuple&& args)
{
  constexpr auto tuple_size = hydra_thrust::tuple_size<typename std::decay<Tuple>::type>::value;
  return apply_impl(HYDRA_THRUST_FWD(func), HYDRA_THRUST_FWD(args), make_index_sequence<tuple_size>{});
}

#else // HYDRA_THRUST_CPP_DIALECT

__hydra_thrust_exec_check_disable__
template <typename Function, typename Tuple, std::size_t... Is>
__host__ __device__
auto apply_impl(Function&& func, Tuple&& args, index_sequence<Is...>)
HYDRA_THRUST_DECLTYPE_RETURNS(func(hydra_thrust::get<Is>(HYDRA_THRUST_FWD(args))...))

template <typename Function, typename Tuple>
__host__ __device__
auto apply(Function&& func, Tuple&& args)
HYDRA_THRUST_DECLTYPE_RETURNS(
    apply_impl(
      HYDRA_THRUST_FWD(func),
      HYDRA_THRUST_FWD(args),
      make_index_sequence<
        hydra_thrust::tuple_size<typename std::decay<Tuple>::type>::value>{})
)

#endif // HYDRA_THRUST_CPP_DIALECT

} // namespace zip_detail
} // namespace detail

/*! \p zip_function is a function object that allows the easy use of N-ary
 *  function objects with \p zip_iterators without redefining them to take a
 *  \p tuple instead of N arguments.
 *
 *  This means that if a functor that takes 2 arguments which could be used with
 *  the \p transform function and \p device_iterators can be extended to take 3
 *  arguments and \p zip_iterators without rewriting the functor in terms of
 *  \p tuple.
 *
 *  The \p make_zip_function convenience function is provided to avoid having
 *  to explicitely define the type of the functor when creating a \p zip_function,
 *  whic is especially helpful when using lambdas as the functor.
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
 *  #include <hydra/detail/external/hydra_thrust/device_vector.h>
 *  #include <hydra/detail/external/hydra_thrust/transform.h>
 *  #include <hydra/detail/external/hydra_thrust/zip_function.h>
 *
 *  struct SumTuple {
 *    float operator()(Tuple tup) {
 *      return std::get<0>(tup) + std::get<1>(tup) + std::get<2>(tup);
 *    }
 *  };
 *  struct SumArgs {
 *    float operator()(float a, float b, float c) {
 *      return a + b + c;
 *    }
 *  };
 *
 *  int main() {
 *    hydra_thrust::device_vector<float> A(3);
 *    hydra_thrust::device_vector<float> B(3);
 *    hydra_thrust::device_vector<float> C(3);
 *    hydra_thrust::device_vector<float> D(3);
 *    A[0] = 0.f; A[1] = 1.f; A[2] = 2.f;
 *    B[0] = 1.f; B[1] = 2.f; B[2] = 3.f;
 *    C[0] = 2.f; C[1] = 3.f; C[2] = 4.f;
 *
 *    // The following four invocations of transform are equivalent
 *    // Transform with 3-tuple
 *    hydra_thrust::transform(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.begin(), B.begin(), C.begin())),
 *                      hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.end(), B.end(), C.end())),
 *                      D.begin(),
 *                      SumTuple{});
 *
 *    // Transform with 3 parameters
 *    hydra_thrust::zip_function<SumArgs> adapted{};
 *    hydra_thrust::transform(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.begin(), B.begin(), C.begin())),
 *                      hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.end(), B.end(), C.end())),
 *                      D.begin(),
 *                      adapted);
 *
 *    // Transform with 3 parameters with convenience function
 *    hydra_thrust::zip_function<SumArgs> adapted{};
 *    hydra_thrust::transform(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.begin(), B.begin(), C.begin())),
 *                      hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.end(), B.end(), C.end())),
 *                      D.begin(),
 *                      hydra_thrust::make_zip_function(SumArgs{}));
 *
 *    // Transform with 3 parameters with convenience function and lambda
 *    hydra_thrust::zip_function<SumArgs> adapted{};
 *    hydra_thrust::transform(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.begin(), B.begin(), C.begin())),
 *                      hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(A.end(), B.end(), C.end())),
 *                      D.begin(),
 *                      hydra_thrust::make_zip_function([] (float a, float b, float c) {
 *                                                  return a + b + c;
 *                                                }));
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see make_zip_function
 *  \see zip_iterator
 */
template <typename Function>
class zip_function
{
  public:
     __host__ __device__
    zip_function(Function func) : func(std::move(func)) {}

// Add workaround for decltype(auto) on C++11-only compilers:
#if HYDRA_THRUST_CPP_DIALECT >= 2014

    template <typename Tuple>
    __host__ __device__
    decltype(auto) operator()(Tuple&& args) const
    {
        return detail::zip_detail::apply(func, HYDRA_THRUST_FWD(args));
    }

#else // HYDRA_THRUST_CPP_DIALECT

    // Can't just use HYDRA_THRUST_DECLTYPE_RETURNS here since we need to use
    // std::declval for the signature components:
    template <typename Tuple>
    __host__ __device__
    auto operator()(Tuple&& args) const
    noexcept(noexcept(detail::zip_detail::apply(std::declval<Function>(), HYDRA_THRUST_FWD(args))))
    HYDRA_THRUST_TRAILING_RETURN(decltype(detail::zip_detail::apply(std::declval<Function>(), HYDRA_THRUST_FWD(args))))
    {
        return detail::zip_detail::apply(func, HYDRA_THRUST_FWD(args));
    }

#endif // HYDRA_THRUST_CPP_DIALECT

  private:
    mutable Function func;
};

/*! \p make_zip_function creates a \p zip_function from a function object.
 *
 *  \param fun The N-ary function object.
 *  \return A \p zip_function that takes a N-tuple.
 *
 *  \see zip_function
 */
template <typename Function>
__host__ __device__
zip_function<typename std::decay<Function>::type>
make_zip_function(Function&& fun)
{
    using func_t = typename std::decay<Function>::type;
    return zip_function<func_t>(HYDRA_THRUST_FWD(fun));
}

/*! \} // end function_object_adaptors
 */

/*! \} // end function_objects
 */

HYDRA_THRUST_NAMESPACE_END

#endif
