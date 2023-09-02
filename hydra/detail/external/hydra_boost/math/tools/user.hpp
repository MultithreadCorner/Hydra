// Copyright John Maddock 2007.
// Copyright Paul A. Bristow 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_USER_HPP
#define HYDRA_BOOST_MATH_TOOLS_USER_HPP

#ifdef _MSC_VER
#pragma once
#endif

// This file can be modified by the user to change the default policies.
// See "Changing the Policy Defaults" in documentation.

// define this if the platform has no long double functions,
// or if the long double versions have only double precision:
//
// #define HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
//
// Performance tuning options:
//
// #define HYDRA_BOOST_MATH_POLY_METHOD 3
// #define HYDRA_BOOST_MATH_RATIONAL_METHOD 3
//
// The maximum order of polynomial that will be evaluated
// via an unrolled specialisation:
//
// #define HYDRA_BOOST_MATH_MAX_POLY_ORDER 17
//
// decide whether to store constants as integers or reals:
//
// #define HYDRA_BOOST_MATH_INT_TABLE_TYPE(RT, IT) IT

//
// Default policies follow:
//
// Domain errors:
//
// #define HYDRA_BOOST_MATH_DOMAIN_ERROR_POLICY throw_on_error
//
// Pole errors:
//
// #define HYDRA_BOOST_MATH_POLE_ERROR_POLICY throw_on_error
//
// Overflow Errors:
//
// #define HYDRA_BOOST_MATH_OVERFLOW_ERROR_POLICY throw_on_error
//
// Internal Evaluation Errors:
//
// #define HYDRA_BOOST_MATH_EVALUATION_ERROR_POLICY throw_on_error
//
// Underflow:
//
// #define HYDRA_BOOST_MATH_UNDERFLOW_ERROR_POLICY ignore_error
//
// Denorms:
//
// #define HYDRA_BOOST_MATH_DENORM_ERROR_POLICY ignore_error
//
// Max digits to use for internal calculations:
//
// #define HYDRA_BOOST_MATH_DIGITS10_POLICY 0
//
// Promote floats to doubles internally?
//
// #define HYDRA_BOOST_MATH_PROMOTE_FLOAT_POLICY true
//
// Promote doubles to long double internally:
//
// #define HYDRA_BOOST_MATH_PROMOTE_DOUBLE_POLICY true
//
// What do discrete quantiles return?
//
// #define HYDRA_BOOST_MATH_DISCRETE_QUANTILE_POLICY integer_round_outwards
//
// If a function is mathematically undefined
// (for example the Cauchy distribution has no mean),
// then do we stop the code from compiling?
//
// #define HYDRA_BOOST_MATH_ASSERT_UNDEFINED_POLICY true
//
// Maximum series iterations permitted:
//
// #define HYDRA_BOOST_MATH_MAX_SERIES_ITERATION_POLICY 1000000
//
// Maximum root finding steps permitted:
//
// define HYDRA_BOOST_MATH_MAX_ROOT_ITERATION_POLICY 200
//
// Enable use of __float128 in numeric constants:
//
// #define HYDRA_BOOST_MATH_USE_FLOAT128
//
// Disable use of __float128 in numeric_constants even if the compiler looks to support it:
//
// #define HYDRA_BOOST_MATH_DISABLE_FLOAT128

#endif // HYDRA_BOOST_MATH_TOOLS_USER_HPP


