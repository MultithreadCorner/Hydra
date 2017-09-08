/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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

#include <hydra/detail/external/thrust/complex.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

/* --- Constructors --- */

template <typename T>
inline __host__ __device__  complex<T>
::complex(const T & re, const T& im)
{
  real(re);
  imag(im);
} 

template <typename T>
template <typename X> 
inline __host__ __device__ complex<T>
::complex(const complex<X> & z)
{
  // The explicit T() is there no prevent Visual Studio from complaining
  // about potential loss of precision
  real(T(z.real()));
  imag(T(z.imag()));
}  

template <typename T>
template <typename X> 
inline __host__ complex<T>
::complex(const std::complex<X> & z)
{
  // The explicit T() is there no prevent Visual Studio from complaining
  // about potential loss of precision
  real(T(z.real()));
  imag(T(z.imag()));
}  



/* --- Compound Assignment Operators --- */

template <typename T>
__host__ __device__  inline 
complex<T>& complex<T>::operator+=(const complex<T> z)
{
  real(real()+z.real());
  imag(imag()+z.imag());
  return *this;
}

template <typename T>
__host__ __device__
inline complex<T>& complex<T>::operator-=(const complex<T> z)
{
  real(real()-z.real());
  imag(imag()-z.imag());
  return *this;
}

template <typename T>
__host__ __device__
inline complex<T>& complex<T>::operator*=(const complex<T> z)
{
  *this = *this * z;
  return *this;
}

template <typename T>
__host__ __device__
inline complex<T>& complex<T>::operator/=(const complex<T> z)
{
  *this = *this / z;
  return *this;
}



/* --- Equality Operators --- */

template <typename T> 
  __host__ __device__
  inline bool operator==(const complex<T>& lhs, const complex<T>& rhs){
  if(lhs.real() == rhs.real() && lhs.imag() == rhs.imag()){
    return true;
  }
  return false;
}

template <typename T> 
  __host__ __device__
  inline bool operator==(const T & lhs, const complex<T>& rhs){
  if(lhs == rhs.real() && rhs.imag() == 0){
    return true;
  }
  return false;
}

template <typename T> 
  __host__ __device__
  inline bool operator==(const complex<T> & lhs, const T& rhs){
  if(lhs.real() == rhs && lhs.imag() == 0){
    return true;
  }
  return false;
}

template <typename T> 
  __host__ __device__
  inline bool operator!=(const complex<T>& lhs, const complex<T>& rhs){
  return !(lhs == rhs);
}

template <typename T> 
  __host__ __device__
  inline bool operator!=(const T & lhs, const complex<T>& rhs){
  return !(lhs == rhs);
}

template <typename T> 
  __host__ __device__
  inline bool operator!=(const complex<T> & lhs, const T& rhs){
  return !(lhs == rhs);
}

} 

HYDRA_EXTERNAL_NAMESPACE_END

#include <hydra/detail/external/thrust/detail/complex/arithmetic.h>
#include <hydra/detail/external/thrust/detail/complex/cproj.h>
#include <hydra/detail/external/thrust/detail/complex/cexp.h>
#include <hydra/detail/external/thrust/detail/complex/cexpf.h>
#include <hydra/detail/external/thrust/detail/complex/clog.h>
#include <hydra/detail/external/thrust/detail/complex/clogf.h>
#include <hydra/detail/external/thrust/detail/complex/cpow.h>
#include <hydra/detail/external/thrust/detail/complex/cpowf.h>
#include <hydra/detail/external/thrust/detail/complex/ccosh.h>
#include <hydra/detail/external/thrust/detail/complex/ccoshf.h>
#include <hydra/detail/external/thrust/detail/complex/csinh.h>
#include <hydra/detail/external/thrust/detail/complex/csinhf.h>
#include <hydra/detail/external/thrust/detail/complex/ctanh.h>
#include <hydra/detail/external/thrust/detail/complex/ctanhf.h>
#include <hydra/detail/external/thrust/detail/complex/csqrt.h>
#include <hydra/detail/external/thrust/detail/complex/csqrtf.h>
#include <hydra/detail/external/thrust/detail/complex/catrig.h>
#include <hydra/detail/external/thrust/detail/complex/catrigf.h>
#include <hydra/detail/external/thrust/detail/complex/stream.h>

