/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * WignerD.h
 *
 *  Created on: 02/08/2018 - 
 *      Author: Davide Brundu -
 */

#ifndef WIGNERD_H_
#define WIGNERD_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <hydra/functions/Utils.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <ratio>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/functions/detail/wigner_d_funcion.h>




namespace hydra {

  template< template<size_t> class T, size_t J, size_t M, size_t N, unsigned int ArgIndex=0>
  class WignerD;
  
  
/*---------------------
 * Specialization for 
 * unit values of J,N,M
 *------------------- */
 
  template<size_t J, size_t M, size_t N, unsigned int ArgIndex>
  class WignerD<_unit, J, M, N, ArgIndex>: public BaseFunctor< WignerD<_unit, J, M, N, ArgIndex>, double, 0>
  {
    constexpr static int JPM  = detail::nearest_int<J+M,1>::value;
    constexpr static int JPN  = detail::nearest_int<J+N,1>::value;
    constexpr static int JMM  = detail::nearest_int<J-M,1>::value;
    constexpr static int JMN  = detail::nearest_int<J-N,1>::value;
    constexpr static int MPN  = detail::nearest_int<M+N,1>::value;

    static_assert((JPM <0 || JPN < 0 || JMM < 0 || JMN < 0 || J < 0 || J > 25 ) ,
                  "[Hydra::WignerD] : Wrong parameters combination");
                  
    public:
    
      WignerD()=default;
      
      __hydra_dual__
      WignerD( WignerD<_unit, J, M, N, ArgIndex> const& other):
            BaseFunctor<WignerD<_unit, J, M, N, ArgIndex>, double, 0>(other)
            {}
            
      __hydra_dual__
      WignerD<_unit, J, M, N, ArgIndex>& operator=( WignerD<_unit, J, M, N, ArgIndex> const& other){

            if(this == &other) return *this;
            BaseFunctor<WignerD<_unit, J, M, N, ArgIndex>, double, 0>::operator=(other);
            return *this;
      }
      
      
      template<typename T>
      __hydra_dual__ inline
      double Evaluate(unsigned int, T*x)  const 
      {
            double beta = x[ArgIndex] ;
            double r = wignerd(beta);
            return  CHECK_VALUE(r, "r=%f", r);
      }

      template<typename T>
      __hydra_dual__ inline
      double Evaluate(T x)  const 
      {
            double beta =  get<ArgIndex>(x);
            double r = wignerd(beta);
            return  CHECK_VALUE(r, "r=%f", r);
      }
      
    private:
    
      __hydra_dual__ inline
      double wignerd( double beta ) const 
      {
            double r = (beta < 0 || beta > 2.0*PI) ? printf("HYDRA WARNING: WignerD: Illegal argument  beta=%g\n", beta):
                  (beta == 0)  ? (JPM == JPN ) :
                  (beta == PI) ? (JPM == JMN ) - 2*(::abs(JPM)%2 == 1):
                  (beta == 2.0*PI) ? (JPM == JPN) - 2*(::abs(MPN)%2 == 1) : wdf(beta);
            return r;
      }
      
      __hydra_dual__
      inline double wdf( double  b) const 
      {
            r = wigner_d_function<_unit<J>,_unit<M>,_unit<N>>(b);
            return CHECK_VALUE(r, "r=%f", r);
      }
      
  };




/*---------------------
 * Specialization for 
 * half values of J,N,M
 *------------------- */
 
  template<size_t J, size_t M, size_t N, unsigned int ArgIndex>
  class WignerD<_half, J, M, N, ArgIndex>: public BaseFunctor< WignerD<_half, J, M, N, ArgIndex>, double, 0>
  {
    constexpr static int JPM  = detail::nearest_int<J+M,2>::value;
    constexpr static int JPN  = detail::nearest_int<J+N,2>::value;
    constexpr static int JMM  = detail::nearest_int<J-M,2>::value;
    constexpr static int JMN  = detail::nearest_int<J-N,2>::value;
    constexpr static int MPN  = detail::nearest_int<M+N,2>::value;

    static_assert((JPM <0 || JPN < 0 || JMM < 0 || JMN < 0 || J < 0 || J > 25 ) ,
                  "[Hydra::WignerD] : Wrong parameters combination");
                  
    public:
    
      WignerD()=default;
      
      __hydra_dual__
      WignerD( WignerD<_half, J, M, N, ArgIndex> const& other):
            BaseFunctor<WignerD<_half, J, M, N, ArgIndex>, double, 0>(other)
            {}
            
      __hydra_dual__
      WignerD<_half, J, M, N, ArgIndex>& operator=( WignerD<_half, J, M, N, ArgIndex> const& other){

            if(this == &other) return *this;
            BaseFunctor<WignerD<_half, J, M, N, ArgIndex>, double, 0>::operator=(other);
            return *this;
      }
      
      
      template<typename T>
      __hydra_dual__ inline
      double Evaluate(unsigned int, T*x)  const 
      {
            double beta = x[ArgIndex] ;
            double r = wignerd(beta);
            return  CHECK_VALUE(r, "r=%f", r);
      }

      template<typename T>
      __hydra_dual__ inline
      double Evaluate(T x)  const 
      {
            double beta =  get<ArgIndex>(x);
            double r = wignerd(beta);
            return  CHECK_VALUE(r, "r=%f", r);
      }
      
    private:
    
      __hydra_dual__ inline
      double wignerd( double beta ) const 
      {
            double r = (beta < 0 || beta > 2.0*PI) ? printf("HYDRA WARNING: WignerD: Illegal argument  beta=%g\n", beta):
                  (beta == 0)  ? (JPM == JPN ) :
                  (beta == PI) ? (JPM == JMN ) - 2*(::abs(JPM)%2 == 1):
                  (beta == 2.0*PI) ? (JPM == JPN) - 2*(::abs(MPN)%2 == 1) : wdf(beta);
            return r;
      }
      
      __hydra_dual__
      inline double wdf( double  b) const 
      {
            r = wigner_d_function<_half<J>,_half<M>,_half<N>>(b);
            return CHECK_VALUE(r, "r=%f", r);
      }
      
  };



/*-----------------------------------------
 * Generic function with open type T 
 * to recall WignerD with types deduction
 *--------------------------------------- */
 
  template<unsigned int ArgIndex=0, template<size_t> class T, size_t J, size_t M, size_t N>
  WignerD<T, J, M, N, ArgIndex> Make_WignerD( T<J> const& obj1 , T<M> const& obj2, T<N> const& obj3)
  {
    return WignerD<T, J, M, N, ArgIndex>{};
  }





}


#endif /* WIGNERD_H_ */
