/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * bessel.h
 *
 *  Created on: 28/08/2020
 *      Author: Antonio Augusto Alves Junior
 *
 *        Note: This implementation inspired by, but not reproducing,
 *        the contents of:
 *
 *        Numerical Recipes in C,
 *        The Art of Scientific Computing - Second Edition
 *        "Chapter 6.Special Functions"
 *        "6.6 Modified Bessel Functions of Integer Order"
 */

#ifndef BESSEL_H_
#define BESSEL_H_


#include <hydra/detail/Config.h>
#include <cmath>
#include <limits>

#define BESSEL_ACC 100.0
#define BESSEL_BIGNO 1.0e10
#define BESSEL_BIGNI 1.0e-10

namespace hydra {


__hydra_host__ __hydra_device__
inline double bessel_j0( const double x )
{
   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax = ::fabs(x)) < 8.0) {

      y=x*x;

      ans1 = 57568490574.0 + y*(-13362590354.0 + y*(651619640.7
         + y*(-11214424.18 + y*(77392.33017 + y*(-184.9052456)))));

      ans2 = 57568490411.0 + y*(1029532985.0 + y*(9494680.718
         + y*(59272.64853 + y*(267.8532712 + y*1.0))));

      ans=ans1/ans2;

   } else {

      z=8.0/ax;

      y=z*z;

      xx=ax-0.785398164;

      ans1 = 1.0 + y*(-0.1098628627e-2 + y*(0.2734510407e-4
         + y*(-0.2073370639e-5 + y*0.2093887211e-6)));

      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         -y*0.934935152e-7)));

      ans = ::sqrt(0.636619772/ax)*(::cos(xx)*ans1-z*::sin(xx)*ans2);
   }

   return ans;
}


__hydra_host__ __hydra_device__
inline double bessel_j1( const double x )
{
   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {

      y=x*x;

      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
         +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));

      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
         +y*(99447.43394+y*(376.9991397+y*1.0))));

      ans=ans1/ans2;

   } else {

	   z=8.0/ax;

	   y=z*z;

	   xx=ax-2.356194491;

	   ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));

	   ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));

	   ans=::sqrt(0.636619772/ax)*(::cos(xx)*ans1-z*::sin(xx)*ans2);

	   if (x < 0.0) ans = -ans;
   }
   return ans;
}

__hydra_host__ __hydra_device__
inline double bessel_j(  const int n, const double x )
{
   int    j, jsum, m;
   double ax, bj, bjm, bjp, sum, tox, ans;


   if (n < 0)
   {
      double  dblank = -std::numeric_limits<double>::infinity();
      return  dblank ;
   }
   ax=::fabs(x);
   if (n == 0)
      return( bessel_j0(ax) );
   if (n == 1)
      return( bessel_j1(ax) );


   if (ax == 0.0)
      return 0.0;
   else if (ax > (double) n) {
      tox=2.0/ax;
      bjm=bessel_j0(ax);
      bj=bessel_j1(ax);
      for (j=1;j<n;j++) {
         bjp=j*tox*bj-bjm;
         bjm=bj;
         bj=bjp;
      }
      ans=bj;
   } else {
      tox=2.0/ax;
      m=2*((n+(int) ::sqrt(BESSEL_ACC*n))/2);
      jsum=0;
      bjp=ans=sum=0.0;
      bj=1.0;
      for (j=m;j>0;j--) {
         bjm=j*tox*bj-bjp;
         bjp=bj;
         bj=bjm;
         if (::fabs(bj) > BESSEL_BIGNO) {
            bj *= BESSEL_BIGNI;
            bjp *= BESSEL_BIGNI;
            ans *= BESSEL_BIGNI;
            sum *= BESSEL_BIGNI;
         }
         if (jsum) sum += bj;
         jsum=!jsum;
         if (j == n) ans=bjp;
      }
      sum=2.0*sum-bj;
      ans /= sum;
   }
   return  x < 0.0 && n%2 == 1 ? -ans : ans;
}

__hydra_host__ __hydra_device__
inline  double bessel_y0(const double x )
{
   double z;
   double xx,y,ans,ans1,ans2;

   if (x < 8.0) {
      y=x*x;
      ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
         +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
      ans2=40076544269.0+y*(745249964.8+y*(7189466.438
         +y*(47447.26470+y*(226.1030244+y*1.0))));
      ans=(ans1/ans2)+0.636619772*bessel_j0(x)*::log(x);
   } else {
      z=8.0/x;
      y=z*z;
      xx=x-0.785398164;
      ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
         +y*(-0.2073370639e-5+y*0.2093887211e-6)));
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         +y*(-0.934945152e-7))));
      ans=::sqrt(0.636619772/x)*(::sin(xx)*ans1+z*::cos(xx)*ans2);
   }

   return ans;
}



__hydra_host__ __hydra_device__
inline  double bessel_y1( const double x )
{
   double z;
   double xx,y,ans,ans1,ans2;

   if (x < 8.0) {
      y=x*x;
      ans1=x*(-0.4900604943e13+y*(0.1275274390e13
         +y*(-0.5153438139e11+y*(0.7349264551e9
         +y*(-0.4237922726e7+y*0.8511937935e4)))));
      ans2=0.2499580570e14+y*(0.4244419664e12
         +y*(0.3733650367e10+y*(0.2245904002e8
         +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
      ans=(ans1/ans2)+0.636619772*(bessel_j1(x)*::log(x)-1.0/x);
   } else {
      z=8.0/x;
      y=z*z;
      xx=x-2.356194491;
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));
      ans=::sqrt(0.636619772/x)*(::sin(xx)*ans1+z*::cos(xx)*ans2);
   }
   return ans;
}



__hydra_host__ __hydra_device__
inline  double bessel_y( const int n, const double x )
{
   int j;
   double by,bym,byp,tox;


   if (n < 0 || x == 0.0)
   {
      double   dblank = -std::numeric_limits<double>::infinity();
      return dblank;
   }
   if (n == 0)
      return( bessel_y0(x) );
   if (n == 1)
      return( bessel_y1(x) );

   tox=2.0/x;
   by=bessel_y1(x);
   bym=bessel_y0(x);
   for (j=1;j<n;j++) {
      byp=j*tox*by-bym;
      bym=by;
      by=byp;
   }
   return by;
}


__hydra_host__ __hydra_device__
inline  double bessel_i0( const double x )
{
   double ax,ans;
   double y;


   if ((ax=::fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
         +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
   } else {
      y=3.75/ax;
      ans=(::exp(ax)/::sqrt(ax))*(0.39894228+y*(0.1328592e-1
         +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
         +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
         +y*0.392377e-2))))))));
   }
   return ans;
}




__hydra_host__ __hydra_device__
inline  double bessel_i1( const double x)
{
   double ax,ans;
   double y;


   if ((ax=::fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
         +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
   } else {
      y=3.75/ax;
      ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
         -y*0.420059e-2));
      ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
         +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
      ans *= (::exp(ax)/::sqrt(ax));
   }
   return x < 0.0 ? -ans : ans;
}

__hydra_host__ __hydra_device__
inline  double bessel_i( const int n, const double x)
{
   int j;
   double bi,bim,bip,tox,ans;


   if (n < 0)
   {
      double   dblank = -std::numeric_limits<double>::infinity();;
      return dblank ;
   }
   if (n == 0)
      return( bessel_i0(x) );

   if (n == 1)
      return( bessel_i1(x) );


   if (x == 0.0)
      return 0.0;
   else {
      tox=2.0/::fabs(x);
      bip=ans=0.0;
      bi=1.0;
      for (j=2*(n+(int) ::sqrt(BESSEL_ACC*n));j>0;j--) {
         bim=bip+j*tox*bi;
         bip=bi;
         bi=bim;
         if (::fabs(bi) > BESSEL_BIGNO) {
            ans *= BESSEL_BIGNI;
            bi  *= BESSEL_BIGNI;
            bip *= BESSEL_BIGNI;
         }
         if (j == n) ans=bip;
      }
      ans *= bessel_i0(x)/bi;
      return  x < 0.0 && n%2 == 1 ? -ans : ans;
   }
}

__hydra_host__ __hydra_device__
inline  double bessel_k0( const double x )
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-::log(x/2.0)*bessel_i0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(::exp(-x)/::sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;
}




__hydra_host__ __hydra_device__
inline  double bessel_k1( const double x )
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(::log(x/2.0)*bessel_i1(x))+(1.0/x)*(1.0+y*(0.15443144
         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
   } else {
      y=2.0/x;
      ans=(::exp(-x)/::sqrt(x))*(1.25331414+y*(0.23498619
         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
         +y*(0.325614e-2+y*(-0.68245e-3)))))));
   }
   return ans;
}

__hydra_host__ __hydra_device__
inline  double bessel_k( const int n, const double x )
{
   int j;
   double bk,bkm,bkp,tox;


   if (n < 0 || x == 0.0)
   {
	   double   dblank = -std::numeric_limits<double>::infinity();
      return dblank ;
   }
   if (n == 0)
      return( bessel_k0(x) );
   if (n == 1)
      return( bessel_k1(x) );

   tox=2.0/x;
   bkm=bessel_k0(x);
   bk=bessel_k1(x);
   for (j=1;j<n;j++) {
      bkp=bkm+j*tox*bk;
      bkm=bk;
      bk=bkp;
   }
   return bk;
}



}  // namespace hydra


#endif /* BESSEL_H_ */
