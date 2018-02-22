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
 *  Created on: 25/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNERD_H_
#define WIGNERD_H_


#include <hydra/detail/Config.h>
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
#include <hydra/detail/external/thrust/extrema.h>

namespace hydra {

/**
 * @class WignerD
 *
 *  Calculates the beta-term
 *                   \f[ d^j_{mn}(beta) \f]
 *  in the matrix element of the finite rotation operator
 *  (Wigner's D-function), according to formula 4.3.1(3) in
 *  D.A. Varshalovich, A.N. Moskalev, and V.K. Khersonskii,
 *  Quantum Theory of Angular Momentum, World Scientific,
 *  Singapore 1988. CERNLIB DDJMNB function.
 */

template<unsigned int ArgIndex=0>
class WignerD: public BaseFunctor<WignerD<ArgIndex>, double, 0>
{
public:

	WignerD()=delete;

	WignerD( double j, double m, double n ):
		fJ(j), fM(m), fN(n){}

	__host__ __device__
	WignerD( WignerD<ArgIndex> const& other):
		BaseFunctor<WignerD<ArgIndex>, double, 0>(other),
		fJ(other.GetJ()),
		fM(other.GetM()),
		fN(other.GetN())
		{}

	__host__ __device__
	WignerD<ArgIndex>& operator=( WignerD<ArgIndex> const& other){

		if(this == &other) return *this;
		BaseFunctor<WignerD<ArgIndex>, double, 0>::operator=(other);
		fJ = other.GetJ();
		fM = other.GetM();
		fN = other.GetN();
		return *this;
	}


	__host__ __device__ inline
	double GetJ() const {
		return fJ;
	}

	__host__ __device__ inline
	void SetJ(double j) {
		fJ = j;
	}

	__host__ __device__ inline
	double GetM() const {
		return fM;
	}

	__host__ __device__ inline
	void SetM(double m) {
		fM = m;
	}

	__host__ __device__ inline
	double GetN() const {
		return fN;
	}

	__host__ __device__ inline
	void SetN(double n) {
		fN = n;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int, T*x)  const	{

		double beta = x[ArgIndex] ;
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "r=%f", r);

	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const {

		double beta =  get<ArgIndex>(x);
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "r=%f", r);

	}

private:

	__host__ __device__ inline
	double wignerd(const double beta ) const {


		const int jpm = nint(fJ+fM);
		const int jpn = nint(fJ+fN);
		const int jmm = nint(fJ-fM);

		const int jmn = nint(fJ-fN);
		const int mpn = nint(fM+fN);

		bool condition = (jpm<0 || jpn < 0 || jmm < 0 || jmn < 0 || fJ < 0 || fJ > 25 || beta < 0 || beta > 2.0*PI);

		double r = condition ? printf("HYDRA WARNING: WignerD: Illegal argument(s) fJ=%g, fM=%g, fN=%g, beta=%g\n",fJ,fM,fN,beta):
		    (beta == 0)  ? (jpm == jpn) :
		    (beta == PI) ? (jpm == jmn) - 2*(::abs(jpm)%2 == 1):
	    	(beta == 2.0*PI) ? (jpm == jpn) - 2*(::abs(mpn)%2 == 1) : wdf(jpm, jpn, jmm, jmn,mpn, beta);

		return r;

	}

	__host__ __device__ inline
	double wdf(const int jpm,const int jpn,const int jmm, const int jmn,const int mpn,const double  b) const {

		using HYDRA_EXTERNAL_NS::thrust::max;
		using HYDRA_EXTERNAL_NS::thrust::min;

		double r = 0;
		double s  = ::log(::sin(b/2.0));
		double c  = ::log(::fabs(::cos(b/2.0)));
		double rt = 0.5*(fcl[jpm]+fcl[jmm]+fcl[jpn]+fcl[jmn]);
		int k0    = max(0,mpn);
		int kq    = (b > PI) ? k0+jpm+mpn: k0+jpm;

		double q  = 1 - 2*(kq%2 == 1);
		kq = k0+k0;
		double cx = kq-mpn;
		double sx = jpm+jpn-kq;

		for( int k=k0; k<= min(jpm,jpn); k++)
		{
			r  += q*::exp(rt-fcl[k]-fcl[jpm-k]-fcl[jpn-k]-fcl[k-mpn]+ cx*c+sx*s);
			cx += 2;
			sx -= 2;
			q   = -q;
		}

		return CHECK_VALUE(r, "r=%f", r);
	}

	static constexpr double fcl[51] = { 0.0 ,0.0,
            6.93147180559945309e-1 ,1.79175946922805500e00,
            3.17805383034794562e00 ,4.78749174278204599e00,
            6.57925121201010100e00 ,8.52516136106541430e00,
            1.06046029027452502e01 ,1.28018274800814696e01,
            1.51044125730755153e01 ,1.75023078458738858e01,
            1.99872144956618861e01 ,2.25521638531234229e01,
            2.51912211827386815e01 ,2.78992713838408916e01,
            3.06718601060806728e01 ,3.35050734501368889e01,
            3.63954452080330536e01 ,3.93398841871994940e01,
            4.23356164607534850e01 ,4.53801388984769080e01,
            4.84711813518352239e01 ,5.16066755677643736e01,
            5.47847293981123192e01 ,5.80036052229805199e01,
            6.12617017610020020e01 ,6.45575386270063311e01,
            6.78897431371815350e01 ,7.12570389671680090e01,
            7.46582363488301644e01 ,7.80922235533153106e01,
            8.15579594561150372e01 ,8.50544670175815174e01,
            8.85808275421976788e01 ,9.21361756036870925e01,
            9.57196945421432025e01 ,9.93306124547874269e01,
            1.02968198614513813e02 ,1.06631760260643459e02,
            1.10320639714757395e02 ,1.14034211781461703e02,
            1.17771881399745072e02 ,1.21533081515438634e02,
            1.25317271149356895e02 ,1.29123933639127215e02,
            1.32952575035616310e02 ,1.36802722637326368e02,
            1.40673923648234259e02 ,1.44565743946344886e02,
            1.48477766951773032e02 };

	double fJ;
	double fM;
	double fN;

};



}  // namespace hydra




#endif /* WIGNERD_H_ */
