/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * WignerDMatrix.h
 *
 *  Created on: 23/10/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef WIGNERDMATRIX_H_
#define WIGNERDMATRIX_H_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Tuple.h>
#include <hydra/functions/Utils.h>
#include <hydra/functions/Math.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <ratio>
#include <hydra/detail/external/thrust/extrema.h>

namespace hydra {

/**
 *  \ingroup common_functions
 * \class WignerDMatrix
 *
 *  Calculates the beta-term
 *                   \f[ d^j_{mn}(beta) \f]
 *  in the matrix element of the finite rotation operator
 *  (Wigner's D-function), according to formula 4.3.4(eq. 13) in
 *  D.A. Varshalovich, A.N. Moskalev, and V.K. Khersonskii,
 *  Quantum Theory of Angular Momentum, World Scientific, Singapore 1988.
 */

template<unsigned int ArgIndex=0>
class WignerDMatrix: public BaseFunctor<WignerDMatrix<ArgIndex>, double, 0>
{

public:

	WignerDMatrix()=delete;

	WignerDMatrix(double j, double m, double n):
		fJ(j),
		fM(m),
		fN(n),
		fF(0),
		fMu(0),
		fNu(0),
		fXi(0),
		fS(0)
	{
		UpdateState();
	}

	__hydra_dual__
	WignerDMatrix( WignerDMatrix<ArgIndex> const& other):
	BaseFunctor<WignerDMatrix<ArgIndex>, double, 0>(other),
	fJ(other.GetJ()),
	fM(other.GetM()),
	fN(other.GetN()),
	fF(other.GetF()),
	fMu(other.GetMu()),
	fNu(other.GetNu()),
	fXi(other.GetXi()),
	fS(other.GetS())
	{}

	__hydra_dual__
	WignerDMatrix<ArgIndex>& operator=( WignerDMatrix<ArgIndex> const& other){

		if(this == &other) return *this;
		BaseFunctor<WignerDMatrix<ArgIndex>, double, 0>::operator=(other);
		fJ = other.GetJ();
		fM = other.GetM();
		fN = other.GetN();
		fF = other.GetF();
		fMu = other.GetMu();
		fNu = other.GetNu();
		fXi = other.GetXi();
		fS  = other.GetS();
		return *this;
	}

	inline void SetJ(double j) {
		fJ = j;
		UpdateState();
	}

	inline void SetM(double m) {
		fM = m;
		UpdateState();
	}

	inline void SetN(double n) {
		fN = n;
		UpdateState();
	}

	__hydra_dual__
	inline double GetJ() const {
		return fJ;
	}

	__hydra_dual__
	inline double GetM() const {
		return fM;
	}

	__hydra_dual__
	inline double GetN() const {
		return fN;
	}

	__hydra_dual__
	inline double GetF() const {
		return fF;
	}


	__hydra_dual__
	inline double GetMu() const {
		return fMu;
	}


	__hydra_dual__
	inline double GetNu() const {
		return fNu;
	}


	__hydra_dual__
	inline unsigned GetS() const {
		return fS;
	}


	__hydra_dual__
	inline int GetXi() const {
		return fXi;
	}

	template<typename T>
	__hydra_dual__ inline
	double Evaluate(unsigned int, T*x)  const	{

		double beta = x[ArgIndex] ;
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "beta=%f r=%f", beta, r);

	}

	template<typename T>
	__hydra_dual__ inline
	double Evaluate(T x)  const {

		double beta =  get<ArgIndex>(x);
		double r = wignerd(beta);

		return  CHECK_VALUE(r, "beta=%f r=%f", beta, r);

	}

private:


	inline void UpdateState()
	{

		if( fJ< 0.0 || (::fabs(fM) > fJ || ::fabs(fN) > fJ) ) {
			std::ostringstream stringStream;

			stringStream << "Ilegal parameter set j="<< fJ <<" m=" << fM << " n="<< fN;

			HYDRA_LOG(ERROR, stringStream.str().c_str()  )

			exit(-1);
		}

		fMu = ::fabs(rint(fM-fN));
		fNu = ::fabs(rint(fM+fN));
		fS	= rint(fJ-0.5*(fMu+fNu));
		fXi = fN>=fM ? 1: ::pow(-1,fN-fM);

		fF = fXi*::sqrt(::tgamma(fS+1)*::tgamma(fS+fMu+fNu+1)/(::tgamma(fS+fMu+1)*::tgamma(fS+fNu+1)));

	}

	__hydra_dual__
	inline double wignerd(double theta) const {

		return fF*::pow(::sin(theta*0.5),fMu)*::pow(::cos(theta*0.5),fNu)*jacobi(fMu, fNu, fS, ::cos(theta));

	}

	double fJ;
	double fM;
	double fN;
	double fF;
	double fMu;
	double fNu;
	int      fXi;
	unsigned fS;

};

}  // namespace hydra


#endif /* WIGNERDMATRIX_H_ */
