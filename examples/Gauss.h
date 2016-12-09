/*
 * Gauss.h
 *
 *  Created on: 26/08/2016
 *      Author: augalves
 */

#ifndef GAUSS_H_
#define GAUSS_H_

#include <hydra/Function.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <initializer_list>

using namespace hydra;

namespace examples{

struct Gauss:public BaseFunctor<Gauss,GReal_t, 2>
{

	Gauss(Parameter const& mean, Parameter const& sigma, GUInt_t position=0 ):
		BaseFunctor<Gauss,GReal_t,2>(),
		fPosition(position)
		{
		SetParameter(0, mean);
		SetParameter(1, sigma);

		}

	__host__ __device__
	inline Gauss(Gauss const& other):
	BaseFunctor<Gauss,GReal_t,2>(other),
	fPosition(other.fPosition)
	{
		SetParameter(0, other.GetParameter(0) );
		SetParameter(1, other.GetParameter(1) );
	}


	__host__ __device__
	inline Gauss& operator=( Gauss const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Gauss,GReal_t,2>::operator=(other);
		this->fPosition = other.fPosition;
		this->SetParameter(0, other.GetParameter(0) );
		this->SetParameter(1, other.GetParameter(1) );

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x, T* p=0)
	{
		return exp(-((x[fPosition] - _par[0] ) * (x[fPosition] - _par[0]))
				/ (2 * _par[1]*_par[1] ))/( _par[1]*sqrt(2 *PI));
	}

	GUInt_t  fPosition;

};

}
#endif /* GAUSS_H_ */
