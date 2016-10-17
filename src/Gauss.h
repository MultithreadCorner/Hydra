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
		fPosition(position),
		fM(mean),
		fS(sigma)
		{ RegistryParameters({&fM, &fS}); }

	__host__ __device__
	inline Gauss(Gauss const& other):
	BaseFunctor<Gauss,GReal_t,2>(other),
	fPosition(other.fPosition),
	fM(other.fM),
	fS(other.fS)
	{ RegistryParameters({&(this->fM), &(this->fS)}); }


	__host__ __device__
	inline Gauss& operator=( Gauss const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Gauss,GReal_t,2>::operator=(other);
		this->fM = other.fM;
		this->fS = other.fS;
		this->fPosition = other.fPosition;
		this->RegistryParameters({&(this->fM), &(this->fS)});

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x, T* p=0)
	{
		//printf(" Gauss : fM %f fS %f\n", fM(),  fS());
		return exp(-((x[fPosition] - fM) * (x[fPosition] - fM)) / (2 * fS * fS));
	}

	GUInt_t  fPosition;
	Parameter fM;
	Parameter fS;

};

}
#endif /* GAUSS_H_ */
