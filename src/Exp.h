/*
 * Exp.h
 *
 *  Created on: 05/09/2016
 *      Author: augalves
 */

#ifndef EXP_H_
#define EXP_H_

#include <hydra/Function.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>

using namespace hydra;

namespace examples {
struct Exp:public BaseFunctor<Exp, GReal_t, 1>
{

	Exp(Parameter const& tau, GUInt_t position=0 ):
		BaseFunctor<Exp,GReal_t,1>(),
		fPosition(position),
		fTau(tau)
		{ RegistryParameters({&fTau}); }

	__host__ __device__
	inline Exp(Exp const& other):
	BaseFunctor<Exp,GReal_t,1>(other),
	fPosition(other.fPosition),
	fTau(other.fTau)
	{ RegistryParameters({&(this->fTau)}); }


	__host__ __device__
	inline Exp& operator=( Exp const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Exp,GReal_t,1>::operator=(other);
		this->fTau = other.fTau;
		this->fPosition = other.fPosition;
		this->RegistryParameters({&(this->fTau)});

		return *this;

	}


	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x, T* p=0)
	{
		//printf(" Exp : fM %f fS %f\n", fM(),  fS());



		return exp(x[fPosition]*fTau);

	}

	GUInt_t  fPosition;
	Parameter fTau;
};

struct InvExp
{

	InvExp(GReal_t const& tau, GReal_t const& x0, GUInt_t position=0 ):
		fPosition(position),
		fTau(tau),
		fX0(x0)
		{ }

	__host__ __device__
	inline InvExp(InvExp const& other):
	fPosition(other.fPosition),
	fTau(other.fTau),
	fX0(other.fX0)
	{ }


	__host__ __device__
	inline InvExp& operator=( InvExp const& other)
	{
		if(this == &other) return *this;
		this->fTau = other.fTau;
		this->fPosition = other.fPosition;
		this->fX0=other.fX0;

		return *this;

	}

	__host__ __device__
	inline GReal_t operator()(GReal_t x )
	{
		//printf(" Exp : fM %f fS %f\n", fM(),  fS());



		return x-exp(fTau*fX0)/fTau;


	}

	GUInt_t  fPosition;
	GReal_t fTau;
	GReal_t fX0;



};
}

#endif /* EXP_H_ */
