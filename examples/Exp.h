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
		BaseFunctor<Exp,GReal_t,1>({tau}),
		fPosition(position)
		{
			//SetParameter(0, tau);
		}

	__host__ __device__
	inline Exp(Exp const& other):
	BaseFunctor<Exp,GReal_t,1>(other),
	fPosition(other.fPosition)
	{
	//	SetParameter(0, other.GetParameter(0) );
	}


	__host__ __device__
	inline Exp& operator=( Exp const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Exp,GReal_t,1>::operator=(other);
		this->fPosition = other.fPosition;
		//this->SetParameter(0, other.GetParameter(0) );

		return *this;

	}


	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x)
	{

		//printf(" Exp : fM %f fS %f\n", fM(),  fS());
		GReal_t tau = _par[0];


		return exp(x[fPosition]*tau );

	}

	GUInt_t  fPosition;

};


struct ExpAnalyticIntegral:public Integrator<ExpAnalyticIntegral>
{
	typedef void hydra_integrator_tag;

	ExpAnalyticIntegral(GReal_t const& lower_lim , GReal_t const& upper_lim):
	fLowerLim(lower_lim),
	fUpperLim(upper_lim)
	{}


	inline ExpAnalyticIntegral(ExpAnalyticIntegral const& other):
	fLowerLim(other.fLowerLim),
	fUpperLim(other.fUpperLim)
	{}


	inline ExpAnalyticIntegral& operator=( ExpAnalyticIntegral const& other)
	{
		if(this == &other) return *this;


		this->fLowerLim = other.fLowerLim;
		this->fUpperLim = other.fUpperLim;

		return *this;
	}



	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor)
	{
		GReal_t tau = functor[0];
		GReal_t r   =  (exp(fUpperLim*tau) - exp(fLowerLim*tau))/tau ;
		return std::make_pair(r,0.0);
	}

	GReal_t fLowerLim;
	GReal_t fUpperLim;

};


}

#endif /* EXP_H_ */
