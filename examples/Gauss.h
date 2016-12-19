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

template<size_t DIM=1>
struct Gauss:public BaseFunctor<Gauss,GReal_t, 2*DIM>
{


	Gauss(const Parameter (&mean)[DIM], 	const Parameter (& sigma)[DIM],	const GUInt_t (&position)[DIM] ):
		BaseFunctor<Gauss,GReal_t,2*DIM>()
		{

		for(size_t i=0; i<DIM; i+=2){

			fPosition[i]=position[i];
			SetParameter(i, mean[i] );
			SetParameter(i+1, sigma[i] );
		}

		}

	__host__ __device__
	inline Gauss(Gauss<DIM>const& other):
	BaseFunctor<Gauss,GReal_t,2*DIM>(other)
	{
		SetParameter(0, other.GetParameter(0) );
		SetParameter(1, other.GetParameter(1) );

		for(size_t i=0; i<DIM; i+=2){

					fPosition[i]= other.fPosition[i];
					SetParameter(i, other.GetParameter(i)  );
					SetParameter(i+1, other.GetParameter(i+1)  );
				}
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
		GReal_t m2 = (x[fPosition] - _par[0] )*(x[fPosition] - _par[0] );
		GReal_t s2 = _par[1]*_par[1];
		GReal_t g=exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		return g;
	}

	GUInt_t  fPosition[DIM];

};

}
#endif /* GAUSS_H_ */
