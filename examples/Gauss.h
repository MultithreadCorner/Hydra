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

using namespace hydra;

namespace examples{

template<size_t DIM=1>
struct Gauss: public BaseFunctor<Gauss<DIM>,GReal_t, DIM+DIM>
{

	Gauss(const Parameter   (&mean)[DIM], const Parameter (& sigma)[DIM], const GUInt_t (&position)[DIM] ):
		BaseFunctor<Gauss,GReal_t,DIM+DIM>(){

		for(size_t i=0; i<DIM; i+=2){
			fPosition[i]=position[i];
			this->SetParameter(i, mean[i] );
			this->SetParameter(i+1, sigma[i] );
		}

	}

	__host__ __device__
	inline Gauss(Gauss<DIM>const& other):
	BaseFunctor<Gauss,GReal_t,DIM+DIM>(other)
	{
		for(size_t i=0; i<DIM; i+=2){
			fPosition[i]= other.fPosition[i];
			this->SetParameter(i, other.GetParameter(i)  );
			this->SetParameter(i+1, other.GetParameter(i+1)  );
		}
	}


	__host__ __device__
	inline Gauss<DIM>& operator=( Gauss<DIM> const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Gauss<DIM>,GReal_t,DIM+DIM>::operator=(other);
		for(size_t i=0; i<DIM; i+=2){

			this->fPosition[i]= other.fPosition[i];
			this->SetParameter(i, other.GetParameter(i)  );
			this->SetParameter(i+1, other.GetParameter(i+1)  );
		}
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(size_t n, T* x)
	{
		GReal_t g=1.0;

		for(size_t i=0; i<DIM; i+=2)
		{
			GReal_t m2 = (x[fPosition[i]] - Fun._par[i] )*(x[fPosition] - this->_par[i] );
			GReal_t s2 = this->_par[i+1]*this->_par[i+1];
			g *= exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI));
		}
		return g;
	}

	GUInt_t  fPosition[DIM];

};

}
#endif /* GAUSS_H_ */
