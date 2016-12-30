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

template<size_t DIM>
struct Gauss: public BaseFunctor<Gauss<DIM>,GReal_t, DIM+DIM>
{
	using    BaseFunctor<Gauss<DIM>,GReal_t, DIM+DIM>::_par;

	Gauss(const Parameter  (&mean)[DIM], const Parameter (& sigma)[DIM],
			const GUInt_t (&position)[DIM], GBool_t auto_normalize=kTrue):
		BaseFunctor<Gauss,GReal_t,DIM+DIM>(),
		fAutoNormalize(auto_normalize)
		{

		for(size_t i=0; i<DIM; i++){
			fPosition[i]=position[i];
			this->SetParameter(2*i, mean[i] );
			this->SetParameter(2*i+1, sigma[i] );
		}
	}

	__host__ __device__
	inline Gauss(Gauss<DIM>const& other):
	BaseFunctor<Gauss,GReal_t,DIM+DIM>(other)
	{
		for(size_t i=0; i<DIM; i++){
			fAutoNormalize = other.fAutoNormalize;
			fPosition[i] = other.fPosition[i];
			this->SetParameter(2*i, other.GetParameter(2*i)  );
			this->SetParameter(2*i+1, other.GetParameter(2*i+1)  );
		}
	}


	__host__ __device__
	inline Gauss<DIM>& operator=( Gauss<DIM> const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Gauss<DIM>,GReal_t,DIM+DIM>::operator=(other);
		for(size_t i=0; i<DIM; i++){

			this->fAutoNormalize = other.fAutoNormalize;
			this->fPosition[i]= other.fPosition[i];
			this->SetParameter(2*i, other.GetParameter(2*i)  );
			this->SetParameter(2*i+1, other.GetParameter(2*i+1)  );
		}
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(size_t n, T* x)
	{
		GReal_t g=1.0;

		for(size_t i=0; i<DIM; i++)
		{
			GReal_t m2 = (x[fPosition[i]] - _par[2*i] )*(x[fPosition[i]] - _par[2*i] );
			GReal_t s2 = _par[2*i+1]*_par[2*i+1];
			g *= fAutoNormalize? exp(-m2/(2.0 * s2 ))/( sqrt(2.0*s2*PI)): exp(-m2/(2.0 * s2 )) ;
		}
		return g;
	}

	GBool_t IsAutoNormalized() const {
		return fAutoNormalize;
	}

	void AutoNormalize(GBool_t autoNormalize=kTrue) {
		fAutoNormalize = autoNormalize;
	}

	GUInt_t  fPosition[DIM];
	GBool_t  fAutoNormalize;

};

template<size_t DIM>
struct GaussAnalyticIntegral
{

	GaussAnalyticIntegral(std::array<GReal_t,DIM> const& lower_lim,
			std::array<GReal_t,DIM> const& upper_lim)
	{
		for(size_t i=0; i<DIM; i++)
		{
			LowerLim[i]=lower_lim[i];
			UpperLim[i]=upper_lim[i];
		}
	}


	__host__ __device__
	inline GaussAnalyticIntegral(GaussAnalyticIntegral<DIM>const& other)
	{
		for(size_t i=0; i<DIM; i++)
		{
			LowerLim[i]=other.LowerLim[i];
			UpperLim[i]=other.UpperLim[i];
		}
	}


	__host__ __device__
	inline GaussAnalyticIntegral<DIM>& operator=( GaussAnalyticIntegral<DIM> const& other)
	{
		if(this == &other) return *this;

		for(size_t i=0; i<DIM; i++)
		{
			this->LowerLim[i]=other.LowerLim[i];
			this->UpperLim[i]=other.UpperLim[i];
		}
		return *this;
	}



	template<typename FUNCTOR>
	__host__ __device__
	inline std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor)
	{
		GReal_t g=1.0;
		GBool_t flag = functor.IsAutoNormalized();

		for(size_t i=0; i<DIM; i++)
		{
			GReal_t fraction = cumulative(functor[2*i], functor[2*i+1], UpperLim[i])
					- cumulative(functor[2*i], functor[2*i+1], LowerLim[i]);

			GReal_t scale = flag?1.0: functor[2*i+1]*sqrt(2.0*PI);

			g*= fraction*scale;
		}
		return std::make_pair(g,0.0);
	}

	GReal_t LowerLim[DIM];
	GReal_t UpperLim[DIM];

private:

	__host__ __device__
	inline GReal_t cumulative(const GReal_t mean, const GReal_t sigma, const GReal_t x)
	{
		return 0.5*(1.0 + erf( (x-mean)/( sigma*sqrt(2) ) ) );
	}

};


}
#endif /* GAUSS_H_ */
