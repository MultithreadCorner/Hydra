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



struct Gauss:public BaseFunctor<Gauss,GReal_t, 2>
{

	Gauss(Parameter const& mean
		, Parameter const& sigma
		, GUInt_t position=0
		, GBool_t auto_normalize=kTrue ):
		BaseFunctor<Gauss,GReal_t,2>({mean, sigma}),
		fPosition(position),
		fAutoNormalize(auto_normalize)
		{

			//this->SetParameter(0, mean );
			//this->SetParameter(1, sigma );
		}

	__host__ __device__
	inline Gauss(Gauss const& other):
	BaseFunctor<Gauss,GReal_t,2>(other),
	fPosition(other.fPosition),
	fAutoNormalize(other.fAutoNormalize)
	{
		//this->SetParameter(0, other.GetParameter(0));
	    //this->SetParameter(1, other.GetParameter(1));
	}


	__host__ __device__
	inline Gauss& operator=( Gauss const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<Gauss,GReal_t,2>::operator=(other);

		this->fAutoNormalize = other.fAutoNormalize;
		this->fPosition = other.fPosition;
		//this->SetParameter(0, other.GetParameter(0));
		//this->SetParameter(1, other.GetParameter(1));

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x)
	{
		GReal_t delta = x[fPosition] - _par[0];
		GReal_t sigma = _par[1];


		return fAutoNormalize?
				exp( -(delta * delta) / (2 * sigma * sigma))/( sigma *sqrt(2.0*PI)):
				exp( -(delta * delta) / (2 * sigma * sigma));

	}

	__host__ __device__
	GBool_t IsAutoNormalized() const {
		return fAutoNormalize;
	}
	__host__ __device__
	void AutoNormalize(GBool_t autoNormalize=kTrue) {
		fAutoNormalize = autoNormalize;
	}

	__host__ __device__
	GUInt_t GetPosition() const {
		return fPosition;
	}

	__host__ __device__
	void SetPosition(GUInt_t position) {
		fPosition = position;
	}

private:
	GUInt_t  fPosition;
	GBool_t  fAutoNormalize;
};


struct GaussAnalyticIntegral: public Integrator<GaussAnalyticIntegral>
{
	typedef void hydra_integrator_tag;

	GaussAnalyticIntegral(GReal_t const& lower_lim, GReal_t const& upper_lim):
		fLowerLim(lower_lim),
		fUpperLim(upper_lim)
	{ }


	inline GaussAnalyticIntegral(GaussAnalyticIntegral const& other):
		fLowerLim(other.GetLowerLim()),
		fUpperLim(other.GetUpperLim())
	{}


	inline GaussAnalyticIntegral& operator=( GaussAnalyticIntegral const& other)
	{
		if(this == &other) return *this;

		this->fLowerLim = other.GetLowerLim();
		this->fUpperLim = other.GetUpperLim();

		return *this;
	}

	GReal_t GetLowerLim() const {
		return fLowerLim;
	}

	void SetLowerLim(GReal_t lowerLim) {
		fLowerLim = lowerLim;
	}

	GReal_t GetUpperLim() const {
		return fUpperLim;
	}

	void SetUpperLim(GReal_t upperLim) {
		fUpperLim = upperLim;
	}

	template<typename FUNCTOR>
	inline std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor)
	{

		GBool_t flag = functor.IsAutoNormalized();

		GReal_t fraction = cumulative(functor[0], functor[1], fUpperLim)
						 - cumulative(functor[0], functor[1], fLowerLim);

		GReal_t scale = flag?1.0: functor[1]*sqrt(2.0*PI);

		GReal_t g = fraction*scale;

		return std::make_pair(g,0.0);
	}


private:

	inline GReal_t cumulative(const GReal_t mean, const GReal_t sigma, const GReal_t x)
	{
		return 0.5*(1.0 + erf( (x-mean)/( sigma*sqrt(2) ) ) );
	}

	GReal_t fLowerLim;
	GReal_t fUpperLim;


};


/***************************************************************************
 * GaussN
 ***************************************************************************/
template<size_t DIM>
struct GaussN: public BaseFunctor<GaussN<DIM>,GReal_t, DIM+DIM>
{
	using    BaseFunctor<GaussN<DIM>,GReal_t, DIM+DIM>::_par;


	GaussN(const Parameter  (&mean)[DIM], const Parameter (& sigma)[DIM],
			const GUInt_t (&position)[DIM], GBool_t auto_normalize=kTrue):
		BaseFunctor<GaussN,GReal_t,DIM+DIM>(),
		fAutoNormalize(auto_normalize)
		{

		for(size_t i=0; i<DIM; i++){
			fPosition[i]=position[i];
			this->SetParameter(2*i, mean[i] );
			this->SetParameter(2*i+1, sigma[i] );
		}
	}

	__host__ __device__
	inline GaussN(GaussN<DIM>const& other):
	BaseFunctor<GaussN,GReal_t,DIM+DIM>(other)
	{
		fAutoNormalize = other.fAutoNormalize;
		for(size_t i=0; i<DIM; i++){
			fPosition[i] = other.fPosition[i];
			this->SetParameter(2*i, other.GetParameter(2*i)  );
			this->SetParameter(2*i+1, other.GetParameter(2*i+1)  );
		}
	}


	__host__ __device__
	inline GaussN<DIM>& operator=( GaussN<DIM> const& other)
	{
		if(this == &other) return *this;

		BaseFunctor<GaussN<DIM>,GReal_t,DIM+DIM>::operator=(other);

		this->fAutoNormalize = other.fAutoNormalize;
		for(size_t i=0; i<DIM; i++){
			this->fPosition[i]= other.fPosition[i];
			this->SetParameter(2*i, other.GetParameter(2*i)  );
			this->SetParameter(2*i+1, other.GetParameter(2*i+1)  );
		}
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GReal_t Evaluate(T* x)
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

	__host__ __device__
	GBool_t IsAutoNormalized() const {
		return fAutoNormalize;
	}
	__host__ __device__
	void AutoNormalize(GBool_t autoNormalize=kTrue) {
		fAutoNormalize = autoNormalize;
	}

	GUInt_t  fPosition[DIM];
	GBool_t  fAutoNormalize;

};

template<size_t DIM>
struct GaussNAnalyticIntegral
{

	GaussNAnalyticIntegral(std::array<GReal_t,DIM> const& lower_lim,
			std::array<GReal_t,DIM> const& upper_lim)
	{
		for(size_t i=0; i<DIM; i++)
		{
			LowerLim[i]=lower_lim[i];
			UpperLim[i]=upper_lim[i];
		}
	}


	inline GaussNAnalyticIntegral(GaussNAnalyticIntegral<DIM>const& other)
	{
		for(size_t i=0; i<DIM; i++)
		{
			LowerLim[i]=other.LowerLim[i];
			UpperLim[i]=other.UpperLim[i];
		}
	}


	inline GaussNAnalyticIntegral<DIM>& operator=( GaussNAnalyticIntegral<DIM> const& other)
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

	inline GReal_t cumulative(const GReal_t mean, const GReal_t sigma, const GReal_t x)
	{
		return sqrt(M_PI_2)*sigma*(1.0 + erf( (x-mean)/( sigma*sqrt(2) ) ) );
	}

};


}
#endif /* GAUSS_H_ */
