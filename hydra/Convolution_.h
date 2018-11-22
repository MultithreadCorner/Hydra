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
 * Convolution.h
 *
 *  Created on: 29/08/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/BaseCompositeFunctor.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/CubicSpiline.h>
#include <hydra/detail/external/thrust/transform_reduce.h>

namespace hydra {

namespace detail {

template<typename Functor, typename Kernel>
struct ConvolutionUnary
{

	ConvolutionUnary()=delete;

	ConvolutionUnary(Functor functor, Kernel  kernel, double x, double delta,	double min ):
	fFunctor(functor),
	fKernel(kernel),
	fX(x),
	fDelta(delta),
	fMin(min)
	{}

	__hydra_host__ __hydra_device__
	ConvolutionUnary(ConvolutionUnary<Functor,Kernel> const& other ):
		fFunctor(other.GetFunctor()),
		fKernel(other.GetKernel()),
		fX(other.GetX()),
		fDelta(other.GetDelta()),
		fMin(other.GetMin())
	{}

	__hydra_host__ __hydra_device__
	ConvolutionUnary<Functor,Kernel>&
	operator=(ConvolutionUnary<Functor,Kernel> const& other ){

		if(this == &other) return *this;

		fFunctor=other.GetFunctor();
		fKernel=other.GetKernel();
		fX=other.GetX();
		fDelta=other.GetDelta();
		fMin=other.GetMin();

		return *this;

	}

	__hydra_host__ __hydra_device__
	double GetDelta() const {
		return fDelta;
	}

	__hydra_host__ __hydra_device__
	Functor GetFunctor() const {
		return fFunctor;
	}

	__hydra_host__ __hydra_device__
	Kernel GetKernel() const {
		return fKernel;
	}

	__hydra_host__ __hydra_device__
	double GetMin() const {
		return fMin;
	}

	__hydra_host__ __hydra_device__
	double GetX() const {
		return fX;
	}

	__hydra_host__ __hydra_device__
	hydra::pair<double, double> operator()(size_t i)
	{

		double	    x  = i*fDelta + fMin;
		double	sigma    = fX - x;
		double  kernel   = fKernel(sigma);

		//printf("i=%d fX=%f x =%f sigma=%f \n", i, fX, x, sigma);
		 return hydra::make_pair( fFunctor(x)*kernel, kernel );
	}


private:

	Functor fFunctor;
	Kernel  fKernel;
	double fX;
	double fDelta;
	double fMin;

};

struct ConvolutionBinary
{
	__hydra_host__ __hydra_device__
    hydra::pair<double, double>
	operator()(hydra::pair<double, double> const& left, hydra::pair<double, double> const& right){

		return hydra::make_pair(left.first + right.first , left.second + right.second );
	}
};

}  // namespace detail

template<typename Functor, typename Kernel,typename  System, unsigned int N, unsigned int ArgIndex=0>
class Convolution;

template<typename Functor, typename Kernel, hydra::detail::Backend  BACKEND, unsigned int N, unsigned int ArgIndex>
class Convolution<Functor, Kernel, hydra::detail::BackendPolicy<BACKEND>, N, ArgIndex>:  public BaseCompositeFunctor<Convolution<Functor, Kernel, hydra::detail::BackendPolicy<BACKEND>, N, ArgIndex>, double, Functor, Kernel>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;

public:
	Convolution() = delete;

	Convolution( Functor const& functor, Kernel const& kernel, double kmin, double kmax):
		BaseCompositeFunctor< Convolution<Functor, Kernel, hydra::detail::BackendPolicy<BACKEND>,N, ArgIndex>, double, Functor, Kernel>(functor,kernel),
		fMin(kmin),
		fMax(kmax),
		fSpiline()
		{
			reconvolute();
		}

	__hydra_host__ __hydra_device__
	Convolution( Convolution<Functor, Kernel,hydra::detail::BackendPolicy<BACKEND>,N,  ArgIndex> const& other):
	BaseCompositeFunctor< Convolution<Functor, Kernel, hydra::detail::BackendPolicy<BACKEND>,N,  ArgIndex>, double,Functor, Kernel>(other),
	fMax(other.GetMax()),
	fMin(other.GetMin()),
	fSpiline(other.GetSpiline())
	{}

	__hydra_host__ __hydra_device__
	Convolution<Functor,Kernel, hydra::detail::BackendPolicy<BACKEND>,N,  ArgIndex>&
	operator=(Convolution<Functor,Kernel, hydra::detail::BackendPolicy<BACKEND>,N,  ArgIndex> const& other){

		if(this == &other) return *this;

		BaseCompositeFunctor<Convolution<Functor,Kernel, hydra::detail::BackendPolicy<BACKEND>,N,  ArgIndex>, double, Functor, Kernel>::operator=(other);

		fMax 	 = other.GetMax();
		fMin 	 = other.GetMin();
		fSpiline = other.GetSpiline();

		return *this;
	}

	__hydra_host__ __hydra_device__
	double GetMax() const {
		return fMax;
	}

	__hydra_host__ __hydra_device__
	void SetMax(double kMax) {
		fMax = kMax;
	}

	__hydra_host__ __hydra_device__
	double GetMin() const {
		return fMin;
	}

	__hydra_host__ __hydra_device__
	void SetMin(double kMin) {
		fMin = kMin;
	}

	const CubicSpiline<2*N+1>& GetSpiline() const {
		return fSpiline;
	}

	virtual void Update() override {

	 reconvolute();

	}

	template<typename T>
     __hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int n, T*x) const	{


		GReal_t X  = x[ArgIndex];

		return fSpiline(X);
	}


	template<typename T>
     __hydra_host__ __hydra_device__
	inline double Evaluate(T& x) const {

		GReal_t X  = get<ArgIndex>(x);

		return fSpiline(X);
	}

	virtual ~Convolution()=default;



private:

     void reconvolute(){

    	 GReal_t delta = (fMax - fMin);
         GReal_t _min =  fMin - 0.5*delta;
         GReal_t _max =  fMax + 0.5*delta;
         GReal_t _epsilon =  (-_min + _max)/(N*2+1);

    	 for(unsigned int i=0 ; i<N*2+1; i++){

    		 GReal_t X_i   = i*_epsilon + _min;

    		 hydra::pair<double, double> init(0.0, 0.0);

    		 auto index = hydra::range(0,N*2+1);

    		 hydra::pair<double, double> result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(system_t(), index.begin() ,index.end(),
    				 detail::ConvolutionUnary<Functor, Kernel>(HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors()),
    						    		HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors()),
    						    		     X_i, _epsilon, _min ) ,init, detail::ConvolutionBinary() );

    		 fSpiline.SetX(i, X_i);
    		 fSpiline.SetD(i, result.first);

    	 }
     }

    CubicSpiline<N*2+1> fSpiline;

	GReal_t fMax;
	GReal_t fMin;

};

template< unsigned int N, unsigned int ArgIndex, typename Functor, typename Kernel, detail::Backend  BACKEND>
auto convolute( detail::BackendPolicy<BACKEND> backend, Functor const& functor, Kernel const& kernel, double kmin, double kmax)
-> Convolution<Functor, Kernel,hydra::detail::BackendPolicy<BACKEND> , N, ArgIndex>
{

	return Convolution<Functor, Kernel, detail::BackendPolicy<BACKEND> , N, ArgIndex>( functor, kernel,kmin,kmax);

}


}  // namespace hydra


#endif /* CONVOLUTION_H_ */
