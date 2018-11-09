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
#include <hydra/CubicSpiline.h>

namespace hydra {


template<typename Functor, typename Kernel, unsigned int N, unsigned int ArgIndex=0>
class Convolution:  public BaseCompositeFunctor<Convolution<Functor, Kernel, N, ArgIndex>, double, Functor, Kernel>
{

public:
	Convolution() = delete;

	Convolution( Functor const& functor, Kernel const& kernel, double kmin, double kmax):
		BaseCompositeFunctor< Convolution<Functor, Kernel,N, ArgIndex>, double, Functor, Kernel>(functor,kernel),
		fMin(kmin),
		fMax(kmax),
		fSpiline()
		{
			reconvolute();
		}

	__hydra_host__ __hydra_device__
	Convolution( Convolution<Functor, Kernel,N,  ArgIndex> const& other):
	BaseCompositeFunctor< Convolution<Functor, Kernel,N,  ArgIndex>, double,Functor, Kernel>(other),
	fMax(other.GetMax()),
	fMin(other.GetMin()),
	fSpiline(other.GetSpiline())
	{}

	__hydra_host__ __hydra_device__
	Convolution<Functor,Kernel,N,  ArgIndex>&
	operator=(Convolution<Functor,Kernel,N,  ArgIndex> const& other){

		if(this == &other) return *this;

		BaseCompositeFunctor<Convolution<Functor,Kernel,N,  ArgIndex>, double, Functor, Kernel>::operator=(other);

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

	const CubicSpiline<N+1>& GetSpiline() const {
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

    	 /*
    	  * 1. Transform from [fMax, fMin] to [-1,1]
    	  * 2. Perform the convolution calculation in the interval [ -5, 5]
    	  * 3.
    	  */

    	 GReal_t delta = (fMax - fMin);

    	 GReal_t _max = fMax + 2.0*delta;
    	 GReal_t _min = fMin - 2.0*delta;
    	 GReal_t _delta = (_max - _min)/N;

    	 for(unsigned int i=0 ; i<N+1; i++){

    		 GReal_t X_i   = i*_delta + _min;
    		 GReal_t norm_i=0, result_i=0;

    		// std::cout << "X_i = " << X_i << std::endl;

    		 for(unsigned int j=0 ; j<N+1; j++){

    			 GReal_t	    x_j = j*_delta + _min;
    			 GReal_t	delta_ij = X_i - x_j;
    			 GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_ij);
    			 norm_i           += kernel;
    			 result_i         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_j)*kernel;
    			// std::cout << "   X_j = " << x_j
    				//	   << "   delta_ij = " << delta_ij  << std::endl;
    		 }

    		// std::cout << "   X_i = " << X_i << " norm_i = " << norm_i << " result_i "<<  result_i << std::endl;
    		 fSpiline.SetX(i, X_i);
    		 fSpiline.SetD(i, result_i/norm_i);

    	 }
     }

    CubicSpiline<N+1> fSpiline;

	GReal_t fMax;
	GReal_t fMin;

};

template<unsigned int N, unsigned int ArgIndex, typename Functor, typename Kernel>
auto convolute( Functor const& functor, Kernel const& kernel, double kmin, double kmax)
-> Convolution<Functor, Kernel, N, ArgIndex>
{

	return Convolution<Functor, Kernel, N, ArgIndex>( functor, kernel,kmin,kmax);

}


}  // namespace hydra


#endif /* CONVOLUTION_H_ */
