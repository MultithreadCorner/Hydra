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

namespace hydra {


template<typename Functor, typename Kernel, unsigned int N, unsigned int ArgIndex=0>
class Convolution:  public BaseCompositeFunctor<Convolution<Functor, Kernel, ArgIndex>, double, Functor, Kernel>
{

public:
	Convolution() = delete;

	Convolution( Functor const& functor, Kernel const& kernel, double kmin, double kmax, unsigned nsamples =100):
		BaseCompositeFunctor< Convolution<Functor, Kernel, ArgIndex>, double, Functor, Kernel>(functor,kernel),
		fMin(kmin),
		fMax(kmax),
		fSpiline()
		{


		}

	__hydra_host__ __hydra_device__
	Convolution( Convolution<Functor, Kernel, ArgIndex> const& other):
	BaseCompositeFunctor< Convolution<Functor, Kernel, ArgIndex>, double,Functor, Kernel>(other),
	fMax(other.GetKernelMax()),
	fMin(other.GetKernelMin()),
	fSpiline(other.GetSpiline())
	{}

	__hydra_host__ __hydra_device__
	Convolution<Functor,Kernel, ArgIndex>&
	operator=(Convolution<Functor,Kernel, ArgIndex> const& other){

		if(this == &other) return *this;

		BaseCompositeFunctor<Convolution<Functor,Kernel, ArgIndex>, double, Functor, Kernel>::operator=(other);

		fMax 	 = other.GetKernelMax();
		fMin 	 = other.GetKernelMin();
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

	const CubicSpiline<N>& GetSpiline() const {
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


		for(unsigned int i=0 ; i<N; i++){

			GReal_t	    x_i = i*fDelta + fMin;
			GReal_t	delta_i = X - x_i;
			GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
			norm           += kernel;
			result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;
			fSpiline;
		}
     }

    CubicSpiline<N> fSpiline;

	GReal_t fMax;
	GReal_t fMin;

};

template<unsigned int ArgIndex, typename Functor, typename Kernel>
auto convolute( Functor const& functor, Kernel const& kernel, double kmin, double kmax, unsigned nsamples =100 )
-> Convolution<Functor, Kernel, ArgIndex>
{

	return Convolution<Functor, Kernel, ArgIndex>( functor, kernel,kmin,kmax,nsamples);

}


}  // namespace hydra


#endif /* CONVOLUTION_H_ */
