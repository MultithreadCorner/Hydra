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
#include <hydra/detail/CompositeBase.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>

namespace hydra {
/*
namespace detail {

template<typename Kernel, typename  Functor>
struct CauchyProduct
{
	CauchyProduct()=delete;

	CauchyProduct(Functor functor, Kernel kernel, double min, double max,size_t n):
		fFunctor(functor),
		fKernel(kernel),
		fKMin(min),
		fKMax(max),
		fDelta((max-min/n)),
		fN(n)
	{ }

	__hydra_host__ __hydra_device__
	CauchyProduct( CauchyProduct const& other):
	fKernel(other.GetKernel()),
	fFunctor(other.GetFunctor()),
	fKMin(other.GetKMin()),
	fKMax(other.GetKMax()),
	fDelta(other.GetDelta()),
	fN(other.GetN())
	{}

	__hydra_host__ __hydra_device__
	CauchyProduct& operator=( CauchyProduct const& other){

		if(this == &other) return *this;

		fKernel  =other.GetKernel();
		fFunctor =other.GetFunctor();
		fDelta   =other.GetDelta();
		fKMin    =other.GetKMin();
		fKMax    =other.GetKMax();
		fN       =other.GetN();

		return *this;
	}

	__hydra_host__ __hydra_device__
	const Kernel& GetKernel() const {
		return fKernel;
	}
	__hydra_host__ __hydra_device__
	void SetKernel(Kernel const& kernel) {
		fKernel = kernel;
	}

	const Functor& GetFunctor() const {
		return fFunctor;
	}

	void SetFunctor(Functor const& functor) {
		fFunctor = functor;
	}

	__hydra_host__ __hydra_device__
	size_t GetN() const {
		return fN;
	}

	double GetKMax() const {
		return fKMax;
	}

	double GetKMin() const {
		return fKMin;
	}

	double GetDelta() const {
		return fDelta;
	}

	__hydra_host__ __hydra_device__
	inline double operator()(double x)
	{

		double result = 0;
		double normalization = 0;

		for(int i=0 ; i<N; i++){

			double	    x_i = i*fDelta + fKMin;
			double	delta_i = x - x_i;
			double  kernel  = fKernel(delta_i);
			result         += fFunctor(x)*kernel;
			normalization  += kernel;
		}

		return result/normalization;
	}


private:

	Kernel  fKernel;
	Functor fFunctor;
	double  fKMin;
	double  fKMax;
	double  fDelta;
	size_t  fN;
};

}  // namespace detail
*/

/**
 * FIXME
 */
template<typename Functor, typename Kernel,  unsigned int ArgIndex=0>
class Convolution:  public detail::CompositeBase<Functor, Kernel>
{

public:

	Convolution( Functor const& functor, Kernel const& kernel, double kmin, double kmax, size_t nsamples =500):
		detail::CompositeBase<Functor, Kernel>(functor,kernel),
		fKMin(kmin),
		fKMax(kmax),
		fDelta((kmax-kmin)/nsamples),
		fN(nsamples),
		fNormalization(0)
		{
		for(int i=0 ; i<fN; i++){
			GReal_t	    x_i = i*fDelta + fKMin;
			fNormalization += HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(x_i);
		}
		}

	__hydra_host__ __hydra_device__
	Convolution( Convolution<Functor,Kernel, ArgIndex> const& other):
	detail::CompositeBase<Functor, Kernel>(other),
	fKMax(other.GetKMax()),
	fKMin(other.GetKMin()),
	fDelta(other.GetDelta()),
	fN(other.GetN()),
	fNormalization(other.GetNormalization())
	{}

	__hydra_host__ __hydra_device__
	Convolution<Functor,Kernel, ArgIndex>&
	operator=(Convolution<Functor,Kernel, ArgIndex> const& other){

		if(this == &other) return *this;

		detail::CompositeBase<Functor, Kernel>::operator=(other);
		fKMax 	 = other.GetKMax();
		fKMin 	 = other.GetKMin();
		fDelta   = other.GetDelta();
		fN       = other.GetN();
		fNormalization = other.GetNormalization();
		return *this;
	}

	__hydra_host__ __hydra_device__
	double GetKMax() const {
		return fKMax;
	}

	__hydra_host__ __hydra_device__
	void SetKMax(double kMax) {
		fKMax = kMax;
	}

	__hydra_host__ __hydra_device__
	double GetKMin() const {
		return fKMin;
	}

	__hydra_host__ __hydra_device__
	void SetKMin(double kMin) {
		fKMin = kMin;
	}
	__hydra_host__ __hydra_device__
	double GetDelta() const {
		return fDelta;
	}
	__hydra_host__ __hydra_device__
	void SetDelta(double delta) {
		fDelta = delta;
	}
	__hydra_host__ __hydra_device__
	GReal_t GetNormalization() const {
		return fNormalization;
	}
	__hydra_host__ __hydra_device__
	void SetNormalization(GReal_t normalization) {
		fNormalization = normalization;
	}

	GReal_t GetN() const {
		return fN;
	}

	void SetN(GReal_t n) {
		fN = n;
	}

	template<typename T>
     __hydra_host__ __hydra_device__
	inline double operator()(unsigned int n, T*x)
	{

		GReal_t result = 0;
		GReal_t X  = x[ArgIndex];

		for(int i=0 ; i<fN; i++){

			GReal_t	    x_i = i*fDelta + fKMin;
			GReal_t	delta_i = X - x_i;
			GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
			result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;

		}

		return result/fNormalization;
	}


	template<typename T>
	__hydra_host__ __hydra_device__
	inline double operator()(unsigned int n, T*x)
	{

		GReal_t result = 0;
		GReal_t X  = x[ArgIndex];

		for(int i=0 ; i<fN; i++){

			GReal_t	    x_i = i*fDelta + fKMin;
			GReal_t	delta_i = X - x_i;
			GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
			result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;

		}

		return result/fNormalization;
	}

	template<template<typename ...T>class C>
	__hydra_host__ __hydra_device__
	inline double operator()(C<T...>)
	{

		GReal_t result = 0;
		GReal_t X  = x[ArgIndex];

		for(int i=0 ; i<fN; i++){

			GReal_t	    x_i = i*fDelta + fKMin;
			GReal_t	delta_i = X - x_i;
			GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
			result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;

		}

		return result/fNormalization;
	}


		__hydra_host__ __hydra_device__
		inline double operator()(GReal_t x)
		{

			GReal_t result = 0;
			GReal_t X  = x[ArgIndex];

			for(int i=0 ; i<fN; i++){

				GReal_t	    x_i = i*fDelta + fKMin;
				GReal_t	delta_i = X - x_i;
				GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
				result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;

			}

			return result/fNormalization;
		}




private:

     __hydra_host__ __hydra_device__
	inline double convolute(double x)  const
	{

		GReal_t result = 0;
		GReal_t X  = x;

		for(int i=0 ; i<fN; i++){

			GReal_t	    x_i = i*fDelta + fKMin;
			GReal_t	delta_i = X - x_i;
			GReal_t  kernel = HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors())(delta_i);
			result         += HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors())(x_i)*kernel;

		}

		return result/fNormalization;
	}

	GReal_t fKMax;
	GReal_t fKMin;
	GReal_t fDelta;
	GReal_t fNormalization;
	GReal_t fN;


};


}  // namespace hydra{


#endif /* CONVOLUTION_H_ */
