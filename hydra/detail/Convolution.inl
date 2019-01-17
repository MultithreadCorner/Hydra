/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * Convolution.inl
 *
 *  Created on: 22/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTION_INL_
#define CONVOLUTION_INL_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Algorithm.h>
#include <hydra/Zip.h>
#include <hydra/Complex.h>
#include <functional>
#include <utility>
#include <type_traits>


namespace hydra {

namespace detail {

namespace convolution {

size_t upper_power_of_two(size_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;

}

template<typename Kernel>
struct KernelSampler
{
	KernelSampler()=delete;

	KernelSampler(Kernel const& kernel, int nsamples , double delta):
		fDelta(delta),

		fKernel(kernel)
	{
		fNZero = nsamples;
		fNMin  = nsamples- nsamples/16;
	    fNMax  = nsamples+ nsamples/16;
	}

	__hydra_host__ __hydra_device__
	KernelSampler( KernelSampler<Kernel> const& other):
		fDelta(other.GetDelta()),
		fNZero(other.GetNZero()),
		fNMin(other.GetNMin()),
		fNMax(other.GetNMax()),
		fKernel(other.GetKernel())
	{}

	__hydra_host__ __hydra_device__
	inline KernelSampler<Kernel>& operator=( KernelSampler<Kernel> const& other)
	{
		if(this == &other) return *this;

			fDelta    = other.GetDelta();
			fNZero    = other.GetNZero();
			fNMin     = other.GetNMin();
			fNMax     = other.GetNMax();
			fKernel   = other.GetKernel();

		 return *this;
	}


	__hydra_host__ __hydra_device__
	inline double operator()( int index) const	{

        double value=0.0;

        if( (0 == index) || (index+1 == 2.0*fNZero ) ){

        		    value = 0.5*fKernel(0.0);
        		  //  std::cout << "branch 0 :" << value << std::endl;
        	}

		if( (0 < index) && (index <=  fNMin) ){

			double t =  index*fDelta ;

		    value = fKernel(t);
		    //std::cout << "branch 1 :" << value << std::endl;
		}

		if (  index >=  fNMax && (index+1 < 2.0*fNZero)){

			double t =  (index+1-2.0*fNZero)*fDelta;

			value = fKernel(t);
			//std::cout << "branch 2 :" << value << std::endl;
		}
		//std::cout <<"result " << value << std::endl;
       return value;

	}

	__hydra_host__ __hydra_device__
	inline double GetDelta() const {	return fDelta; }

	__hydra_host__ __hydra_device__
	inline void SetDelta(double delta) { fDelta = delta; }

	__hydra_host__ __hydra_device__
	inline int GetNZero() const { return fNZero; }

	__hydra_host__ __hydra_device__
	inline void SetNZero(int n) { fNZero = n; }

	__hydra_host__ __hydra_device__
	inline int GetNMax() const { return fNMax; }

	__hydra_host__ __hydra_device__
	inline void SetNMax(int nMax) { fNMax = nMax; }

	__hydra_host__ __hydra_device__
	inline int GetNMin() const { return fNMin; }

	__hydra_host__ __hydra_device__
	inline void SetNMin(int nMin) { fNMin = nMin; }

	__hydra_host__ __hydra_device__
	inline Kernel GetKernel() const { return fKernel;}


	__hydra_host__ __hydra_device__
	inline void SetKernel(Kernel const& kernel) { fKernel = kernel;}


private:

	double fDelta;
	int    fNZero;
	int    fNMin;
	int    fNMax;

	Kernel fKernel;
};

template<typename Functor>
struct FunctorSampler
{
	FunctorSampler()=delete;

	FunctorSampler(Functor const& functor, int nsamples, double min, double delta):
		fDelta(delta),
		fMin(min),
		fNSamples(nsamples),
		fFunctor(functor)
	{}

	__hydra_host__ __hydra_device__
	inline FunctorSampler( FunctorSampler<Functor> const& other):
		fDelta(other.GetDelta()),
		fMin(other.GetMin()),
		fNSamples(other.GetNSamples()),
		fFunctor(other.GetFunctor())
	{}

	__hydra_host__ __hydra_device__
	inline FunctorSampler<Functor>&	operator=( FunctorSampler<Functor> const& other)
	{
		if(this == &other) return *this;

			fDelta    = other.GetDelta();
			fMin      = other.GetMin();
			fNSamples = other.GetNSamples();
			fFunctor  = other.GetFunctor();

		 return *this;
	}

	__hydra_host__ __hydra_device__
	inline double operator()( int index) const	{

        double value=0.0;

        double t =  index*fDelta + fMin;

        if(  (t >= fMin) && ( t <=  (fNSamples)*fDelta + fMin)){

		    value = fFunctor(t);
		}

       return value;
	}
	__hydra_host__ __hydra_device__
	inline double GetDelta() const {	return fDelta; }

	__hydra_host__ __hydra_device__
	inline void SetDelta(double delta) { fDelta = delta; }

	__hydra_host__ __hydra_device__
	inline double GetMin() const { return fMin; }

	__hydra_host__ __hydra_device__
	inline void SetMin(int Min) { fMin = Min; }

	__hydra_host__ __hydra_device__
	inline Functor GetFunctor() const { return fFunctor;}

	__hydra_host__ __hydra_device__
	inline void SetFunctor(Functor const& functor) { fFunctor = functor;}

	__hydra_host__ __hydra_device__
	inline int GetNSamples() const { return fNSamples; }

	__hydra_host__ __hydra_device__
	inline void SetNSamples(int nSamples) { fNSamples = nSamples;}

private:

	double fDelta;
	double fMin;
	int    fNSamples;
	Functor fFunctor;
};

template<typename T>
struct MultiplyFFT
{
	typedef hydra::complex<T> complex_type;

	__hydra_host__ __hydra_device__
	inline complex_type
	operator()(hydra::tuple<complex_type, complex_type> const& points){

		return hydra::get<0>(points)*hydra::get<1>(points);
	}
};

template<typename T>
struct NormalizeFFT: public  std::unary_function<T,T>
{
	NormalizeFFT()=delete;

	NormalizeFFT(T norm):
		fNorm(1.0/norm)
	{}

	__hydra_host__ __hydra_device__
	inline NormalizeFFT( NormalizeFFT<T> const& other):
	fNorm(other.GetNorm())
	{}

	__hydra_host__ __hydra_device__
	inline NormalizeFFT<T>& operator=( NormalizeFFT<T> const& other){

		if(this == &other) return *this;

		fNorm =other.GetNorm();

		return *this;
	}

	__hydra_host__ __hydra_device__
	inline T operator()(T& value){

		return value*fNorm;
	}

	__hydra_host__ __hydra_device__
	inline T GetNorm() const {
		return fNorm;
	}

	__hydra_host__ __hydra_device__
	inline void SetNorm(size_t norm) {
		fNorm = 1.0/norm;
	}

private:

	T fNorm;
};



}  // namespace convolution

}  // namespace detail

}  // namespace hydra

#endif /* CONVOLUTION_INL_ */
