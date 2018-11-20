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
 * ConvolutionFFT.h
 *
 *  Created on: 19/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTIONFFT_H_
#define CONVOLUTIONFFT_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/BaseCompositeFunctor.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/FFTW.h>
#include <hydra/Algorithm.h>

namespace hydra {

namespace detail {

namespace convolution {

template<typename Kernel>
struct KernelSampler
{
	KernelSampler()=delete;

	KernelSampler(Kernel const& kernel, int nsamples , double delta):
		fDelta(delta),
		fNZero(nsamples),
		fNMin(0.5*nsamples),
	    fNMax(1.5*nsamples),
		fKernel(kernel)
	{}

	KernelSampler( KernelSampler<Kernel> const& other):
		fDelta(other.GetDelta()),
		fNZero(other.GetNZero()),
		fNMin(other.GetNMin()),
		fNMax(other.GetNMax()),
		fKernel(other.GetKernel())
	{}

	KernelSampler<Kernel>&
	operator=( KernelSampler<Kernel> const& other)
	{
		if(this == &other) return *this;

			fDelta    = other.GetDelta();
			fNZero    = other.GetNZero();
			fNMin     = other.GetNMin();
			fNMax     = other.GetNMax();
			fKernel   = other.GetKernel();

		 return *this;
	}


	double operator()( int index) const	{

        double value=0.0;

		if( 0 <= index && index <=  fNMin ){

			double t =  index*fDelta ;

		    value = fKernel(t);
		}

		if (  index >=  fNMax ){

			double t =  (index-2*nsamples)*fDelta;

			std::cout << index << " "<< t << std::endl;

			value = fKernel(t);

		}

       return value;

	}

	double GetDelta() const {	return fDelta; }

	void SetDelta(double delta) { fDelta = delta; }

	int GetNZero() const { return fNZero; }

	void SetNZero(int n) { fNZero = n; }

	int GetNMax() const { return fNMax; }

	void SetNMax(int nMax) { fNMax = nMax; }

	int GetNMin() const { return fNMin; }

	void SetNMin(int nMin) { fNMin = nMin; }

	Kernel GetKernel() const { return fKernel;}

	void SetKernel(Kernel kernel) { fKernel = kernel;}


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

	FunctorSampler( FunctorSampler<Functor> const& other):
		fDelta(other.GetDelta()),
		fMin(other.GetMin()),
		fNSamples(other.GetNSamples()),
		fFunctor(other.GetKernel())
	{}

	FunctorSampler<Functor>&
	operator=( FunctorSampler<Functor> const& other)
	{
		if(this == &other) return *this;

			fDelta    = other.GetDelta();
			fMin      = other.GetNMin();
			fNSamples = other.GetNSamples();
			fFunctor  = other.GetKernel();

		 return *this;
	}


	double operator()( int index) const	{

        double value=0.0;

		if( index <  fNSamples ){

			double t =  index*fDelta + fMin;

		    value = fFunctor(t);
		}

       return value;
	}

	double GetDelta() const {	return fDelta; }

	void SetDelta(double delta) { fDelta = delta; }

	int GetMin() const { return fMin; }

	void SetMin(int Min) { fMin = Min; }

	Kernel GetFunctor() const { return fFunctor;}

	void SetFunctor(Kernel kernel) { fFunctor = kernel;}

	int GetNSamples() const { return fNSamples; }

	void SetNSamples(int nSamples) { fNSamples = nSamples;}

private:

	double fDelta;
	double fMin;
	int    fNSamples;
	Functor fFunctor;
};


}  // namespace convolution

}  // namespace detail


template<typename Functor, typename Kernel>
class ConvolutionFFT
{
	ConvolutionFFT()=delete;

	ConvolutionFFT(Functor const& functor,   Kernel const& kernel, unsigned int nsamples,
			double min, double max, double buffer_fraction=0.15 ):
				fFunctor(fucntor),
				fKernel(kernel),
				fMax(max),
				fMin(min),
				fFraction( buffer_fraction ),
				fNSamples(nsamples),
				fKernelSamples(int(2*nsamples),0.0),
				fFunctorSamples(int(2*nsamples),0.0)
	{}



private:

	void SampleKernel(){

		using detail::convolution::KernelSampler;
		auto sampler = KernelSampler(fKernel, fNSamples, (fMax -fMin)/(2*fNSamples));

		hydra::transform( range(0, 2*fNSamples), fKernelSamples, sampler);

	}

	void SampleFunctor(){

		using detail::convolution::FunctorSampler;
		auto sampler = FunctorSampler(fKernel, fNSamples, (fMax -fMin)/(2*fNSamples));

		hydra::transform( range(0, (2+fFraction)*fNSamples), fFunctorSamples, sampler);

	}



	std::vector<double> fKernelSamples;
	std::vector<double> fFunctorSamples;
	Functor	fFunctor;
	Kernel  fKernel;
	double fMax;
	double fMin;
	double fFraction;
	unsigned int fNSamples;


};


}  // namespace hydra




#endif /* CONVOLUTIONFFT_H_ */
