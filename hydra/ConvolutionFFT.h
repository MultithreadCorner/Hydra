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

namespace hydra {

namespace detail {

namespace convolution {





template<typename Functor, typename Kernel>
class ConvolutionFFT
{
	ConvolutionFFT()=delete;

	ConvolutionFFT(Functor const& functor,   Kernel const& kernel,
			double min, double max, unsigned int nsamples ):
				fFunctor(fucntor),
				fKernel(kernel),
				fMax(max),
				fMin(min),
				fNSamples(nsamples),
				fKernelSamples(int(2*nsamples),0.0),
				fFunctorSamples(int(2*nsamples),0.0)
	{}



private:

	void SampleKernel(){

		unsigned int N_zero = nsamples;
		unsigned int N_min = N_zero - nsamples/2;
		unsigned int N_max = N_zero + nsamples/2;

		double delta = (fMax-fMin)/nsamples;

		auto sampler = [ ]__hydra_dual__ (size_t index ){

			if( 0 < index < N_min)
			{
				double t =  index*delta ;
				return fKernel(t);
			}
			if else (  index > N_max  ){

				double t =  (index-2*N)*delta;
				return fKernel(t);

			}
			else
				return 0.0;

		};




	}

	std::vector<double> fKernelSamples;
	std::vector<double> fFunctorSamples;
	Functor	fFunctor;
	Kernel  fKernel;
	double fMax;
	double fMin;
	unsigned int fNSamples;

};

}  // namespace convolution

}  // namespace detail

}  // namespace hydra




#endif /* CONVOLUTIONFFT_H_ */
