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
 *  Created on: 22/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <hydra/detail/Config.h>
#include <hydra/cpp/System.h>
#include <hydra/Types.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/FFTW.h>
#include <hydra/Algorithm.h>
#include <hydra/Zip.h>
#include <hydra/detail/Convolution.inl>
#include <hydra/Complex.h>

#include <thrust/iterator/transform_iterator.h>

#include <utility>
#include <type_traits>

namespace hydra {

template<typename Functor, typename Kernel, typename Iterable,
     typename T = typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type>
inline typename std::enable_if< std::is_floating_point<T>::value && hydra::detail::is_iterable<Iterable>::value, void>::type
convolute(Functor const& functor, Kernel const& kernel,  T min,  T max, Iterable&& output  ){

	typedef hydra::complex<double> complex_type;

	int nsamples = std::forward<Iterable>(output).size();
	     T delta = (max - min)/(nsamples);

	hydra::cpp::vector< complex_type > complex_buffer(nsamples+1, complex_type(0,0));
	hydra::cpp::vector<T>   kernel_samples(2*nsamples, 0.0);
	hydra::cpp::vector<T>  functor_samples(2*nsamples, 0.0);


	// sample kernel
	auto kernel_sampler = hydra::detail::convolution::KernelSampler<Kernel>(kernel, nsamples, delta);

	hydra::transform( range(0, 2*nsamples), kernel_samples, kernel_sampler);

	// sample function
	auto functor_sampler = hydra::detail::convolution::FunctorSampler<Functor>(functor, nsamples,  min, delta);

	hydra::transform( range(0, 2*nsamples), functor_samples, functor_sampler);

	//transform kernel
	auto fft_kernel = hydra::RealToComplexFFT<T>( kernel_samples.size());

	fft_kernel.LoadInputData(kernel_samples);
	fft_kernel.Execute();

	auto fft_kernel_output =  fft_kernel.GetOutputData();
	auto fft_kernel_range  = make_range( fft_kernel_output.first,
			fft_kernel_output.first + fft_kernel_output.second);

	//transform functor
	auto fft_functor = hydra::RealToComplexFFT<T>( functor_samples.size() );

	fft_functor.LoadInputData(functor_samples);
	fft_functor.Execute();

	auto fft_functor_output =  fft_functor.GetOutputData();
	auto fft_functor_range  = make_range( fft_functor_output.first,
			fft_functor_output.first + fft_functor_output.second);

	//element wise product
	auto ffts = hydra::zip(fft_functor_range,  fft_kernel_range );

	hydra::transform( ffts, complex_buffer, detail::convolution::MultiplyFFT<T>());

	//transform product back to real

	auto fft_product = hydra::ComplexToRealFFT<T>( 2*nsamples );

	fft_product.LoadInputData(complex_buffer);
	fft_product.Execute();

	auto fft_product_output =  fft_product.GetOutputData();
	auto normalize_fft =  detail::convolution::NormalizeFFT<T>( 2*nsamples);

	auto fft_product_range = make_range(
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator( fft_product_output.first,normalize_fft),
			HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(	fft_product_output.first + nsamples,normalize_fft));

	hydra::copy(fft_product_range,  std::forward<Iterable>(output));

}


}  // namespace hydra



#endif /* CONVOLUTION_H_ */
