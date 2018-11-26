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
#include <hydra/detail/BackendPolicy.h>
#include <hydra/cpp/System.h>
#include <hydra/Types.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/FFTW.h>
#include <hydra/Algorithm.h>
#include <hydra/Zip.h>
#include <hydra/detail/Convolution.inl>
#include <hydra/Complex.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/thrust/memory.h>

#include <utility>
#include <type_traits>

namespace hydra {

template<hydra::detail::Backend BACKEND, typename Functor, typename Kernel, typename Iterable,
     typename T = typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type>
inline typename std::enable_if<std::is_floating_point<T>::value && hydra::detail::is_iterable<Iterable>::value, void>::type
convolute(detail::BackendPolicy<BACKEND> policy, Functor const& functor, Kernel const& kernel,
		     T min,  T max, Iterable&& output , bool power_up=true ){

	typedef hydra::complex<double> complex_type;

	if(power_up) std::forward<Iterable>(output).resize(	hydra::detail::convolution::upper_power_of_two(
			std::forward<Iterable>(output).size()));

	int nsamples = std::forward<Iterable>(output).size();


	T delta = (max - min)/(nsamples);

	auto complex_buffer  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<complex_type>(policy, nsamples+1);
	auto kernel_samples  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(policy, 2*nsamples);
	auto functor_samples = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(policy, 2*nsamples);

	//
	auto counting_samples = range(0, 2*nsamples);
	// sample kernel
	auto kernel_sampler = hydra::detail::convolution::KernelSampler<Kernel>(kernel, nsamples, delta);

	HYDRA_EXTERNAL_NS::thrust::transform(policy, counting_samples.begin(), counting_samples.end(),
			kernel_samples.first , kernel_sampler);

	// sample function
	auto functor_sampler = hydra::detail::convolution::FunctorSampler<Functor>(functor, nsamples,  min, delta);

	HYDRA_EXTERNAL_NS::thrust::transform( policy, counting_samples.begin(), counting_samples.end(),
			functor_samples.first, functor_sampler);

	//transform kernel
	auto fft_kernel = hydra::RealToComplexFFT<T>( kernel_samples.second );

	fft_kernel.LoadInputData( kernel_samples.second,
			HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(kernel_samples.first));
	fft_kernel.Execute();

	auto fft_kernel_output =  fft_kernel.GetOutputData();
	auto fft_kernel_range  = make_range( fft_kernel_output.first,
			fft_kernel_output.first + fft_kernel_output.second);

	//transform functor
	auto fft_functor = hydra::RealToComplexFFT<T>( functor_samples.second );

	fft_functor.LoadInputData(functor_samples.second,
			HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(functor_samples.first));
	fft_functor.Execute();

	auto fft_functor_output =  fft_functor.GetOutputData();
	auto fft_functor_range  = make_range( fft_functor_output.first,
			fft_functor_output.first + fft_functor_output.second);

	//element wise product
	auto ffts = hydra::zip(fft_functor_range,  fft_kernel_range );


	HYDRA_EXTERNAL_NS::thrust::transform( policy, ffts.begin(),  ffts.end(),
			complex_buffer.first, detail::convolution::MultiplyFFT<T>());

	//transform product back to real

	auto fft_product = hydra::ComplexToRealFFT<T>( 2*nsamples );

	fft_product.LoadInputData(complex_buffer.second,
			reinterpret_cast<double (*)[2]>(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(complex_buffer.first)));
	fft_product.Execute();

	auto fft_product_output =  fft_product.GetOutputData();

	T n = ::pow(10.0, (long unsigned)::log10((double)nsamples*nsamples ));

	auto normalize_fft =  detail::convolution::NormalizeFFT<T>(n);

    auto first = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator( fft_product_output.first,normalize_fft);
    auto last  = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(fft_product_output.first + nsamples,normalize_fft);

	auto fft_product_range = make_range(first, last);

	hydra::copy(fft_product_range,  std::forward<Iterable>(output));

	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer( policy,  complex_buffer.first  );
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer( policy,  kernel_samples.first  );
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer( policy, functor_samples.first  );
}


}  // namespace hydra



#endif /* CONVOLUTION_H_ */
