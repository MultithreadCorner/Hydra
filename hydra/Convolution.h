/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
#include <hydra/Types.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/detail/FFTPolicy.h>
#include <hydra/Algorithm.h>
#include <hydra/Zip.h>
#include <hydra/Complex.h>
#include <hydra/detail/Convolution.inl>
#include <hydra/detail/ArgumentTraits.h>
#include <hydra/detail/external/hydra_thrust/transform.h>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/memory.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>

#include <utility>
#include <type_traits>

namespace hydra {

template<detail::Backend BACKEND, detail::FFTCalculator FFTBackend,  typename Functor, typename Kernel, typename Iterable,
     typename T = typename detail::stripped_type<typename hydra_thrust::iterator_traits<decltype(std::declval<Iterable>().begin())>::value_type>::type,
     typename USING_CUDA_BACKEND = typename std::conditional< std::is_convertible<detail::BackendPolicy<BACKEND>,hydra_thrust::system::cuda::tag >::value, std::integral_constant<int, 1>,std::integral_constant<int, 0>>::type,
     typename USING_CUFFT = typename std::conditional< FFTBackend==detail::CuFFT, std::integral_constant<int, 1>,std::integral_constant<int, 0>>::type,
     typename GPU_DATA = typename std::conditional< std::is_convertible<typename hydra_thrust::iterator_system< decltype(std::declval<Iterable>().begin())>::type,
                        hydra_thrust::system::cuda::tag>::value
         , std::integral_constant<int, 1>, std::integral_constant<int, 0> >::type>
inline typename std::enable_if<std::is_floating_point<T>::value  && hydra::detail::is_iterable<Iterable>::value
                   // && (USING_CUDA_BACKEND::value == USING_CUFFT::value)
                   //  && (USING_CUDA_BACKEND::value == GPU_DATA::value),
,void>::type
convolute(detail::BackendPolicy<BACKEND> policy, detail::FFTPolicy<T, FFTBackend> fft_policy,
		  Functor const& functor, Kernel const& kernel,
		  T min,  T max, Iterable&& output, bool power_up=true ){


	typedef hydra::complex<double> complex_type;
	typedef typename detail::FFTPolicy<T, FFTBackend>::R2C _RealToComplexFFT;
	typedef typename detail::FFTPolicy<T, FFTBackend>::C2R _ComplexToRealFFT;


	if(power_up) {
		std::forward<Iterable>(output).resize(
				hydra::detail::convolution::upper_power_of_two(std::forward<Iterable>(output).size()));
	}

	int nsamples = std::forward<Iterable>(output).size();

	T delta = (max - min)/(nsamples);

	auto complex_buffer  = hydra_thrust::get_temporary_buffer<complex_type>(policy, nsamples+1);
	auto kernel_samples  = hydra_thrust::get_temporary_buffer<T>(policy, 2*nsamples);
	auto functor_samples = hydra_thrust::get_temporary_buffer<T>(policy, 2*nsamples);

	//

	auto counting_samples = range(0, 2*nsamples);

	// sample kernel
	auto kernel_sampler = hydra::detail::convolution::KernelSampler<Kernel>(kernel, nsamples, delta);

	hydra_thrust::transform(policy, counting_samples.begin(), counting_samples.end(),
			kernel_samples.first , kernel_sampler);


	//auto norm_factor = hydra_thrust::reduce(policy, kernel_samples.first, kernel_samples.first + kernel_samples.second );


	// sample function
	auto functor_sampler = hydra::detail::convolution::FunctorSampler<Functor>(functor, nsamples,  min, delta);

	hydra_thrust::transform( policy, counting_samples.begin(), counting_samples.end(),
			functor_samples.first, functor_sampler);

	//norm_factor *= hydra_thrust::reduce(policy, functor_samples.first, functor_samples.first + functor_samples.second );

	//transform kernel
	auto fft_kernel = _RealToComplexFFT( kernel_samples.second );

	fft_kernel.LoadInputData( kernel_samples.second, kernel_samples.first);
	fft_kernel.Execute();

	auto fft_kernel_output =  fft_kernel.GetOutputData();
	auto fft_kernel_range  = make_range( fft_kernel_output.first,
			fft_kernel_output.first + fft_kernel_output.second);

	//transform functor
	auto fft_functor = _RealToComplexFFT( functor_samples.second );

	fft_functor.LoadInputData(functor_samples.second,functor_samples.first);
	fft_functor.Execute();

	auto fft_functor_output =  fft_functor.GetOutputData();
	auto fft_functor_range  = make_range( fft_functor_output.first,
			fft_functor_output.first + fft_functor_output.second);

	//element wise product
	auto ffts = hydra::zip(fft_functor_range,  fft_kernel_range );


	hydra_thrust::transform( policy, ffts.begin(),  ffts.end(),
			complex_buffer.first, detail::convolution::MultiplyFFT<T>());

	//transform product back to real


	auto fft_product = _ComplexToRealFFT( 2*complex_buffer.second-2 );


	fft_product.LoadInputData(complex_buffer.second, complex_buffer.first);
	fft_product.Execute();

	auto fft_product_output =  fft_product.GetOutputData();


	T n = 2*complex_buffer.second-2;

	auto normalize_fft =  detail::convolution::NormalizeFFT<T>(n);

    	auto first = hydra_thrust::make_transform_iterator( fft_product_output.first,normalize_fft);
    	auto last  = hydra_thrust::make_transform_iterator(fft_product_output.first + nsamples+1,normalize_fft);

	auto fft_product_range = make_range(first, last);

	hydra::copy(fft_product_range,  std::forward<Iterable>(output));

	hydra_thrust::return_temporary_buffer( policy,  complex_buffer.first  );
	hydra_thrust::return_temporary_buffer( policy,  kernel_samples.first  );
	hydra_thrust::return_temporary_buffer( policy, functor_samples.first  );
}


}  // namespace hydra



#endif /* CONVOLUTION_H_ */
