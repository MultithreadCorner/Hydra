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
#include <hydra/Spiline.h>
#include <hydra/Convolution.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/FFTPolicy.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <type_traits>


namespace hydra {

namespace detail {

	namespace convolution {

		template<typename T>
		struct _delta
		{
			_delta(T min,  T delta):
				fMin(min),
				fDelta(delta)
			{}

			__hydra_host__ __hydra_device__
			_delta( _delta<T> const& other):
			fMin(other.fMin),
			fDelta(other.fDelta)
			{}

			inline T operator()(unsigned bin){
				return fMin + bin*fDelta;
			}

		private:
			T fMin;
			T fDelta;
		};

	}  // namespace convolution

}  // namespace detail

template<typename Functor, typename Kernel, typename FFT, unsigned int ArgIndex=0>
class ConvolutionFunctor;

template<typename Functor, typename Kernel, detail::FFTCalculator  FFT, unsigned int ArgIndex>
class ConvolutionFunctor<Functor, Kernel,
   detail::FFTPolicy<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, FFT>, ArgIndex>:
   public BaseCompositeFunctor< ConvolutionFunctor<Functor, Kernel,
   detail::FFTPolicy<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, FFT>, ArgIndex>,
        typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, Functor, Kernel>
{
	//typedef
	typedef BaseCompositeFunctor< ConvolutionFunctor<Functor, Kernel,
	           detail::FFTPolicy<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, FFT>, Iterator, ArgIndex>,
	           typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, Functor, Kernel> super_type;

	typedef hydra::detail::FFTPolicy<typename std::common_type<
			typename Functor::return_type, typename Kernel::return_type>::type, FFT>    fft_type;

	typedef typename fft_type::backend_type system_type;

	typedef HYDRA_EXTERNAL_NS::thrust::pointer<T, typename  system_type>  pointer_type;

	//aliases
	using value_type  = typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type;
	using return_type = typename super_type::return_type;

	typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator< _delta<value_type>,
	   HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> > abiscissae_type;

public:

	Convolution() = delete;

	Convolution( Functor const& functor, Kernel const& kernel,Iterator first, Iterator last, double kmin, double kmax, size_t nsamples, bool power_up=true):
		super_type(functor,kernel),
		fMin(kmin),
		fMax(kmax)
	{
		using HYDRA_EXTERNAL_NS::thrust::counting_iterator;
		using HYDRA_EXTERNAL_NS::thrust::transform_iterator;

		fNSamples = power_up ? hydra::detail::convolution::upper_power_of_two(nsamples): nsamples ;

		auto shift = _delta<T>(kmin, (kmax-kmin)/fNSamples);

		fStorage=HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(system_type(), fNSamples);

		fXMin =  abiscissae_type(counting_iterator<unsigned>(0), shift);
		fXMax =  abiscissae_type(counting_iterator<unsigned>(fNSamples), shift);
	}

	__hydra_host__ __hydra_device__
	Convolution( ConvolutionFunctor<Functor, Kernel, fft_type, ArgIndex> const& other):
	super_type(other),
	fMax(other.GetMax()),
	fMin(other.GetMin()),
	fXMax(other.GetXMax()),
	fXMin(other.GetXMin()),
	fNSamples(other.GetNSamples()),
	fStorage(other.GetStorage())
	{}

	__hydra_host__ __hydra_device__
	ConvolutionFunctor<Functor,Kernel, fft_type,  ArgIndex>&
	operator=(ConvolutionFunctor<Functor,Kernel, fft_type,  ArgIndex> const& other){

		if(this == &other) return *this;

		super_type::operator=(other);

		fMax 	  = other.GetMax();
		fMin 	  = other.GetMin();
		fXMax     = other.GetXMax();
		fXMin     = other.GetXMin();
		fNSamples = other.GetNSamples();
		fStorage  = other.GetStorage();

		return *this;
	}

	__hydra_host__ __hydra_device__
	double GetMax() const {
		return fMax;
	}

	__hydra_host__ __hydra_device__
	double GetMin() const {
		return fMin;
	}

	__hydra_host__ __hydra_device__
	size_t GetNSamples() const {
		return fNSamples;
	}

	__hydra_host__ __hydra_device__
	const abiscissae_type& GetXMax() const
	{
		return fXMax;
	}

	__hydra_host__ __hydra_device__
	const abiscissae_type& GetXMin() const
	{
		return fXMin;
	}

	__hydra_host__ __hydra_device__
	const pointer_type& GetStorage() const
	{
		return fStorage;
	}

	virtual void Update() override
	{
		auto result = make_range(fStorage.get(), fStorage.get() + fNSamples );

		convolute(system_type(), fft_type(),
				HYDRA_EXTERNAL_NS::get<0>(GetFunctors()),
				HYDRA_EXTERNAL_NS::get<1>(GetFunctors()),
				fMin, fMax, result,0);
	}

	void Dispose(){

		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_type(),fStorage);
	}

	template<typename T>
     __hydra_host__ __hydra_device__
	inline return_type Evaluate(unsigned int n, T*x) const	{

		T X  = x[ArgIndex];

		return spiline( fXMin, fXMax, fStorage.get() , X);

	}


	template<typename T>
     __hydra_host__ __hydra_device__
	inline double Evaluate(T& x) const {

		auto X  = get<ArgIndex>(x);

		return spiline( fXMin, fXMax, fStorage.get() , X);
	}

	virtual ~Convolution(){};

private:

	abiscissae_type fXMin;
	abiscissae_type fXMax;
	pointer_type fStorage; // non raii
    size_t  fNSamples;
	GReal_t fMax;
	GReal_t fMin;

};

template<unsigned int ArgIndex,  typename Functor, typename Kernel,  detail::FFTCalculator FFTBackend,
typename T=typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type>
inline typename std::enable_if< std::is_floating_point<T>::value, ConvolutionFunctor<Functor, Kernel, detail::FFTPolicy<T, FFT>, ArgIndex>>::type
make_convolution( detail::FFTPolicy<T, FFT>, Functor const& functor, Kernel const& kernel, T kmin, T kmax, unsigned nsamples)
{

	return ConvolutionFunctor<Functor, Kernel, detail::BackendPolicy<BACKEND> , N, ArgIndex>( functor, kernel,kmin,kmax);

}


}  // namespace hydra


#endif /* CONVOLUTION_H_ */
