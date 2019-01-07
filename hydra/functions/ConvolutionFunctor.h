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

#ifndef CONVOLUTIONFUNCTOR_H_
#define CONVOLUTIONFUNCTOR_H_

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
			_delta()=default;


			_delta(T min,  T delta):
				fMin(min),
				fDelta(delta)
			{}

			__hydra_host__ __hydra_device__
			_delta( _delta<T> const& other):
			fMin(other.fMin),
			fDelta(other.fDelta)
			{}

			__hydra_host__ __hydra_device__
			inline T operator()(unsigned bin){
				return fMin + bin*fDelta;
			}


			T fMin;
			T fDelta;
		};

	}  // namespace convolution

}  // namespace detail

template<typename Functor, typename Kernel, typename Backend, unsigned int ArgIndex=0>
class ConvolutionFunctor;

template<typename Functor, typename Kernel, detail::Backend BACKEND, unsigned int ArgIndex>
class ConvolutionFunctor<Functor, Kernel, detail::BackendPolicy<BACKEND>, ArgIndex>:
   public BaseCompositeFunctor< ConvolutionFunctor<Functor, Kernel, detail::BackendPolicy<BACKEND>,
        detail::FFTPolicy<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, FFT>, NSamples, ArgIndex>,
        typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, Functor, Kernel>
{
	//typedef
	typedef BaseCompositeFunctor< ConvolutionFunctor<Functor, Kernel, detail::BackendPolicy<BACKEND>,
	           detail::FFTPolicy<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, FFT>, NSamples, ArgIndex>,
	           typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type, Functor, Kernel> super_type;

	typedef hydra::detail::FFTPolicy<typename std::common_type<
			typename Functor::return_type, typename Kernel::return_type>::type, FFT>    fft_type;

	//hydra backend
	typedef typename detail::BackendPolicy<BACKEND> device_system_type;
	typedef typename fft_type::host_backend_type      host_system_type;

	//thrust backend
	typedef typename std::remove_const<decltype(std::declval<device_system_type>().backend)>::type
			super_device_system_type;

	typedef typename std::remove_const<decltype(std::declval<  host_system_type>().backend)>::type
			super_host_system_type;

	//aliases
	using value_type  = typename std::common_type<typename Functor::return_type,
			typename Kernel::return_type>::type;

	using return_type = typename super_type::return_type;

	//pointers
	typedef HYDRA_EXTERNAL_NS::thrust::pointer<value_type, super_host_system_type>    host_pointer_type;
	typedef HYDRA_EXTERNAL_NS::thrust::pointer<value_type, super_device_system_type>  device_pointer_type;

	typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator< detail::convolution::_delta<value_type>,
	          HYDRA_EXTERNAL_NS::thrust::counting_iterator<unsigned> > abiscissae_type;


public:

	ConvolutionFunctor() = delete;

	ConvolutionFunctor( Functor const& functor, Kernel const& kernel,
			value_type kmin, value_type kmax, unsigned nsamples=1024,
			bool interpolate=1, bool power_up=true):
		super_type(functor,kernel),
		fNSamples(0),
		fMin(kmin),
		fMax(kmax),
	    fXMin(abiscissae_type{}),
		fXMax(abiscissae_type{}),
		fInterpolate(interpolate)
	{
		fNSamples = power_up ? hydra::detail::convolution::upper_power_of_two(nsamples): nsamples ;


		fXMin = abiscissae_type(HYDRA_EXTERNAL_NS::thrust::counting_iterator<unsigned>(0),
				detail::convolution::_delta<value_type>(kmin, (kmax-kmin)/NSamples) );
		fXMax = fXMin+NSamples;

		fHostStorage=HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<value_type>(
				super_host_system_type(), fNSamples).first;

		fDeviceStorage=HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<value_type>(
				super_device_system_type(), fNSamples).first;


		Update();
	}

	__hydra_host__ __hydra_device__
	ConvolutionFunctor( ConvolutionFunctor<Functor, Kernel, fft_type, NSamples, ArgIndex> const& other):
	super_type(other),
	fMax(other.GetMax()),
	fMin(other.GetMin()),
	fXMax(other.GetXMax()),
	fXMin(other.GetXMin()),
	fInterpolate(other.IsInterpolated())
	{
		for(size_t i=0; i<NSamples; i++)
			fConvRep[i]=other.GetConvRep()[i];
	}

	__hydra_host__ __hydra_device__
	ConvolutionFunctor<Functor,Kernel, fft_type, NSamples, ArgIndex>&
	operator=(ConvolutionFunctor<Functor,Kernel, fft_type, NSamples, ArgIndex> const& other){

		if(this == &other) return *this;

		super_type::operator=(other);

		fMax 	  = other.GetMax();
		fMin 	  = other.GetMin();
		fXMax     = other.GetXMax();
		fXMin     = other.GetXMin();
		fInterpolate = other.IsInterpolated();
		for(size_t i=0; i<NSamples; i++)
					fConvRep[i]=other.GetConvRep()[i];

		return *this;
	}

	__hydra_host__ __hydra_device__
	value_type GetMax() const {
		return fMax;
	}

	__hydra_host__ __hydra_device__
	value_type GetMin() const {
		return fMin;
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



	virtual void Update() override
	{
		auto scratch = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<value_type>(device_system_type(), NSamples).first;

		auto data = make_range(scratch, scratch + NSamples );

		hydra::convolute(device_system_type(), fft_type(),
				HYDRA_EXTERNAL_NS::thrust::get<0>(this->GetFunctors()),
				HYDRA_EXTERNAL_NS::thrust::get<1>(this->GetFunctors()),
				fMin, fMax, data, false);

		HYDRA_EXTERNAL_NS::thrust::copy(device_system_type(), scratch, scratch + NSamples, fConvRep );

		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(device_system_type(), scratch);
	}

	__hydra_host__ __hydra_device__
	const value_type* GetConvRep() const
	{
		return fConvRep;
	}

	__hydra_host__ __hydra_device__
	inline bool IsInterpolated() const{
		return fInterpolate;
	}

	__hydra_host__ __hydra_device__
	inline void SetInterpolate(bool interpolate){
		fInterpolate = interpolate;
	}


	template<typename T>
     __hydra_host__ __hydra_device__
	inline return_type Evaluate(unsigned int n, T*x) const	{

		T X = x[ArgIndex];

		if( fInterpolate ) return spiline( fXMin, fXMax, fConvRep , X);
		else{


			unsigned i = NSamples*(X-fMin)/(fMax-fMin);
			return fConvRep[i];
		}


	}


	template<typename T>
     __hydra_host__ __hydra_device__
     inline double Evaluate(T& x) const {

		auto X  = get<ArgIndex>(x);

		if( fInterpolate ) return spiline( fXMin, fXMax, fConvRep , X);
		else{


			unsigned i = NSamples*(X-fMin)/(fMax-fMin);
			return fConvRep[i];
		}

	}



	virtual ~ConvolutionFunctor()=default;



private:

	abiscissae_type fXMin;
	abiscissae_type fXMax;
	value_type fConvRep[NSamples]; // non raii
    value_type fMax;
    value_type fMin;
    bool       fInterpolate;

};

template<unsigned int NSamples, unsigned int ArgIndex,  typename Functor, typename Kernel,  detail::FFTCalculator FFTBackend,
typename T=typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type>
inline typename std::enable_if< std::is_floating_point<T>::value, 
     ConvolutionFunctor<Functor, Kernel, detail::FFTPolicy<T, FFTBackend>, NSamples,ArgIndex>>::type
make_convolution( detail::FFTPolicy<T, FFTBackend> policy, Functor const& functor, Kernel const& kernel, T kmin, T kmax)
{

	return ConvolutionFunctor<Functor, Kernel, detail::FFTPolicy<T, FFTBackend>, NSamples, ArgIndex>( functor, kernel,kmin,kmax);

}


}  // namespace hydra


#endif /* CONVOLUTIONFUNCTOR_H_ */
