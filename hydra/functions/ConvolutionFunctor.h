/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>
#include <hydra/detail/FFTPolicy.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <type_traits>


namespace hydra {

namespace detail {

	namespace convolution {

	    template<typename T, typename ArgType>
	    struct _traits;

	    template<typename Functor, typename Kernel,typename ArgType >
	  	struct _traits< hydra_thrust::tuple<Functor, Kernel>, ArgType>
	  	{
	      typedef typename std::common_type<
	    		  	  typename Functor::return_type,
	    		  	  typename Kernel::return_type
	    		  >::type return_type;

	      typedef typename detail::stripped_type<ArgType>::type value_type;

	      using signature = return_type(ArgType) ;

	  	};


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

/*
 * The convolution functor needs to combine four elements in order calculate the convolution and evaluate it for arbitrary points
 *
 */
template<typename Functor, typename Kernel, typename Backend, typename FFT, typename ArgType>
class ConvolutionFunctor;

template<typename Functor, typename Kernel, detail::Backend BACKEND, detail::FFTCalculator FFT,  typename ArgType>
class ConvolutionFunctor< Functor, Kernel, detail::BackendPolicy<BACKEND>,
                 detail::FFTPolicy<typename std::common_type<
                 	 typename Functor::return_type,
                 	 typename Kernel::return_type>::type, FFT>,
                 ArgType> :
   public BaseCompositeFunctor<
                 ConvolutionFunctor< Functor, Kernel, detail::BackendPolicy<BACKEND>,
                 	 detail::FFTPolicy< typename std::common_type<
                 	 	 typename Functor::return_type,
                 	 	 typename Kernel::return_type>::type, FFT>,
                 	 ArgType>, hydra_thrust::tuple<Functor, Kernel>,
                 	 typename  detail::convolution::_traits<hydra_thrust::tuple<Functor, Kernel>, ArgType>::signature
                >
{
	//typedef
	typedef typename detail::convolution::_traits<hydra_thrust::tuple<Functor, Kernel>, ArgType>::value_type   value_type;


	typedef ConvolutionFunctor< Functor, Kernel, detail::BackendPolicy<BACKEND>,
        	 detail::FFTPolicy< typename std::common_type<
        	 	 typename Functor::return_type,
        	 	 typename Kernel::return_type>::type, FFT>,
        	 ArgType>  this_type;

	typedef BaseCompositeFunctor<
            ConvolutionFunctor< Functor, Kernel, detail::BackendPolicy<BACKEND>,
            	 detail::FFTPolicy< typename std::common_type<
            	 	 typename Functor::return_type,
            	 	 typename Kernel::return_type>::type, FFT>,
            	 ArgType>, hydra_thrust::tuple<Functor, Kernel>,
            	 typename  detail::convolution::_traits<hydra_thrust::tuple<Functor, Kernel>, ArgType>::signature
           > super_type;



	typedef hydra::detail::FFTPolicy<value_type, FFT>    fft_type;

	//hydra backend
	typedef typename detail::BackendPolicy<BACKEND> device_system_type;
	typedef typename fft_type::host_backend_type      host_system_type;
	typedef typename fft_type::device_backend_type     fft_system_type;

	//raw thrust backend
	typedef typename std::remove_const<decltype(   std::declval<fft_system_type>().backend)>::type	raw_fft_system_type;
	typedef typename std::remove_const<decltype(std::declval<device_system_type>().backend)>::type  raw_device_system_type;
	typedef typename std::remove_const<decltype( std::declval< host_system_type>().backend)>::type  raw_host_system_type;

	//pointers
	typedef hydra_thrust::pointer<value_type, raw_host_system_type>      host_pointer_type;
	typedef hydra_thrust::pointer<value_type, raw_device_system_type>  device_pointer_type;
	typedef hydra_thrust::pointer<value_type, raw_fft_system_type>        fft_pointer_type;

	//iterator
	typedef hydra_thrust::transform_iterator< detail::convolution::_delta<value_type>,
	          hydra_thrust::counting_iterator<unsigned> > abiscissae_type;


public:

typedef typename detail::convolution::_traits<hydra_thrust::tuple<Functor, Kernel>, ArgType>::return_type return_t;

	ConvolutionFunctor() = delete;

	ConvolutionFunctor( Functor const& functor, Kernel const& kernel,
			value_type kmin, value_type kmax, unsigned nsamples=1024,
			bool interpolate=true, bool power_up=true):
		super_type(functor,kernel),
		fNSamples(power_up ? hydra::detail::convolution::upper_power_of_two(nsamples): nsamples),
		fMin(kmin),
		fMax(kmax),
	    fXMin(abiscissae_type{}),
		fXMax(abiscissae_type{}),
		fInterpolate(interpolate)
	{
		//std::cout << ">>ConvolutionFunctor()"<<std::endl;

		using hydra_thrust::get_temporary_buffer;

		fXMin = abiscissae_type(hydra_thrust::counting_iterator<unsigned>(0),
				        detail::convolution::_delta<value_type>(kmin, (kmax-kmin)/fNSamples) );
		fXMax = fXMin + fNSamples;

		fFFTData   = get_temporary_buffer<value_type>(raw_fft_system_type(), fNSamples).first;
		fHostData  = get_temporary_buffer<value_type>(raw_host_system_type(), fNSamples).first;

		fDeviceData= get_temporary_buffer<value_type>(raw_device_system_type(), fNSamples).first;


		Update();
		//std::cout << "<<ConvolutionFunctor()"<<std::endl;

	}

	__hydra_host__ __hydra_device__
	ConvolutionFunctor( this_type const& other):
	super_type(other),
	fNSamples(other.GetNSamples()),
	fMax(other.GetMax()),
	fMin(other.GetMin()),
	fXMax(other.GetXMax()),
	fXMin(other.GetXMin()),
	fInterpolate(other.IsInterpolated()),
	fDeviceData(other.GetDeviceData()),
	fHostData(other.GetHostData()),
	fFFTData(other.GetFFTData())
	{}

	__hydra_host__ __hydra_device__
	this_type& operator=(this_type const& other){

		if(this == &other) return *this;

		super_type::operator=(other);

		fNSamples  = other.GetNSamples();
		fMax 	  = other.GetMax();
		fMin 	  = other.GetMin();
		fXMax     = other.GetXMax();
		fXMin     = other.GetXMin();
		fInterpolate = other.IsInterpolated();
		fDeviceData  = other.GetDeviceData();
		fHostData    = other.GetHostData();
		fFFTData     = other.GetFFTData();

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

		//std::cout << ">>Update()"<<std::endl;

		auto data = make_range(fFFTData, fFFTData + fNSamples );

		hydra::convolute(fft_system_type(), fft_type(),
				hydra_thrust::get<0>(this->GetFunctors()),
				hydra_thrust::get<1>(this->GetFunctors()),
				fMin, fMax, data, false);

		sync_data<FFT>();

		//std::cout << "<< Update()"<<std::endl;

	}


	__hydra_host__ __hydra_device__
	inline bool IsInterpolated() const{
		return fInterpolate;
	}

	__hydra_host__ __hydra_device__
	inline void SetInterpolate(bool interpolate){
		fInterpolate = interpolate;
	}



     __hydra_host__ __hydra_device__
	inline return_t Evaluate(ArgType X) const	{


#ifdef __CUDA_ARCH__
		if( fInterpolate ) return spiline( fXMin, fXMax, fDeviceData, X);
		else{

			unsigned i = fNSamples*(X-fMin)/(fMax-fMin);
			return fDeviceData[i];
		}
#else
		if( fInterpolate ) return spiline( fXMin, fXMax, fHostData, X);
			else{

				unsigned i = fNSamples*(X-fMin)/(fMax-fMin);
				return fDeviceData[i];
			}
#endif

	}

void Dispose(){
		using hydra_thrust::return_temporary_buffer;

		return_temporary_buffer(  device_system_type(), fDeviceData );
		return_temporary_buffer(  host_system_type(),   fHostData );
		return_temporary_buffer(  fft_system_type()  , fFFTData );

	}

	virtual ~ConvolutionFunctor()=default;

	__hydra_host__ __hydra_device__
	std::size_t GetNSamples() const
	{
		return fNSamples;
	}

	__hydra_host__ __hydra_device__
	const device_pointer_type& GetDeviceData() const {
		return fDeviceData;
	}

	__hydra_host__ __hydra_device__
	const fft_pointer_type& GetFFTData() const {
		return fFFTData;
	}

	__hydra_host__ __hydra_device__
	const host_pointer_type& GetHostData() const {
		return fHostData;
	}
private:


	template<detail::FFTCalculator FFTC=FFT>
	inline typename std::enable_if<FFTC ==detail::CuFFT>::type
	sync_data()
	{
		hydra_thrust::copy_n( fFFTData, fNSamples, fDeviceData );
		hydra_thrust::copy_n( fFFTData, fNSamples, fHostData );
	}

	template<detail::FFTCalculator FFTC=FFT>
	inline typename std::enable_if<FFTC !=detail::CuFFT>::type
	sync_data()
	{
		hydra_thrust::copy_n(device_system_type(), fFFTData, fNSamples, fDeviceData );
		hydra_thrust::copy_n(device_system_type(), fFFTData, fNSamples, fHostData );
	}

	std::size_t          fNSamples;
	abiscissae_type fXMin;
	abiscissae_type fXMax;
    value_type      fMax;
    value_type      fMin;
    bool            fInterpolate;
    device_pointer_type fDeviceData;
    host_pointer_type   fHostData;
    fft_pointer_type    fFFTData ;


};

template<typename ArgType,  typename Functor, typename Kernel, detail::Backend BACKEND, detail::FFTCalculator FFT,
               typename T=typename detail::stripped_type<typename std::common_type<typename Functor::return_type, typename Kernel::return_type>::type>::type>
inline typename std::enable_if< std::is_floating_point<T>::value, ConvolutionFunctor<Functor, Kernel,
                 detail::BackendPolicy<BACKEND>, detail::FFTPolicy<T, FFT>, ArgType>>::type
make_convolution( detail::BackendPolicy<BACKEND> const&, detail::FFTPolicy<T, FFT> const&, Functor const& functor, Kernel const& kernel,
		T kmin, T kmax, unsigned nsamples=1024,	bool interpolate=true, bool power_up=true)
{
	return ConvolutionFunctor<Functor, Kernel,
			detail::BackendPolicy<BACKEND>, detail::FFTPolicy<T, FFT>, ArgType>(functor, kernel, kmin,  kmax, nsamples);

}


}  // namespace hydra


#endif /* CONVOLUTIONFUNCTOR_H_ */
