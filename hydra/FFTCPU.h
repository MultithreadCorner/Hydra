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
 * FFTCPU.h
 *
 *  Created on: 13/11/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FFTCPU_H_
#define FFTCPU_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/Range.h>
#include <hydra/Tuple.h>

#include <cassert>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <complex.h>

//FFTW3
#include <fftw3.h>


namespace hydra {

	namespace detail {

		namespace fftw {

			template<typename T>
			struct DeleterFFT {
				inline void operator()(T* ptr){ fftw_free(ptr);}
			};

			template<typename T>
			struct RealToComplexFFTPlanner;

			template<> struct RealToComplexFFTPlanner<double>
			{
				inline fftw_plan operator()(int n, double *in, fftw_complex *out, unsigned flags ){

					return fftw_plan_dft_r2c_1d(n, in, out, flags);
				}
			};

			template<> struct RealToComplexFFTPlanner<float>
			{
				inline fftwf_plan operator()(int n, float *in, fftwf_complex *out, unsigned flags ){

					return fftwf_plan_dft_r2c_1d(n, in, out, flags);
				}
			};

			template<typename T>
			struct ComplexToRealFFTPlanner;

			template<> struct ComplexToRealFFTPlanner<double>
			{
				inline fftw_plan operator()(int n, fftw_complex *in, double *out,  unsigned flags ){

					return fftw_plan_dft_c2r_1d(n, in, out, flags);
				}
			};

			template<> struct ComplexToRealFFTPlanner<float>
			{
				inline fftwf_plan operator()(int n, fftwf_complex *in,float *out,  unsigned flags ){

					return fftwf_plan_dft_c2r_1d(n, in, out, flags);
				}
			};



		}  // namespace fftw

	}  // namespace detail



template<typename T,
   typename CType= typename std::conditional< std::is_same<double,T>::value, fftw_complex, fftwf_complex>::type,
   typename PType= typename std::conditional< std::is_same<double,T>::value, fftw_plan, fftwf_plan>::type,
   bool SinglePrecision=std::is_same<double,T>::value>
class RealToComplexFFT
{
	typedef CType complex_type;
	typedef T real_type;
	typedef PType plan_type;


public:

	RealToComplexFFT()=delete;

	RealToComplexFFT(size_t nsamples):
		fNsamples(nsamples),
		fComplexPtr(reinterpret_cast<complex_type*>(fftw_malloc(sizeof(complex_type)*(fNsamples/2 +1  )))),
		fRealPtr(reinterpret_cast<real_type*>(fftw_malloc(sizeof(real_type)*fNsamples)))
	{
		detail::fftw::RealToComplexFFTPlanner<T> planner;

		fPlan =  planner( fNsamples,  fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE);

		if(fPlan==NULL){

			throw std::runtime_error("hydra::RealToComplexFFT : can not allocate fftw_plan");
		}
	}

	RealToComplexFFT(RealToComplexFFT<T>&& other):
		fNsamples(other.GetNsamples()),
		fComplexPtr(std::move(fComplexPtr)),
		fRealPtr(std::move(fRealPtr))
	{
		fftw_destroy_plan(fPlan);
		fPlan =  fftw_plan_dft_r2c_1d( fNsamples,
				fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE);
	}

	RealToComplexFFT<T>& operator=(RealToComplexFFT<T>&& other)
	{
		if(this ==&other) return *this;

		fNsamples   = other.GetNsamples();
		fComplexPtr = std::move(fComplexPtr);
		fRealPtr    = std::move(fRealPtr);

		fftw_destroy_plan(fPlan);

		detail::fftw::RealToComplexFFTPlanner<T> planner;

		fPlan =  planner( fNsamples, fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE);

		 return *this;
	}

	template<typename Iterable, typename Type =	typename decltype(*std::declval<Iterable&>().begin())::value_type>
	inline typename std::enable_if<std::is_same<real_type, Type>::value
	    && detail::is_iterable<Iterable>::value, void>::type
	LoadInputData( Iterable&& container){

		LoadInput(std::forward<Iterable>(container).size(),
				HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(std::forward<Iterable>(container).data()));
	}

	inline void	LoadInputData(size_t n, const real_type* data){

		LoadInput(n,data);
	}

	inline void Execute(){	fftw_execute(fPlan); }

	inline hydra::pair<complex_type*, size_t>
	GetTransformedData(){
		return hydra::make_pair(&fComplexPtr.get()[0], (fNsamples/2 +1) );
	}


	inline size_t GetNsamples() const { return fNsamples; }

	~RealToComplexFFT(){ fftw_destroy_plan(fPlan); }

private:

	inline std::unique_ptr<complex_type, detail::fftw::DeleterFFT<complex_type> >
	GetComplexPtr() { return std::move(fComplexPtr); }

	inline std::unique_ptr<real_type, detail::fftw::DeleterFFT<real_type> >
	GetRealPtr() { return std::move(fRealPtr); }


	void LoadInput(size_t n, const real_type* data ){

		assert(n <= fNsamples);
		memcpy(fRealPtr.get(), data, sizeof(real_type)*n);
		memset(&fRealPtr.get()[n], 0, sizeof(real_type)*(fNsamples - n));
	}

	size_t fNsamples;
	plan_type fPlan;
	std::unique_ptr<complex_type, detail::fftw::DeleterFFT<complex_type>> fComplexPtr;
	std::unique_ptr<real_type   , detail::fftw::DeleterFFT<real_type>>       fRealPtr;
};


template<typename T,
		typename CType= typename std::conditional< std::is_same<double,T>::value, fftw_complex, fftwf_complex>::type,
		typename PType= typename std::conditional< std::is_same<double,T>::value, fftw_plan, fftwf_plan>::type,
		bool SinglePrecision=std::is_same<double,T>::value >
class ComplexToRealFFT
{
	typedef CType complex_type;
	typedef T real_type;
	typedef PType plan_type;

public:

	ComplexToRealFFT()=delete;

	ComplexToRealFFT(size_t nsamples):
		fNsamples(nsamples),
		fComplexPtr(reinterpret_cast<complex_type*>(fftw_malloc((fNsamples)))),
		fRealPtr(reinterpret_cast<real_type*>(fftw_alloc_real(fNsamples*2 - 1)))
	{
		detail::fftw::ComplexToRealFFTPlanner<T> planner;

		fPlan= planner( nsamples, fComplexPtr.get(), fRealPtr.get(), FFTW_ESTIMATE);

		if(fPlan==NULL){

			throw std::runtime_error("hydra::RealToComplexFFT : can not allocate fftw_plan");
		}
	}

	ComplexToRealFFT(ComplexToRealFFT<T>&& other):
		fNsamples(other.GetNsamples()),
		fComplexPtr(std::move(fComplexPtr)),
		fRealPtr(std::move(fRealPtr))
	{
		fftw_destroy_plan(fPlan);
		fPlan =  fftw_plan_dft_c2r_1d( fNsamples,
				    fComplexPtr.get(), fRealPtr.get(), FFTW_ESTIMATE);
	}

	ComplexToRealFFT<T>& operator=(ComplexToRealFFT<T>&& other)
	{
		if(this ==&other) return *this;

		fNsamples   = other.GetNsamples();
		fComplexPtr = std::move(fComplexPtr);
		fRealPtr    = std::move(fRealPtr);

		fftw_destroy_plan(fPlan);

		detail::fftw::ComplexToRealFFTPlanner<T> planner;
		fPlan =  planner( fNsamples,
				 fComplexPtr.get(),fRealPtr.get(), FFTW_ESTIMATE);

		 return *this;
	}

	template<typename Iterable, typename Type =	typename decltype(*std::declval<Iterable&>().begin())::value_type>
	inline typename std::enable_if<std::is_convertible<fftw_complex*, Type*>::value
		    && detail::is_iterable<Iterable>::value, void>::type
	LoadInputData( Iterable&& container){

		LoadInput(std::forward<Iterable>(container).size(),
				reinterpret_cast<fftw_complex*>(std::forward<Iterable>(container).data()));
	}

	inline void	LoadInputData(size_t n, const complex_type* data){

		LoadInput(n,data);
	}


	inline void Execute(){ fftw_execute(fPlan); }

	inline hydra::pair<real_type*, size_t>
	GetTransformedData(){
		return hydra::make_pair(&fRealPtr.get()[0], (fNsamples*2 - 1) );
	}

	inline size_t GetNsamples() const { return fNsamples; }

	~ComplexToRealFFT(){ fftw_destroy_plan(fPlan); }

private:

	inline std::unique_ptr<complex_type, detail::fftw::DeleterFFT<complex_type> >
	GetComplexPtr() { return std::move(fComplexPtr); }

	inline std::unique_ptr<real_type, detail::fftw::DeleterFFT<real_type> >
	GetRealPtr() { return std::move(fRealPtr); }

	void LoadInput(size_t n, const complex_type* data ){

		assert(n <= fNsamples);
		memcpy(fComplexPtr.get(), data, sizeof(complex_type)*n);
		memset(&fComplexPtr.get()[n], 0, sizeof(complex_type)*(fNsamples - n));
	}

	size_t fNsamples;
	plan_type fPlan;
	std::unique_ptr<complex_type, detail::fftw::DeleterFFT<complex_type>> fComplexPtr;
	std::unique_ptr<real_type   , detail::fftw::DeleterFFT<real_type>>       fRealPtr;

};

}  // namespace hydra




#endif /* FFTCPU_H_ */
