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
			struct _Free;

			template<> struct _Free<double> {

				template<typename PointerType>
				inline void operator()(PointerType* ptr){ fftw_free(ptr);}
			};

			template<> struct _Free<float> {

				template<typename PointerType>
				inline void operator()(PointerType* ptr){ fftwf_free(ptr);}
			};



			/* wrapping plan creator functions
			 * parameterized by float precision
			 */
			template<typename T>
			struct _PlanRealToComplex;

			template<> struct _PlanRealToComplex<double>
			{
				inline fftw_plan operator()(int n, double *in, fftw_complex *out, unsigned flags ){

					return fftw_plan_dft_r2c_1d(n, in, out, flags);
				}
			};

			template<> struct _PlanRealToComplex<float>
			{
				inline fftwf_plan operator()(int n, float *in, fftwf_complex *out, unsigned flags ){

					return fftwf_plan_dft_r2c_1d(n, in, out, flags);
				}
			};

			template<typename T>
			struct _PlanComplexToReal;

			template<> struct _PlanComplexToReal<double>
			{
				inline fftw_plan operator()(int n, fftw_complex *in, double *out,  unsigned flags ){

					return fftw_plan_dft_c2r_1d(n, in, out, flags);
				}
			};

			template<> struct _PlanComplexToReal<float>
			{
				inline fftwf_plan operator()(int n, fftwf_complex *in,float *out,  unsigned flags ){

					return fftwf_plan_dft_c2r_1d(n, in, out, flags);
				}
			};

			/* wrapping plan destroy functions
			 * parameterized by float precision
			 */
			template<typename T>
			struct _PlanDestroy;

			template<>	struct _PlanDestroy<double> {
				inline void operator()(fftw_plan plan ){

					fftw_destroy_plan(plan);
				}
			};

			template<>	struct _PlanDestroy<float> {
				inline void operator()(fftwf_plan plan ){

					fftwf_destroy_plan(plan);
				}
			};


			/* wrapping plan execution functions
			 * parameterized by float precision
			 */
			template<typename T>
			struct _Execute;

			template<>	struct _Execute<double> {
				inline void operator()(fftw_plan& plan ){

					fftw_execute(plan);
				}
			};

			template<>	struct _Execute<float> {
				inline void operator()(fftwf_plan& plan ){

					fftwf_execute(plan);
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

	RealToComplexFFT(int  logical_size):
		fNReal( logical_size ),
		fNComplex( logical_size/2+1 ),
		fComplexPtr(reinterpret_cast<complex_type*>(fftw_malloc(sizeof(complex_type)*(logical_size/2 +1 )))),
		fRealPtr(reinterpret_cast<real_type*>(fftw_malloc(sizeof(real_type)*logical_size)))
	{
		detail::fftw::_PlanRealToComplex<T> planner;

		fPlan =  planner( logical_size,  fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE);

		if(fPlan==NULL){

			throw std::runtime_error("hydra::RealToComplexFFT : can not allocate fftw_plan");
		}
	}

	RealToComplexFFT(RealToComplexFFT<T>&& other):
		fNReal(other.GetNReal()),
		fNComplex(other.GetNComplex()),
		fComplexPtr(std::move(fComplexPtr)),
		fRealPtr(std::move(fRealPtr))
	{
		detail::fftw::_PlanDestroy<real_type>(fPlan);
		detail::fftw::_PlanRealToComplex<T> planner;
		fPlan(planner( other.GetSize() ,fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE));
	}

	RealToComplexFFT<T>& operator=(RealToComplexFFT<T>&& other)
	{
		if(this ==&other) return *this;

		fNReal      = other.GetNReal();
		fNComplex   = other.GetNComplex();
		fComplexPtr = std::move(fComplexPtr);
		fRealPtr    = std::move(fRealPtr);

		detail::fftw::_PlanRealToComplex<T> planner;

		fPlan =  planner( other.GetSize(), fRealPtr.get(), fComplexPtr.get(), FFTW_ESTIMATE);

		 return *this;
	}

	template<typename Iterable, typename Type =	typename decltype(*std::declval<Iterable&>().begin())::value_type>
	inline typename std::enable_if<std::is_same<real_type, Type>::value
	    && detail::is_iterable<Iterable>::value, void>::type
	LoadInputData( Iterable&& container){

		LoadInput(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(std::forward<Iterable>(container).data()));
	}

	inline void	LoadInputData( const real_type* data){

		LoadInput(data);
	}

	inline void Execute(){	detail::fftw::_Execute<real_type>()(fPlan); }

	inline hydra::pair<complex_type*, int>
	GetTransformedData(){
		return hydra::make_pair(&fComplexPtr.get()[0], fNComplex );
	}

	inline int GetSize() const { return fNReal; }

	inline int GetNComplex() const {	return fNComplex; }

	inline int GetNReal() const	{ return fNReal; }


	~RealToComplexFFT(){ detail::fftw::_PlanDestroy<real_type>()(fPlan); }

private:

	inline std::unique_ptr<complex_type, detail::fftw::_Free<real_type> >
	GetComplexPtr() { return std::move(fComplexPtr); }

	inline std::unique_ptr<real_type, detail::fftw::_Free<real_type> >
	GetRealPtr() { return std::move(fRealPtr); }


	void LoadInput(const real_type* data ){

		memcpy(&fRealPtr.get()[0], data, sizeof(real_type)*fNReal);
	}

	int fNReal;
	int fNComplex;
	plan_type fPlan;
	std::unique_ptr<complex_type, detail::fftw::_Free<real_type>> fComplexPtr;
	std::unique_ptr<real_type   , detail::fftw::_Free<real_type>> fRealPtr;
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

	ComplexToRealFFT(int logical_size):
		fNReal( logical_size ),
		fNComplex( logical_size/2 +1 ),
		fComplexPtr(reinterpret_cast<complex_type*>(fftw_malloc(sizeof(complex_type)*(logical_size/2+1)))),
		fRealPtr(reinterpret_cast<real_type*>(fftw_malloc( sizeof(real_type)*(logical_size))))
	{

		detail::fftw::_PlanComplexToReal<T> planner;

		fPlan= planner(logical_size , fComplexPtr.get(), fRealPtr.get(), FFTW_ESTIMATE);

		if(fPlan==NULL){

			throw std::runtime_error("hydra::ComplexToRealFFT : can not allocate fftw_plan");
		}
	}

	ComplexToRealFFT(ComplexToRealFFT<T>&& other):
		fNReal(other.GetNReal()),
		fNComplex(other.GetNComplex()),
		fComplexPtr(std::move(fComplexPtr)),
		fRealPtr(std::move(fRealPtr))
	{

		detail::fftw::_PlanComplexToReal<T> planner{};
		fPlan=  planner(other.GetSize() , fComplexPtr.get(), fRealPtr.get(), FFTW_ESTIMATE);
	}

	ComplexToRealFFT<T>& operator=(ComplexToRealFFT<T>&& other)
	{
		if(this ==&other) return *this;

		fNReal      = other.GetNReal();
		fNComplex   = other.GetNComplex();
		fComplexPtr = std::move(fComplexPtr);
		fRealPtr    = std::move(fRealPtr);

		detail::fftw::_PlanDestroy<real_type>(fPlan);
		detail::fftw::_PlanComplexToReal<T> planner;
		fPlan =  planner( other.GetSize() , fComplexPtr.get(), fRealPtr.get(), FFTW_ESTIMATE);

		 return *this;
	}

	template<typename Iterable, typename Type =	typename decltype(*std::declval<Iterable&>().begin())::value_type>
	inline typename std::enable_if<std::is_convertible<complex_type*, Type*>::value
		    && detail::is_iterable<Iterable>::value, void>::type
	LoadInputData( Iterable&& container){

		LoadInput(reinterpret_cast<complex_type*>(HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast(std::forward<Iterable>(container).data())));
	}

	inline void	LoadInputData(const complex_type* data) {

		LoadInput(data);
	}


	inline void Execute(){
		detail::fftw::_Execute<T>(fPlan);

		//for(int i=0; i<fNReal;i++)
    	  // std::cout << fRealPtr.get()[i] << std::endl;

		for(int i=0; i<fNComplex;i++)
				    	  std::cout << fComplexPtr.get()[i][0]<< " "<<fComplexPtr.get()[i][1] << std::endl;
	}

	inline hydra::pair<real_type*, int>
	GetTransformedData(){
		return hydra::make_pair(fRealPtr.get(),fNReal  );
	}

	inline int GetSize() const { return fNReal; }

	inline int GetNComplex() const {	return fNComplex; }

	inline int GetNReal() const	{ return fNReal; }

	~ComplexToRealFFT(){ detail::fftw::_PlanDestroy<real_type>(fPlan); }



private:

	inline std::unique_ptr<complex_type,  detail::fftw::_Free<real_type> >
	GetComplexPtr() { return std::move(fComplexPtr); }

	inline std::unique_ptr<real_type,  detail::fftw::_Free<real_type> >
	GetRealPtr() { return std::move(fRealPtr); }

	void LoadInput(const complex_type* data ){

		memcpy( fComplexPtr.get(),data,  sizeof(complex_type)*fNComplex);
		for(int i=0; i<fNComplex;i++)
		    	  std::cout << fComplexPtr.get()[i][0]<< " "<<fComplexPtr.get()[i][1] << std::endl;
	}


	int fNReal;
	int fNComplex;
	plan_type fPlan;
	std::unique_ptr<complex_type, detail::fftw::_Free<real_type>> fComplexPtr;
	std::unique_ptr<real_type   , detail::fftw::_Free<real_type>> fRealPtr;



};

}  // namespace hydra




#endif /* FFTCPU_H_ */
