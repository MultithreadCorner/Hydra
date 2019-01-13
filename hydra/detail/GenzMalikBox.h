/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * GenzMalikBox.h
 *
 *  Created on: 06/03/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENZMALIKBOX_H_
#define GENZMALIKBOX_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>
#include <array>
#include <initializer_list>
#include <math.h>
#include <cmath>




namespace hydra {

namespace detail {

template<size_t N>
class GenzMalikBox;


struct AddResultGenzMalikBoxes
{
	__hydra_host__ __hydra_device__
	hydra::pair<double, double> operator()( hydra::pair<double, double> const& r1,  hydra::pair<double, double> const& r2 ){
		return hydra::make_pair(r1.first + r2.first, r1.second + r2.second);
	}
};


template<size_t N>
struct CompareGenzMalikBoxes
{
	__hydra_host__ __hydra_device__
	bool operator()( detail::GenzMalikBox<N> const& box1,  detail::GenzMalikBox<N> const& box2 ){
		return box1.GetError() < box2.GetError();
	}
};

template<size_t N>
class GenzMalikBox
{

	typedef typename hydra::detail::tuple_type<N+2,double>::type result_type;

public:

	GenzMalikBox()=default;

	__hydra_host__ __hydra_device__
	GenzMalikBox(GReal_t (&LowerLimit)[N], GReal_t (&UpperLimit)[N]):
		fRule7(0),
		fRule5(0),
		fIntegral(0),
		fError(0),
		fErrorSq(0),
		fVolume(1.0),
		fCutAxis(-1)
	{
		for(size_t i=0; i<N; i++) {

		    fUpperLimit[i]=UpperLimit[i];
			fLowerLimit[i]=LowerLimit[i];
			fVolume*=(UpperLimit[i]-LowerLimit[i]);
		}

	}

	GenzMalikBox(std::array<GReal_t,N> const& LowerLimit, std::array<GReal_t,N> const&  UpperLimit):
		fRule7(0),
		fRule5(0),
		fIntegral(0),
		fError(0),
		fErrorSq(0),
		fVolume(1.0),
		fCutAxis(-1)
	{
		for(size_t i=0; i<N; i++)
		{
			fUpperLimit[i]=UpperLimit[i];
			fLowerLimit[i]=LowerLimit[i];
			fVolume *=(UpperLimit[i]-LowerLimit[i]);
		}


	}

	__hydra_host__ __hydra_device__
	GenzMalikBox(GenzMalikBox<N> const& other):
		fRule7(other.GetRule7() ),
		fRule5(other.GetRule5() ),
		fVolume(other.GetVolume() ),
		fIntegral(other.GetIntegral()),
		fError(other.GetError()),
		fErrorSq(other.GetErrorSq()),
		fCutAxis(other.GetCutAxis())

	{
		for(size_t i=0; i<N; i++)
		{
			fUpperLimit[i]=other.GetUpperLimit(i);
			fLowerLimit[i]=other.GetLowerLimit(i);

		}
	}

	__hydra_host__ __hydra_device__
	GenzMalikBox<N>& operator=(GenzMalikBox<N> const& other)
	{
		if(this==&other) return *this;

		this->fRule7 = other.GetRule7() ;
		this->fRule5 = other.GetRule5() ;
		this->fVolume = other.GetVolume() ;
		this->fIntegral = other.GetIntegral();
		this->fError = other.GetError();
		this->fErrorSq = other.GetErrorSq();
		this->fCutAxis = other.GetCutAxis();

		for(size_t i=0; i<N; i++)
		{
			this->fUpperLimit[i]=other.GetUpperLimit(i);
			this->fLowerLimit[i]=other.GetLowerLimit(i);
		}

		return *this;
	}

	__hydra_host__ __hydra_device__
	GenzMalikBox<N>& operator=(result_type const& other)
	{
		auto _pair = hydra::detail::split_tuple<2>(other);

		this->fRule5 = hydra::get<0>(_pair.first ) ;
		this->fRule7 = hydra::get<1>(_pair.first ) ;

			this->fCutAxis = hydra::detail::max( _pair.second);

			GReal_t factor = this->fVolume/::pow(2.0, N);

			this->fIntegral = factor*this->fRule7;
			this->fError    = factor*std::abs(this->fRule7-this->fRule5);
			this->fErrorSq  = this->fError*this->fError;

			return *this;
	}
	__hydra_host__ __hydra_device__
	operator hydra::pair<double, double>() const {
		return hydra::make_pair(this->fIntegral, this->fErrorSq );
	}



	void Print()
	{
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Genz-Malik hyperbox begin: " << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Integral: "  << fIntegral << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Error: "     << fError << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Volume: "  << fVolume << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Rule7: "   << fRule7  << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Rule5: "   << fRule5  << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Cutting dimension: "   << fCutAxis  << HYDRA_ENDL;

		for(size_t i=0; i<N; i++ )
		{
			HYDRA_SPACED_MSG <<"Dimension: " << i << ", Limits: [ "
					<< fLowerLimit[i] << ", "
					<< fUpperLimit[i] <<  "]"<< HYDRA_ENDL;
		}
		HYDRA_MSG << HYDRA_ENDL;

		HYDRA_MSG << "Genz-Malik hyperbox end." << HYDRA_ENDL;

	}

	inline hydra::pair<detail::GenzMalikBox<N>, detail::GenzMalikBox<N>> Divide(){

		auto middle =  0.5*(fUpperLimit[fCutAxis] - fLowerLimit[fCutAxis])+fLowerLimit[fCutAxis];

        detail::GenzMalikBox<N> lower_box(*this);
	    lower_box.SetUpperLimit(fCutAxis, middle);
	    lower_box.SetCutAxis(-1);
	    lower_box.SetError(0.0);
	    lower_box.SetErrorSq(0.0);
	    lower_box.SetIntegral(0.0);


		detail::GenzMalikBox<N> upper_box(*this);
		upper_box.SetLowerLimit(fCutAxis, middle);
		upper_box.SetCutAxis(-1);
		upper_box.SetError(0.0);
		upper_box.SetErrorSq(0.0);
		upper_box.SetIntegral(0.0);
		upper_box.SetIntegral(0.0);


		return hydra::make_pair(lower_box, upper_box);

	}


	__hydra_host__ __hydra_device__
	GReal_t GetRule5() const {
		return fRule5;
	}

	__hydra_host__ __hydra_device__
	void SetRule5(GReal_t rule5) {
		fRule5 = rule5;
	}

	__hydra_host__ __hydra_device__
	GReal_t GetRule7() const {
		return fRule7;
	}

	__hydra_host__ __hydra_device__
	void SetRule7(GReal_t rule7) {
		fRule7 = rule7;
	}

	__hydra_host__ __hydra_device__
	GReal_t GetVolume() const {
		return fVolume;
	}

	__hydra_host__ __hydra_device__
	void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	__hydra_host__ __hydra_device__
	void SetLowerLimit(size_t i, double value) {
		fLowerLimit[i]=value;
		this->UpdateVolume();
	}

	__hydra_host__ __hydra_device__
	void SetUpperLimit(size_t i, double value) {
		fUpperLimit[i]=value;
		this->UpdateVolume();
	}


	__hydra_host__ __hydra_device__
	GReal_t GetLowerLimit(size_t i) const {
		return fLowerLimit[i];
	}

	__hydra_host__ __hydra_device__
	GReal_t GetUpperLimit(size_t i) const {
			return fUpperLimit[i];
		}

	__hydra_host__ __hydra_device__
	GReal_t* GetLowerLimit()  {
		return fLowerLimit;
	}

	__hydra_host__ __hydra_device__
	GReal_t* GetUpperLimit()  {
		return fUpperLimit;
	}

	__hydra_host__ __hydra_device__
	GReal_t GetError() const {

		return fError;
	}

	__hydra_host__ __hydra_device__
	void SetError(GReal_t error) {
		fError = error;
	}

	__hydra_host__ __hydra_device__
	GReal_t GetIntegral() const {

		return fIntegral;
	}

	__hydra_host__ __hydra_device__
	void SetIntegral(GReal_t integral) {
		fIntegral = integral;
	}

	__hydra_host__ __hydra_device__
	GReal_t GetErrorSq() const {
		return fErrorSq;
	}

	__hydra_host__ __hydra_device__
	void SetErrorSq(GReal_t errorSq) {
		fErrorSq = errorSq;
	}
	__hydra_host__ __hydra_device__
	GInt_t GetCutAxis() const {
		return fCutAxis;
	}
	__hydra_host__ __hydra_device__
	void SetCutAxis(GInt_t cutAxis) {
		fCutAxis = cutAxis;
	}

private:

	__hydra_host__ __hydra_device__
	void UpdateVolume(){
		fVolume =1.0;
		for(size_t i=0; i<N; i++)
			fVolume *=(fUpperLimit[i]-fLowerLimit[i]);

	}


	GReal_t fIntegral;
	GReal_t fError;
	GReal_t fErrorSq;
	GReal_t fVolume;
	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fUpperLimit[N];
	GReal_t fLowerLimit[N];
	GInt_t  fCutAxis;

};

}  // namespace detail

} // namespace hydra



#endif /* GENZMALIKBOX_H_ */
