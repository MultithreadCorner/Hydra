/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
#include <hydra/Types.h>
#include <hydra/detail/utility/Generic.h>
#include <thrust/tuple.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <array>
#include <initializer_list>
#include <math.h>
#include <cmath>




namespace hydra {

namespace experimental {

namespace detail {

/*
template <size_t N>
struct GenzMalikBoxResult
{
	typedef void hydra_convertible_to_tuple_tag;

	typedef typename hydra::detail::tuple_type<N+2, GReal_t>::type data_type;


	__host__ __device__
	GenzMalikBoxResult():
	fData( data_type() )
	{	}

	__host__ __device__
	GenzMalikBoxResult( GenzMalikBoxResult<N>const & other)
	{
        this->fData = other.fData;
	}

	template<typename ...T>
	__host__ __device__
	GenzMalikBoxResult( thrust::tuple<T...> const& t):
	fData(t)
	{	}

	__host__ __device__
	GenzMalikBoxResult<N>& operator=( GenzMalikBoxResult<N>const & other)
	{
		if(this==&other) return *this;

		this->fData = other.fData;

		return *this;
	}




	template<typename ...T>
	__host__ __device__
	GenzMalikBoxResult<N>& operator= ( thrust::tuple<T...> const& t )
	{
		this->fData = t;
	    return *this;
	}

	template<typename ...T>
	__host__ __device__
    GenzMalikBoxResult<N>& operator= (thrust::detail::tuple_of_iterator_references<T&...> const&  t )
	{
		this->fData = t;
		return *this;
	}



	template<typename ...T>
	__host__ __device__
	operator thrust::tuple<T...> ( )
	{

		return  fData;

	}

	template<typename ...T>
	__host__ __device__
	operator thrust::detail::tuple_of_iterator_references<T &...> ( )
	{

			return  fData;
	}


	data_type fData;


};
*/

template<size_t N>
struct GenzMalikBox
{
	typedef typename hydra::detail::tuple_type<N+2, GReal_t>::type data_type;

	GenzMalikBox()=delete;


	GenzMalikBox(GReal_t (&LowerLimit)[N], GReal_t (&UpperLimit)[N]):
		fRule7(0),
		fRule5(0),
		fIntegral(0),
		fError(0),
		fErrorSq(0)
	{
		fVolume =1.0;
		for(size_t i=0; i<N; i++)
		{
			fFourDifference[i]=0;
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
		fErrorSq(0)
	{
		fVolume =1.0;
		for(size_t i=0; i<N; i++)
		{
			fFourDifference[i]=0;
			fUpperLimit[i]=UpperLimit[i];
			fLowerLimit[i]=LowerLimit[i];
			fVolume *=(UpperLimit[i]-LowerLimit[i]);
		}


	}

	__host__ __device__
	GenzMalikBox(GenzMalikBox<N> const& other):
		fRule7(other.GetRule7() ),
		fRule5(other.GetRule5() ),
		fVolume(other.GetVolume() ),
		fIntegral(other.GetIntegral()),
		fError(other.GetError()),
		fErrorSq(other.GetErrorSq())

	{
		for(size_t i=0; i<N; i++)
		{
			fFourDifference[i]=other.GetFourDifference(i);
			fUpperLimit[i]=other.GetUpperLimit(i);
			fLowerLimit[i]=other.GetLowerLimit(i);

		}
	}

	__host__ __device__
	GenzMalikBox<N>& operator=(GenzMalikBox<N> const& other)
	{
		if(this==&other) return *this;

		this->fRule7 = other.GetRule7() ;
		this->fRule5 = other.GetRule5() ;
		this->fVolume = other.GetVolume() ;
		this->fIntegral = other.GetIntegral();
		this->fError = other.GetError();
		this->fErrorSq = other.GetErrorSq();

		for(size_t i=0; i<N; i++)
		{
			this->fFourDifference[i]=other.GetFourDifference(i);
			this->fUpperLimit[i]=other.GetUpperLimit(i);
			this->fLowerLimit[i]=other.GetLowerLimit(i);
		}

		return *this;
	}

	__host__ __device__
	GenzMalikBox<N>& operator=(data_type & other)
	{

		GReal_t _temp[N+2];
		hydra::detail::tupleToArray(other, &_temp[0]);

			this->fRule5 = _temp[0] ;
			this->fRule7 = _temp[1] ;

			for(size_t i=0; i<N; i++)
			{
				this->fFourDifference[i]=_temp[i+2];
			}

			GReal_t factor = this->fVolume/hydra::detail::power<2, N>::value;

			this->fIntegral = factor*this->fRule7;
			this->fError    = factor*std::abs(this->fRule7-this->fRule5);
			this->fErrorSq  = this->fError*this->fError;

			return *this;
	}

	void Print()
	{
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Genz-Malik hyperbox begin: " << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Integral: "  << fIntegral << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Volume: "  << fVolume << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Rule7: "   << fRule7  << HYDRA_ENDL;
		HYDRA_SPACED_MSG << "Rule5: "   << fRule5  << HYDRA_ENDL;
		for(size_t i=0; i<N; i++ )
		{
			HYDRA_SPACED_MSG <<"Dimension: " << i << ", Limits: [ "
					<< fLowerLimit[i] << ", "
					<< fUpperLimit[i] <<  "] , Four Difference: "
					<< fFourDifference[i] << HYDRA_ENDL;
		}
		HYDRA_MSG << HYDRA_ENDL;

		HYDRA_MSG << "Genz-Malik hyperbox end." << HYDRA_ENDL;

	}

	__host__ __device__
	GReal_t GetFourDifference(size_t i) const {
		return fFourDifference[i];
	}

	__host__ __device__
	GReal_t GetRule5() const {
		return fRule5;
	}

	__host__ __device__
	void SetRule5(GReal_t rule5) {
		fRule5 = rule5;
	}

	__host__ __device__
	GReal_t GetRule7() const {
		return fRule7;
	}

	__host__ __device__
	void SetRule7(GReal_t rule7) {
		fRule7 = rule7;
	}

	__host__ __device__
	GReal_t GetVolume() const {
		return fVolume;
	}

	__host__ __device__
	void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	__host__ __device__
	GReal_t GetLowerLimit(size_t i) const {
		return fLowerLimit[i];
	}

	__host__ __device__
	GReal_t GetUpperLimit(size_t i) const {
			return fUpperLimit[i];
		}

	__host__ __device__
	GReal_t* GetLowerLimit()  {
		return fLowerLimit;
	}

	__host__ __device__
	GReal_t* GetUpperLimit()  {
		return fUpperLimit;
	}




	__host__ __device__
	GReal_t GetError() const {

		return fError;
	}

	__host__ __device__
	void SetError(GReal_t error) {
		fError = error;
	}

	__host__ __device__
	GReal_t GetIntegral() const {

		return fIntegral;
	}

	__host__ __device__
	void SetIntegral(GReal_t integral) {
		fIntegral = integral;
	}

	__host__ __device__
	GReal_t GetErrorSq() const {
		return fErrorSq;
	}

	__host__ __device__
	void SetErrorSq(GReal_t errorSq) {
		fErrorSq = errorSq;
	}

private:

	GReal_t fIntegral;
	GReal_t fError;
	GReal_t fErrorSq;
	GReal_t fVolume;
	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fFourDifference[N];
	GReal_t fUpperLimit[N];
	GReal_t fLowerLimit[N];

};

}  // namespace detail

}  // namespace experimental

} // namespace hydra



#endif /* GENZMALIKBOX_H_ */
