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
 * Parameter.h
 *
 *  Created on: 26/08/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <string>
#include <ostream>


namespace hydra {

/**
 *  @ingroup fit, generic
 *  @brief This class represents named parameters that hold information of value, error, limits and implements the interface with ROOT::Minuit2.
 *
 *  hydra::Parameter instances are constructible, assignable and copiable in all backends, and because that the storage to hold the name
 *  needs be managed by the user (ex. no std::string support in CUDA).
 *  hydra::Parameter overloads the GReal_t operator()() and the arithmetic operators.
 *  hydra::Parameter instances can be constructed using named parameter semantic or parameter list semantic:
 @code{cpp}
    //using named parameter idiom
	Parameter  mean   = Parameter::Create()
								 .Name(name)
								 .Value(3.0)
								 .Error(0.000)
								 .Limits(1.0, 4.0)
								 ;

	//using unnamed parameter idiom
	Parameter  mean(name, 3.0, 0.000, 1.0, 4.0);

 @endcode
 *
 *
 */
struct Parameter{


	__host__ __device__
	Parameter():
	fName(const_cast<GChar_t*>("")),
	fValue(detail::TypeTraits<GReal_t>::invalid()),
	fError(detail::TypeTraits<GReal_t>::invalid()),
	fLowerLim(detail::TypeTraits<GReal_t>::invalid()),
	fUpperLim(detail::TypeTraits<GReal_t>::invalid()),
	fIndex(detail::TypeTraits<GInt_t>::invalid()),
	fLimited(0),
    fHasError(0)
	{}


	__host__ __device__
	Parameter(GReal_t value):
	fName(const_cast<GChar_t*>("")),
	fValue(value),
	fError(detail::TypeTraits<GReal_t>::zero()),
	fLowerLim(detail::TypeTraits<GReal_t>::zero()),
	fUpperLim(detail::TypeTraits<GReal_t>::zero()),
	fIndex(detail::TypeTraits<GInt_t>::invalid()),
	fLimited(0),
	fHasError(0)
	{}




	Parameter(std::string const& name, GReal_t value, GReal_t error, GReal_t downlim, GReal_t uplim):
	fName(const_cast<GChar_t*>(name.data())),
	fValue(value),
	fError(error),
	fLowerLim(downlim),
	fUpperLim(uplim),
	fIndex(detail::TypeTraits<GInt_t>::invalid()),
	fLimited(1),
	fHasError(1)
	{ }

	Parameter(std::string const& name, GReal_t value, GReal_t error):
		fName(const_cast<GChar_t*>(name.data())),
		fValue(value),
		fError(error),
		fLowerLim(detail::TypeTraits<GReal_t>::invalid()),
		fUpperLim(detail::TypeTraits<GReal_t>::invalid()),
		fIndex(detail::TypeTraits<GInt_t>::invalid()),
		fLimited(0),
		fHasError(1)
	{ }

	Parameter(GChar_t *name, GReal_t value, GReal_t error):
			fName(name),
			fValue(value),
			fError(error),
			fLowerLim(detail::TypeTraits<GReal_t>::invalid()),
			fUpperLim(detail::TypeTraits<GReal_t>::invalid()),
			fIndex(detail::TypeTraits<GInt_t>::invalid()),
			fLimited(0),
			fHasError(1)
		{ }

	Parameter(std::string const& name, GReal_t value):
		fName(const_cast<GChar_t*>(name.data())),
		fValue(value),
		fError(detail::TypeTraits<GReal_t>::invalid()),
		fLowerLim(detail::TypeTraits<GReal_t>::invalid()),
		fUpperLim(detail::TypeTraits<GReal_t>::invalid()),
		fIndex(detail::TypeTraits<GInt_t>::invalid()),
		fLimited(0),
		fHasError(0)
	{ }




	__host__ __device__
	inline Parameter( Parameter const& other ):
		fValue(other.GetValue()),
		fError(other.GetError()),
		fLowerLim(other.GetLowerLim()),
		fUpperLim(other.GetUpperLim()),
		fIndex(other.GetIndex()),
		fLimited( other.IsLimited()),
		fHasError(other.HasError())
	{


		fName= const_cast<GChar_t*>(other.GetName());

	}

	__host__ __device__
	inline Parameter& operator=(Parameter const& other)
	{
		if(this != &other){
			this->fValue    = other.GetValue();
			this->fError    = other.GetError();
			this->fLowerLim  = other.GetLowerLim();
			this->fUpperLim    = other.GetUpperLim();
			this->fIndex    = other.GetIndex();
			this->fLimited  = other.IsLimited();
			this->fHasError = other.HasError();
			this->fName    = const_cast<GChar_t*>(other.GetName());
		}
		return *this;
	}

	__host__ __device__
	inline Parameter& operator=(const GReal_t value)
	{
			this->fValue   = value;

		return *this;
	}

	__host__ __device__
	inline Parameter& operator+=(const GReal_t value)
	{
		this->fValue   += value;

		return *this;
	}

	__host__ __device__
	inline Parameter& operator+=(Parameter const& other)
	{
		this->fValue   += other.GetValue();

		return *this;
	}

	__host__ __device__
	inline Parameter& operator-=(const GReal_t value)
	{
		this->fValue   -= value;

		return *this;
	}

	__host__ __device__
	inline Parameter& operator-=(Parameter const& other)
	{
		this->fValue   -= other.GetValue();

		return *this;
	}

	__host__ __device__
	inline Parameter& operator*=(const GReal_t value)
	{
		this->fValue   *= value;

		return *this;
	}

	__host__ __device__
	inline Parameter& operator*=(Parameter const& other)
	{
		this->fValue   *= other.GetValue();

		return *this;
	}


	__host__ __device__
	inline Parameter& operator/=(const GReal_t value)
	{
		this->fValue   /= value;

		return *this;
	}

	__host__ __device__
	inline Parameter& operator/=(Parameter const& other)
	{
		this->fValue   /= other.GetValue();

		return *this;
	}


	__host__ __device__
	inline GReal_t operator()() {
		return this->fValue;
	}

	__host__ __device__
	inline GReal_t operator()() const {
			return this->fValue;
		}

	__host__ __device__
	inline GReal_t GetLowerLim() const {
		return fLowerLim;
	}

	__host__ __device__
	inline void SetLowerLim(GReal_t downLim) {
		fLowerLim = downLim;
		fLimited=1;
	}

	__host__ __device__
	inline unsigned int GetIndex() const {
		return fIndex;
	}

	__host__ __device__
	inline void SetIndex(unsigned int index) {
		fIndex = index;
	}

	__host__ __device__
	inline GReal_t GetUpperLim() const {
		return fUpperLim;
	}

	__host__ __device__
	inline void SetLimits(GReal_t lower , GReal_t upper) {
		fLowerLim = lower;
		fUpperLim = upper;
		fLimited=1;
	}

	__host__ __device__
	inline void SetUpperLim(GReal_t upLim) {
		fUpperLim = upLim;
		fLimited=1;
	}

	__host__ __device__
	inline void SetValue(GReal_t value) {
		fValue = value;
	}

	__host__ __device__
	inline const GChar_t* GetName() const {
		return fName;
	}

	__host__
	inline void SetName(const std::string& name) {
		this->fName = const_cast<GChar_t*>(name.c_str());

	}

	__host__ __device__
	inline GReal_t GetValue() const {
		return fValue;
	}

	__host__ inline
	void Reset(const std::vector<double>& parameters)
	{
		//if(fIndex <0) return;
		fValue=parameters[fIndex];
	}

	__host__ __device__
	inline GReal_t GetError() const {
		return fError;
	}

	__host__ __device__
	inline void SetError(GReal_t error) {
		fError = error;
		fHasError=1;
	}

	__host__ __device__
		inline GBool_t IsLimited() const {
		return fLimited;
	}

	__host__ __device__
			inline void SetLimited(GBool_t limited) {
		fLimited = limited;
	}

	__host__ __device__
			inline GBool_t HasError() const {
		return fHasError;
	}

	__host__ __device__
	inline void SetHasError(GBool_t nullError) {
		fHasError = nullError;
	}

	__host__ __device__
	inline operator GReal_t() { return fValue; }

	__host__ __device__
	inline operator GReal_t() const { return fValue; }


	__host__
	static Parameter Create() {
	      return Parameter();
	    }

	__host__
	Parameter& Name(std::string const& name ){
		this->fName = const_cast<GChar_t*>(name.data());
		return *this;
	}

	__host__
	Parameter& Error(GReal_t error){
		this->fError = error;
		this->fHasError=1;
		return *this;
	}

	__host__
	Parameter& Value(GReal_t value){
		this->fValue = value;
		return *this;
	}

	__host__
	Parameter& Limits(GReal_t lowlim, GReal_t uplim){
		this->fUpperLim=uplim;
		this->fLowerLim=lowlim;
		this->fLimited=1;
		return *this;
	}

private:

	GChar_t* fName;
	GReal_t  fValue;
	GReal_t  fError;
	GReal_t  fLowerLim;
	GReal_t  fUpperLim;
	GUInt_t  fIndex;
	GBool_t  fLimited;
	GBool_t  fHasError;

};

/*
 * addition
 */
__host__ __device__
inline Parameter operator+(Parameter par1, Parameter const& par2)
{
		par1  += par2;

		return par1;
}

__host__ __device__
inline GReal_t operator+(Parameter par1, GReal_t par2)
{
		par1  += par2;

		return par1;
}


/*
 * subtraction
 */
__host__ __device__
inline Parameter operator-(Parameter par1, Parameter const&  par2)
{
		par1  -= par2;

		return par1;
}

__host__ __device__
inline GReal_t operator-(Parameter par1, GReal_t  par2)
{
		par1  -= par2;

		return par1;
}

__host__ __device__
inline GReal_t operator-(GReal_t par1, Parameter  par2)
{
		par1  -= par2;

		return par1;
}

/*
 * multiplication
 */
__host__ __device__
inline Parameter operator*(Parameter par1, Parameter const&  par2)
{
		par1  *= par2;

		return par1;
}


__host__ __device__
inline GReal_t operator*(Parameter par1, GReal_t  par2)
{
		par1  *= par2;

		return par1;
}

/*
 * division
 */
__host__ __device__
inline Parameter operator/(Parameter par1, Parameter const par2)
{
		par1  /= par2;

		return par1;
}

__host__ __device__
inline GReal_t operator/(Parameter par1, GReal_t par2)
{
		par1  /= par2;

		return par1;
}


__host__ __device__
inline GReal_t operator/( GReal_t par1, Parameter par2 )
{
		par1  /= par2;

		return par1;
}



__host__
inline std::ostream& operator<<(std::ostream& os, Parameter const& var){

	return os<< "Hydra::Variable: "<< var.GetName()  << "[ " << var.GetValue()
			 << ", " << var.GetError() << ", " << var.GetLowerLim()
			 << ", " << var.GetUpperLim() << "] Index: "<< var.GetIndex() ;
}

}  // namespace hydra
#endif /* PARAMETER_H_ */
