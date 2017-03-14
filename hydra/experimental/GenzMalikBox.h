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
#include <array>
#include <initializer_list>
#include <math.h>
#include <cmath>

namespace hydra {

namespace experimental {

template<size_t N>
struct GenzMalikBox
{
	GenzMalikBox()=delete;

	__host__ __device__
	GenzMalikBox(GReal_t (&LowerLimit)[N], GReal_t (&UpperLimit)[N]):
		fRule7(0),
		fRule5(0)
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
		fRule5(0)
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
		fVolume(other.GetVolume() )
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

		for(size_t i=0; i<N; i++)
		{
			this->fFourDifference[i]=other.GetFourDifference(i);
			this->fUpperLimit[i]=other.GetUpperLimit(i);
			this->fLowerLimit[i]=other.GetLowerLimit(i);
		}

		return *this;
	}


	void Print()
	{
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Genz-Malik hyperbox begin: " << HYDRA_ENDL;
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
		HYDRA_MSG << "Genz-Malik hyperbox end." << HYDRA_ENDL;

	}

	__host__ __device__
	const GReal_t GetFourDifference(size_t i) const {
		return fFourDifference[i];
	}

	__host__ __device__
	const GReal_t GetLowerLimit(size_t i) const {
		return fLowerLimit[i];
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
	const GReal_t GetUpperLimit(size_t i) const {
		return fUpperLimit[i];
	}
	__host__ __device__
	GReal_t GetVolume() const {
		return fVolume;
	}
	__host__ __device__
	void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

private:

	GReal_t fVolume;
	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fFourDifference[N];
	GReal_t fUpperLimit[N];
	GReal_t fLowerLimit[N];

};


}  // namespace experimental

} // namespace hydra



#endif /* GENZMALIKBOX_H_ */
