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

	GenzMalikBox(GReal_t (&LowerLimit)[N], GReal_t (&UpperLimit)[N]):
		fRule7(0),
		fRule5(0),
		fVolume(1.0)
	{
		for(size_t i=0; i<N; i++)
		{
			fFourDifference[N]=0;
			fUpperLimit[N]=UpperLimit[N];
			fLowerLimit[N]=LowerLimit[N];
			fVolume*=(UpperLimit[N]-LowerLimit[N]);
		}
	}



	GenzMalikBox(std::array<GReal_t,N> const& LowerLimit, std::array<GReal_t,N> const&  UpperLimit[N]):
				fRule7(0),
				fRule5(0),
				fVolume(1.0)
			{
				for(size_t i=0; i<N; i++)
				{
					fFourDifference[N]=0;
					fUpperLimit[N]=UpperLimit[N];
					fLowerLimit[N]=LowerLimit[N];
					fVolume*=(UpperLimit[N]-LowerLimit[N]);
				}
			}

	GenzMalikBox(GenzMalikBox<N> const& other):
					fRule7(other.GetRule7() ),
					fRule5(other.GetRule5() ),
					fVolume(other.GetVolume() )
				{
					for(size_t i=0; i<N; i++)
					{
						fFourDifference[i]=other.GetFourDifference()[i];
						fUpperLimit[i]=other.GetUpperLimit()[i];
						fLowerLimit[i]=other.GetLowerLimit()[i];
						fVolume*=(fUpperLimit[i]-fLowerLimit[i]);
					}
				}

	GenzMalikBox<N>& operator=(GenzMalikBox<N> const& other)
	{
		if(this==&other) return *this;

		this->fRule7 = other.GetRule7() ;
		this->fRule5 = other.GetRule5() ;
		this->fVolume = other.GetVolume() ;

		for(size_t i=0; i<N; i++)
		{
			this->fFourDifference[i]=other.GetFourDifference()[i];
			this->fUpperLimit[i]=other.GetUpperLimit()[i];
			this->fLowerLimit[i]=other.GetLowerLimit()[i];
			this->fVolume*=(fUpperLimit[i]-fLowerLimit[i]);
		}

		return *this;
	}


	const GReal_t* GetFourDifference() const {
		return fFourDifference;
	}

	const GReal_t* GetLowerLimit() const {
		return fLowerLimit;
	}

	GReal_t GetRule5() const {
		return fRule5;
	}

	void SetRule5(GReal_t rule5) {
		fRule5 = rule5;
	}

	GReal_t GetRule7() const {
		return fRule7;
	}

	void SetRule7(GReal_t rule7) {
		fRule7 = rule7;
	}

	const GReal_t* GetUpperLimit() const {
		return fUpperLimit;
	}

	GReal_t GetVolume() const {
		return fVolume;
	}

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
