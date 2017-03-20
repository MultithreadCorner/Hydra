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

template<size_t N, unsigned int BACKEND=hydra::host>
struct GenzMalikBox
{

	//abscissa<0> -> degree five  weight
	//abscissa<1> -> degree seven weight
	//abscissa<(Index >=2)> -> multidimensional abscissa values
    typedef typename GenzMalikRule< N, BACKEND>::abscissa_t rule_abscissa_t;
    typedef typename hydra::detail::tuple_type<N+4,GReal_t >::type abscissa_t;
    //system selection
	typedef hydra::detail::BackendTraits<BACKEND> system_t;

	//container template vector<abscissa> on device or host memory
	typedef typename system_t::template container<abscissa_t> super_t;

	//container
	typedef multivector<super_t> vector_abscissa_t;
	typedef typename multivector<super_t>::iterator vector_abscissa_iterator;
	typedef typename multivector<super_t>::const_iterator vector_abscissa_const_iterator;


	GenzMalikBox()=delete;


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

		abscissa_t abscissa;

		for( auto original_abscissa: GenzMalikRule< N, BACKEND>().GetAbscissas() )
		{
			thrust::get<0>(abscissa)  = thrust::get<0>(original_abscissa);
			thrust::get<1>(abscissa)  = thrust::get<1>(original_abscissa);
			thrust::get<2>(abscissa)  = thrust::get<3>(original_abscissa);
			thrust::get<3>(abscissa)  = thrust::get<4>(original_abscissa);

			GetTransformedAbscissa(original_abscissa, abscissa);
			fAbscissas.push_back(abscissa);
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

		GenzMalikRule< N, BACKEND> GenzMalikRule;
		abscissa_t abscissa;

		for( auto original_abscissa: GenzMalikRule.GetAbscissas() )
		{
			thrust::get<0>(abscissa)  = thrust::get<0>(original_abscissa);
		thrust::get<1>(abscissa)  = thrust::get<1>(original_abscissa);
		thrust::get<2>(abscissa)  = thrust::get<3>(original_abscissa);
		thrust::get<3>(abscissa)  = thrust::get<4>(original_abscissa);

			GetTransformedAbscissa(original_abscissa, abscissa);
			fAbscissas.push_back(abscissa);
		}

	}


	GenzMalikBox(GenzMalikBox<N> const& other):
		fRule7(other.GetRule7() ),
		fRule5(other.GetRule5() ),
		fVolume(other.GetVolume() ),
		fAbscissas(other.GetAbscissas())
	{
		for(size_t i=0; i<N; i++)
		{
			fFourDifference[i]=other.GetFourDifference(i);
			fUpperLimit[i]=other.GetUpperLimit(i);
			fLowerLimit[i]=other.GetLowerLimit(i);

		}
	}


	GenzMalikBox<N>& operator=(GenzMalikBox<N> const& other)
	{
		if(this==&other) return *this;

		this->fRule7 = other.GetRule7() ;
		this->fRule5 = other.GetRule5() ;
		this->fVolume = other.GetVolume() ;
		this->fAbscissas = other.GetAbscissas();

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
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Abscissas Rule begin:"              << HYDRA_ENDL;
		HYDRA_MSG << "(weight #5, weight #7, ...{abscissas})" << HYDRA_ENDL;
		for(auto row:fAbscissas)
		{
			HYDRA_SPACED_MSG << row << HYDRA_ENDL;
		}
		HYDRA_SPACED_MSG << "Number of function calls: "<< fAbscissas.size() << HYDRA_ENDL;

		HYDRA_MSG << "Genz-Malik hyperbox end." << HYDRA_ENDL;

	}

	const GReal_t GetFourDifference(size_t i) const {
		return fFourDifference[i];
	}

	const GReal_t GetLowerLimit(size_t i) const {
		return fLowerLimit[i];
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


	const GReal_t GetUpperLimit(size_t i) const {
		return fUpperLimit[i];
	}

	GReal_t GetVolume() const {
		return fVolume;
	}

	void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	const vector_abscissa_t& GetAbscissas() const {
		return fAbscissas;
	}

	void SetAbscissas(const vector_abscissa_t& abscissas) {
		fAbscissas = abscissas;
	}

private:

	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	GetTransformedAbscissa( rule_abscissa_t const& original_abscissa, abscissa_t& transformed_abscissa )
	{	}

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	GetTransformedAbscissa( rule_abscissa_t const& original_abscissa, abscissa_t& transformed_abscissa  )
	{
		GReal_t a = (fUpperLimit[I] - fLowerLimit[I])/2.0;
		GReal_t b = (fUpperLimit[I] + fLowerLimit[I])/2.0;
/*
		I==0? thrust::get<0>(transformed_abscissa)  = thrust::get<0>(original_abscissa):0;
		I==0? thrust::get<1>(transformed_abscissa)  = thrust::get<1>(original_abscissa):0;
		I==0? thrust::get<2>(transformed_abscissa)  = thrust::get<3>(original_abscissa):0;
		I==0? thrust::get<3>(transformed_abscissa)  = thrust::get<4>(original_abscissa):0;
*/
		thrust::get<I+4>(transformed_abscissa)  =  a*thrust::get<2>(original_abscissa )*thrust::get<I+5>(original_abscissa )+ b;


		//std::cout<< I << " " <<  1.0*thrust::get<I>(original_abscissa )<< std::endl;

		GetTransformedAbscissa<I+1>(original_abscissa,transformed_abscissa );
	}



	GReal_t fVolume;
	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fFourDifference[N];
	GReal_t fUpperLimit[N];
	GReal_t fLowerLimit[N];
	vector_abscissa_t fAbscissas;

};


}  // namespace experimental

} // namespace hydra



#endif /* GENZMALIKBOX_H_ */
