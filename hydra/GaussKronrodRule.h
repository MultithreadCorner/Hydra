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
 * GaussKronrodRule.h
 *
 *  Created on: Jan 25, 2017
 *      Author: Antonio Augusto Alves Junior
 */



#ifndef GAUSSKRONRODRULE_H_
#define GAUSSKRONRODRULE_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>

#include <hydra/detail/external/thrust/pair.h>
#include <initializer_list>
#include <iomanip>

namespace hydra {

/**
 * \ingroup numerical_integration
 * \brief Rules for Gauss-Kronrod quadrature.
 *
 */
template<size_t N>
struct GaussKronrodRule
{
	constexpr static size_t KronrodN = (N+1)/2;
	constexpr static size_t GaussN   = (N+1)/2;


	GaussKronrodRule()=delete;


	GaussKronrodRule(std::initializer_list<GReal_t> const& x,
			std::initializer_list<GReal_t> const& gauss_weight,
			std::initializer_list<GReal_t> const& kronrod_weight)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			X[i]=x.begin()[i];
			KronrodWeight[i]=kronrod_weight.begin()[i];
			GaussWeight[i]=gauss_weight.begin()[i];
		}

	}


	__hydra_host__  __hydra_device__
	GaussKronrodRule( GReal_t const (&x)[(N+1)/2],
			GReal_t const (&gauss_weight)[(N+1)/2],
			GReal_t const (&kronrod_weight)[(N+1)/2]	)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			X[i]=x[i];
			KronrodWeight[i]=kronrod_weight[i];
			GaussWeight[i]=gauss_weight[i];
		}


	}

	__hydra_host__  __hydra_device__
	GaussKronrodRule(GaussKronrodRule<N> const& other	)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			this->X[i] = other.X[i];
			this->KronrodWeight[i] = other.KronrodWeight[i];
			this->GaussWeight[i] = other.GaussWeight[i];
		}

	}

	__hydra_host__  __hydra_device__
	inline GaussKronrodRule& operator=(GaussKronrodRule<N> const& other	)
	{
		if(this == &other) return *this;

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			this->X[i] = other.X[i];
			this->KronrodWeight[i] = other.KronrodWeight[i];
			this->GaussWeight[i] = other.GaussWeight[i];
		}


		return *this;
	}

	void Print()
	{
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Gauss-Kronrod Rule #" << 2*KronrodN-1 << " begin:" << HYDRA_ENDL;
		HYDRA_MSG << "Kronrod nodes #" << 2*KronrodN-1 << " begin:" << HYDRA_ENDL;
		for(size_t i=0; i<KronrodN; i++ )
		{
			HYDRA_MSG << std::setprecision(50) << "X[" << i << "] = " << (i?"+-":"") << X[i]
			                                                     << " Weight[" << i << "] = " << KronrodWeight[i]
			                                                                                                   << HYDRA_ENDL;
		}
		HYDRA_MSG << "Gauss nodes #" << 2*GaussN-1 << HYDRA_ENDL;
		for(size_t i=0; i<GaussN; i++ )
		{
			HYDRA_MSG << std::setprecision(50) <<"X[" << i << "] = "<< (i?"+-":"") << X[i]
			                                                    << " Weight[" << i << "] = " << GaussWeight[i]
			                                                                                                << HYDRA_ENDL;
		}
		HYDRA_MSG << "Gauss-Kronrod Rule #" << KronrodN << " end." << HYDRA_ENDL;
	}

	__hydra_host__  __hydra_device__
	inline HYDRA_EXTERNAL_NS::thrust::tuple<GReal_t, GReal_t, GReal_t>
	GetAbscissa(size_t index, GReal_t xlower, GReal_t xupper  )
		{

		GReal_t a = (xupper - xlower)/2.0;
		GReal_t b = (xupper + xlower)/2.0;
		GReal_t x_p =  a*X[index] + b;
		GReal_t x_m = -a*X[index] + b;

		return HYDRA_EXTERNAL_NS::thrust::make_tuple(x_p, x_m, index==0?a/2.0:a);

		}

	GReal_t X[KronrodN] ;
	GReal_t GaussWeight[GaussN] ;
	GReal_t KronrodWeight[KronrodN] ;

};


}//namespace hydra


#endif /* GAUSSKRONRODRULE_H_ */
