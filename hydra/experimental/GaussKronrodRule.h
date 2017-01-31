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

#include <initializer_list>


namespace hydra {

namespace experimental {


template<size_t N>
struct GaussKronrodRule
{
	constexpr static size_t KronrodN = (N+1)/2;
	constexpr static size_t GaussN   = (KronrodN+1)/2;


	GaussKronrodRule()=delete;

	GaussKronrodRule(std::initializer_list<GReal_t> const& x,
			std::initializer_list<GReal_t> const& gauss_weight,
			std::initializer_list<GReal_t> const& kronrod_weight)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			X[i]=x.begin()[i];
			KronrodWeight[i]=kronrod_weight.begin()[i];
		}

		for(size_t i=0; i<(((N+1)/2)+1)/2;i++ )
		{
			GaussWeight[i]=gauss_weight.begin()[i];
		}

	}



	GaussKronrodRule( GReal_t const (&x)[(N+1)/2],
			GReal_t const (&gauss_weight)[(((N+1)/2)+1)/2],
			GReal_t const (&kronrod_weight)[(N+1)/2]	)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			X[i]=x[i];
			KronrodWeight[i]=kronrod_weight[i];
		}

		for(size_t i=0; i<(((N+1)/2)+1)/2;i++ )
		{
			GaussWeight[i]=gauss_weight[i];
		}
	}
	__host__  __device__
	GaussKronrodRule(GaussKronrodRule<N> const& other	)
	{

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			this->X[i] = other.X[i];
			this->KronrodWeight[i] = other.KronrodWeight[i];
		}

		for(size_t i=0; i<(((N+1)/2)+1)/2;i++ )
		{
			this->GaussWeight[i] = other.GaussWeight[i];
		}
	}
	__host__  __device__
	GaussKronrodRule& operator=(GaussKronrodRule<N> const& other	)
	{
		if(this == &other) return *this;

		for(size_t i=0; i<(N+1)/2;i++ )
		{
			this->X[i] = other.X[i];
			this->KronrodWeight[i] = other.KronrodWeight[i];
		}

		for(size_t i=0; i<(((N+1)/2)+1)/2;i++ )
		{
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
		        HYDRA_MSG << "X[" << i << "] = " << (i?"+-":"") << X[i]
		        		  << " Weight[" << i << "] = " << KronrodWeight[i]
		        		  << HYDRA_ENDL;
		    }
		HYDRA_MSG << "Gauss nodes #" << 2*GaussN-1 << HYDRA_ENDL;
		    for(size_t i=0; i<GaussN; i++ )
		    {
		        HYDRA_MSG << "X[" << i << "] = "<< (i?"+-":"") << X[2*i]
		                  << " Weight[" << i << "] = " << GaussWeight[i]
		                  << HYDRA_ENDL;
		    }
		HYDRA_MSG << "Gauss-Kronrod Rule #" << KronrodN << " end." << HYDRA_ENDL;
	}

	thrust::pair<GReal_t, GReal_t> GetAbscissa(size_t index, GReal_t xlower, GReal_t xupper  )
	{

		GReal_t a = (xupper - xlower)/2.0;
		GReal_t b = (xupper + xlower)/2.0;
	    GReal_t x = a*X[index] + b;

	    return thrust::make_pair(x, a);

	}

	GReal_t X[KronrodN] ;
	GReal_t GaussWeight[GaussN] ;
	GReal_t KronrodWeight[KronrodN] ;

};


    GaussKronrodRule<15> GaussKronrodRule15(
    		{
					0.00000000000000000000000000000000000000000000000000,
					0.20778495500789846760068940377324491347978440714517,
					0.40584515137739716690660641207696146334738201409937,
					0.58608723546769113029414483825872959843678075060436,
					0.74153118559939443986386477328078840707414764714139,
					0.86486442335976907278971278864092620121097230707409,
					0.94910791234275852452618968404785126240077093767062,
					0.99145537112081263920685469752632851664204433837033
    		},
    		//--------------------
    		{
    				0.00000000000000000000000000000000000000000000000000,
    				0.40584515137739716690660641207696146334738201409937,
    				0.74153118559939443986386477328078840707414764714139,
    				0.94910791234275852452618968404785126240077093767062
    		},
    		//--------------------
    		{
    				0.20948214108472782801299917489171426369776208022370,
    				0.20443294007529889241416199923464908471651760418072,
    				0.19035057806478540991325640242101368282607807545536,
    				0.16900472663926790282658342659855028410624490030294,
    				0.14065325971552591874518959051023792039988975724800,
    				0.10479001032225018383987632254151801744375665421383,
    				0.06309209262997855329070066318920428666507115721155,
    				0.02293532201052922496373200805896959199356081127575
    		}
    );


}

}


#endif /* GAUSSKRONRODRULE_H_ */
