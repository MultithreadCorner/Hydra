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
 * GenzMalikRule.h
 *
 *  Created on: 22/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENZMALIKRULE_H_
#define GENZMALIKRULE_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/experimental/multivector.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
//std
#include <type_traits>
#include <array>
#include <initializer_list>
#include <math.h>
#include <vector>
#include <tuple>
#include <utility>

namespace hydra {

namespace experimental {


template<typename T>
struct GenzMalikRuleBase{};

template<size_t DIM, unsigned int BACKEND=hydra::host>
struct GenzMalikRule: GenzMalikRuleBase<typename std::enable_if< (DIM>1), void >::type >
	{


	//abscissa<0> -> degree five  weight
	//abscissa<1> -> degree seven weight
	//abscissa<(Index >=2)> -> multidimensional abscissa values
	typedef typename hydra::detail::tuple_type<DIM+2,GReal_t >::type abscissa_t;

	//system selection
	typedef hydra::detail::BackendTraits<BACKEND> system_t;

	//container template vector<abscissa> on device or host memory
	typedef typename system_t::template container<abscissa_t> super_t;

	//container
	typedef multivector<super_t> vector_abscissa_t;
	typedef typename multivector<super_t>::iterator vector_abscissa_iterator;
	typedef typename multivector<super_t>::const_iterator vector_abscissa_const_iterator;


		GenzMalikRule():
			fLambda2( sqrt(9.0/70.0) ),
			fLambda3( sqrt(9.0/10.0) ),
			fLambda4( sqrt(9.0/10.0) ),
			fLambda5( sqrt(9.0/19.0) )
			{

			GULong64_t factor  = twoN<DIM>();
			fRule7Weight1= factor*GReal_t(12824.0 - 9120.0*DIM + 400.0*DIM*DIM)/19683.0;
			fRule7Weight2= factor*GReal_t(980.0/6561.0);
			fRule7Weight3= factor*GReal_t(1820.0 - 400.0*DIM)/19683.0;
			fRule7Weight4= factor*GReal_t(200.0/19683.0);
			fRule7Weight5= factor*GReal_t(6859.0/19683.0);

			fRule5Weight1= factor*GReal_t(729.0 - 950.0*DIM + 50.0*DIM*DIM)/729.0;
			fRule5Weight2= factor*GReal_t(245.0/486.0);
			fRule5Weight3= factor*GReal_t(265.0 - 100.0*DIM)/1458.0;
			fRule5Weight4= factor*GReal_t(25.0/729.0);

			set_abscissas();

			}

		template<unsigned int BACKEND2>
		GenzMalikRule( GenzMalikRule<DIM,BACKEND2> const& other):
			fLambda2( other.GetLambda2() ),
			fLambda3( other.GetLambda3() ),
			fLambda4( other.GetLambda4() ),
			fLambda5( other.GetLambda5() ),
			fRule7Weight1( other.GetRule7Weight1() ),
			fRule7Weight2( other.GetRule7Weight2() ),
			fRule7Weight3( other.GetRule7Weight3() ),
			fRule7Weight4( other.GetRule7Weight4() ),
			fRule7Weight5( other.GetRule7Weight5() ),
			fRule5Weight1( other.GetRule5Weight1() ),
			fRule5Weight2( other.GetRule5Weight2() ),
			fRule5Weight3( other.GetRule5Weight3() ),
			fRule5Weight4( other.GetRule5Weight4() ),
			fAbscissas( other.GetAbscissas() )
		{}

		template<unsigned int BACKEND2>
		GenzMalikRule<DIM,BACKEND>& operator= ( GenzMalikRule<DIM,BACKEND2> const& other)
		{

			if(this==&other) return *this;

			fLambda2 = other.GetLambda2() ;
			fLambda3 = other.GetLambda3() ;
			fLambda4 = other.GetLambda4() ;
			fLambda5 = other.GetLambda5() ;
			fRule7Weight1 = other.GetRule7Weight1() ;
			fRule7Weight2 = other.GetRule7Weight2() ;
			fRule7Weight3 = other.GetRule7Weight3() ;
			fRule7Weight4 = other.GetRule7Weight4() ;
			fRule7Weight5 = other.GetRule7Weight5() ;
			fRule5Weight1 = other.GetRule5Weight1() ;
			fRule5Weight2 = other.GetRule5Weight2() ;
			fRule5Weight3 = other.GetRule5Weight3() ;
			fRule5Weight4 = other.GetRule5Weight4() ;
			fAbscissas = other.GetAbscissas() ;

			return *this;
		}

		void Print()
			{
				HYDRA_MSG << HYDRA_ENDL;
				HYDRA_MSG << "Genz-Malik Rule begin:"              << HYDRA_ENDL;
				HYDRA_MSG << "(weight #5, weight #7, ...{abscissas})" << HYDRA_ENDL;
				for(auto row:fAbscissas)
				{
					HYDRA_MSG << row << HYDRA_ENDL;
				}
				HYDRA_MSG << "Number of function calls: "<< fAbscissas.size() << HYDRA_ENDL;
				HYDRA_MSG << "Genz-Malik Rule end."              << HYDRA_ENDL;
			}

	const vector_abscissa_t& GetAbscissas() const
	{
		return fAbscissas;
	}

	void SetAbscissas(const vector_abscissa_t& abscissas)
	{
		fAbscissas = abscissas;
	}

	GReal_t GetLambda2() const
	{
		return fLambda2;
	}

	void SetLambda2(GReal_t lambda2)
	{
		fLambda2 = lambda2;
	}

	GReal_t GetLambda3() const
	{
		return fLambda3;
	}

	void SetLambda3(GReal_t lambda3)
	{
		fLambda3 = lambda3;
	}

	GReal_t GetLambda4() const
	{
		return fLambda4;
	}

	void SetLambda4(GReal_t lambda4)
	{
		fLambda4 = lambda4;
	}

	GReal_t GetLambda5() const
	{
		return fLambda5;
	}

	void SetLambda5(GReal_t lambda5)
	{
		fLambda5 = lambda5;
	}

	GReal_t GetRule5Weight1() const
	{
		return fRule5Weight1;
	}

	void SetRule5Weight1(GReal_t rule5Weight1)
	{
		fRule5Weight1 = rule5Weight1;
	}

	GReal_t GetRule5Weight2() const
	{
		return fRule5Weight2;
	}

	void SetRule5Weight2(GReal_t rule5Weight2)
	{
		fRule5Weight2 = rule5Weight2;
	}

	GReal_t GetRule5Weight3() const
	{
		return fRule5Weight3;
	}

	void SetRule5Weight3(GReal_t rule5Weight3)
	{
		fRule5Weight3 = rule5Weight3;
	}

	GReal_t GetRule5Weight4() const
	{
		return fRule5Weight4;
	}

	void SetRule5Weight4(GReal_t rule5Weight4)
	{
		fRule5Weight4 = rule5Weight4;
	}

	GReal_t GetRule7Weight1() const
	{
		return fRule7Weight1;
	}

	void SetRule7Weight1(GReal_t rule7Weight1)
	{
		fRule7Weight1 = rule7Weight1;
	}

	GReal_t GetRule7Weight2() const
	{
		return fRule7Weight2;
	}

	void SetRule7Weight2(GReal_t rule7Weight2)
	{
		fRule7Weight2 = rule7Weight2;
	}

	GReal_t GetRule7Weight3() const
	{
		return fRule7Weight3;
	}

	void SetRule7Weight3(GReal_t rule7Weight3)
	{
		fRule7Weight3 = rule7Weight3;
	}

	GReal_t GetRule7Weight4() const
	{
		return fRule7Weight4;
	}

	void SetRule7Weight4(GReal_t rule7Weight4)
	{
		fRule7Weight4 = rule7Weight4;
	}

	GReal_t GetRule7Weight5() const
	{
		return fRule7Weight5;
	}

	void SetRule7Weight5(GReal_t rule7Weight5)
	{
		fRule7Weight5 = rule7Weight5;
	}

	private:

		template<size_t N=0>
		constexpr typename std::enable_if< (N==0), GULong64_t>::type twoN()
		{
			return 1;
		}

		template<size_t N=0>
		constexpr typename std::enable_if< (N>0), GULong64_t>::type twoN()
		{
			return 2*twoN<N-1>();
		}

		void set_abscissas()
		{
			for(unsigned int odr=0; odr<6; odr++)
			add_abscissas<GReal_t, DIM>(odr);
		}

		template<typename T, size_t N=1>
		void permute_abscissas(GReal_t rule5_weight, GReal_t rule7_weight,
				std::array<T, N> const& seed, vector_abscissa_t& container)
		{

			auto abscissa_temp = seed;
			std::sort(abscissa_temp.begin(), abscissa_temp.end());
			do{

				auto abscissa_tuple =
					thrust::tuple_cat(thrust::make_tuple(rule5_weight,rule7_weight),
						hydra::detail::arrayToTuple(abscissa_temp));

				container.push_back(abscissa_tuple);

			}while(std::next_permutation(abscissa_temp.begin(), abscissa_temp.end()));

		}


		template<typename T, size_t N=1>
		void add_abscissas( unsigned int order)
		{
		  typedef std::array<T, N> X_t;

		  switch( order )
		  {
		  case 0:
		  {

			  abscissa_t x;
			  thrust::get<0>(x)= fRule5Weight1;
			  thrust::get<1>(x)= fRule7Weight1;
			  fAbscissas.push_back(x);
			  break;
		  }
			  // order = 2
		  case 2:
		  {
			  auto x = X_t();
			  x[0]= T(fLambda2);
			  permute_abscissas(fRule5Weight2, fRule7Weight2, x,  fAbscissas);
			  x[0]= T(-fLambda2);
			  permute_abscissas(fRule5Weight2, fRule7Weight2, x,  fAbscissas);

			  break;
		  }
			  // order = 3
		  case 3:
		  {
			  auto  x = X_t();
			  x[0]= T(fLambda3);
			  permute_abscissas(fRule5Weight3, fRule7Weight3, x,  fAbscissas);
			  x[0]= T(-fLambda3);
			  permute_abscissas(fRule5Weight3, fRule7Weight3, x,  fAbscissas);

			  break;
		  }
			  // order = 4
		  case 4:
		  {
			  auto 	  x = X_t();
			  x[0]= T(fLambda4); x[1]= T(fLambda4);
			  permute_abscissas(fRule5Weight4, fRule7Weight4, x,  fAbscissas);

			  x[0]= T(-fLambda4); x[1]= T(-fLambda4);
			  permute_abscissas(fRule5Weight4, fRule7Weight4, x,  fAbscissas);

			  x[0]= T(fLambda4); x[1]= T(-fLambda4);
			  permute_abscissas(fRule5Weight4, fRule7Weight4, x,  fAbscissas);

			  break;
		  }
			  // order = 5
		  case 5:
		  {
			  auto 	  x = X_t();
			  for(size_t i=0;i<N+1;i++)
			  {
				  for(size_t j=0;j<N;j++)
				  {
					  x[j]= T(j<i?-fLambda5:fLambda5);
				  }
				  permute_abscissas(0.0, fRule7Weight5, x,  fAbscissas);
			  }

			  break;
		  }
		  default:
			  break;

		  }
		}

		GReal_t fLambda2;
		GReal_t fLambda3;
		GReal_t fLambda4;
		GReal_t fLambda5;

		GReal_t fRule7Weight1;
		GReal_t fRule7Weight2;
		GReal_t fRule7Weight3;
		GReal_t fRule7Weight4;
		GReal_t fRule7Weight5;

		GReal_t fRule5Weight1;
		GReal_t fRule5Weight2;
		GReal_t fRule5Weight3;
		GReal_t fRule5Weight4;

		vector_abscissa_t fAbscissas;

	};

}  // namespace experimental

} // namespace hydra

#endif /* GENZMALIKRULE_H_ */
