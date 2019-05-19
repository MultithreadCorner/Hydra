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
 * GenzMalikRule.h
 *
 *  Created on: 22/02/2017
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef GENZMALIKRULE_H_
#define GENZMALIKRULE_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/multivector.h>
//thrust
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>
//std
#include <type_traits>
#include <array>
#include <initializer_list>
#include <math.h>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>

namespace hydra {


template<typename T>
class GenzMalikRuleBase{};

template<size_t DIM, typename BACKEND>
class GenzMalikRule;

/**
 * \ingroup numerical_integration
 *
 * \brief Class representing Genz-Malik rule.
 *
 */
template<size_t DIM, hydra::detail::Backend   BACKEND>
class GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND>>:
           GenzMalikRuleBase<typename std::enable_if< (DIM>1), void >::type >
	{


	typedef typename hydra::detail::tuple_type<DIM+2,GChar_t >::type char_abscissa_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple<GReal_t,GReal_t, GReal_t> real_abscissa_t;
	typedef typename hydra::detail::tuple_cat_type<real_abscissa_t, char_abscissa_t>::type abscissa_t;

	typedef  hydra::detail::BackendPolicy<BACKEND> system_t;


public:

	//container
	typedef multivector< abscissa_t, hydra::detail::BackendPolicy<BACKEND>> vector_abscissa_t;
	typedef typename vector_abscissa_t::iterator abscissa_iterator;
	typedef typename vector_abscissa_t::const_iterator const_abscissa_iterator;

	enum AbscissaCategory_t
	{ Central = 0, FirstLeft, SecondLeft, FirstRight, SecondRight, Multidimensional };


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
			fRule7Weight5= GReal_t(6859.0/19683.0);

			fRule5Weight1= factor*GReal_t(729.0 - 950.0*DIM + 50.0*DIM*DIM)/729.0;
			fRule5Weight2= factor*GReal_t(245.0/486.0);
			fRule5Weight3= factor*GReal_t(265.0 - 100.0*DIM)/1458.0;
			fRule5Weight4= factor*GReal_t(25.0/729.0);
			fAbscissas.reserve( hydra::detail::power<2,DIM>::value
					+ 2*hydra::detail::power<DIM,2>::value
					+ 2*DIM +1);

			set_abscissas();



			}


		GenzMalikRule( GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND>> const& other):
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

		template<hydra::detail::Backend BACKEND2>
		GenzMalikRule( GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND2>> const& other):
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

		GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>>&
		operator=( GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>> const& other)
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

		template<hydra::detail::Backend BACKEND2>
		inline GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>>&
		operator=( GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND2>> const& other)
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

	inline const vector_abscissa_t& GetAbscissas() const
	{
		return fAbscissas;
	}

	inline void SetAbscissas(const vector_abscissa_t& abscissas)
	{
		fAbscissas = abscissas;
	}

	inline GReal_t GetLambda2() const
	{
		return fLambda2;
	}

	inline void SetLambda2(GReal_t lambda2)
	{
		fLambda2 = lambda2;
	}

	inline GReal_t GetLambda3() const
	{
		return fLambda3;
	}

	inline void SetLambda3(GReal_t lambda3)
	{
		fLambda3 = lambda3;
	}

	inline GReal_t GetLambda4() const
	{
		return fLambda4;
	}

	inline void SetLambda4(GReal_t lambda4)
	{
		fLambda4 = lambda4;
	}

	inline GReal_t GetLambda5() const
	{
		return fLambda5;
	}

	inline void SetLambda5(GReal_t lambda5)
	{
		fLambda5 = lambda5;
	}

	inline GReal_t GetRule5Weight1() const
	{
		return fRule5Weight1;
	}

	inline void SetRule5Weight1(GReal_t rule5Weight1)
	{
		fRule5Weight1 = rule5Weight1;
	}

	inline GReal_t GetRule5Weight2() const
	{
		return fRule5Weight2;
	}

	inline void SetRule5Weight2(GReal_t rule5Weight2)
	{
		fRule5Weight2 = rule5Weight2;
	}

	inline GReal_t GetRule5Weight3() const
	{
		return fRule5Weight3;
	}

	inline void SetRule5Weight3(GReal_t rule5Weight3)
	{
		fRule5Weight3 = rule5Weight3;
	}

	inline GReal_t GetRule5Weight4() const
	{
		return fRule5Weight4;
	}

	inline void SetRule5Weight4(GReal_t rule5Weight4)
	{
		fRule5Weight4 = rule5Weight4;
	}

	inline GReal_t GetRule7Weight1() const
	{
		return fRule7Weight1;
	}

	inline void SetRule7Weight1(GReal_t rule7Weight1)
	{
		fRule7Weight1 = rule7Weight1;
	}

	inline GReal_t GetRule7Weight2() const
	{
		return fRule7Weight2;
	}

	inline void SetRule7Weight2(GReal_t rule7Weight2)
	{
		fRule7Weight2 = rule7Weight2;
	}

	inline GReal_t GetRule7Weight3() const
	{
		return fRule7Weight3;
	}

	inline void SetRule7Weight3(GReal_t rule7Weight3)
	{
		fRule7Weight3 = rule7Weight3;
	}

	inline GReal_t GetRule7Weight4() const
	{
		return fRule7Weight4;
	}

	inline void SetRule7Weight4(GReal_t rule7Weight4)
	{
		fRule7Weight4 = rule7Weight4;
	}

	inline GReal_t GetRule7Weight5() const
	{
		return fRule7Weight5;
	}

	inline void SetRule7Weight5(GReal_t rule7Weight5)
	{
		fRule7Weight5 = rule7Weight5;
	}

	private:

		template<size_t N=0>
		inline   typename std::enable_if< (N==0), GULong64_t>::type twoN()
		{
			return 1;
		}

		template<size_t N=0>
		inline typename std::enable_if< (N>0), GULong64_t>::type twoN()
		{
			return 2*twoN<N-1>();
		}

		inline void set_abscissas()
		{
			for(unsigned int odr=0; odr<6; odr++)
			add_abscissas<GChar_t, DIM>(odr);
		}

		template<typename T, size_t N=1>
		inline void permute_abscissas( AbscissaCategory_t category, GReal_t rule5_weight, GReal_t rule7_weight, GReal_t lambda,
			 std::array<T, N> const& seed, vector_abscissa_t& container)
		{

			auto abscissa_temp = seed;

			std::sort(abscissa_temp.begin(), abscissa_temp.end());

			do{
				GChar_t index = -1;
				GChar_t four_difference_weight = 0;
				switch (category) {

				case Central:
					index = N;
					four_difference_weight = 6;
					break;

				case FirstRight:
					index = std::distance( abscissa_temp.begin(), std::max_element ( abscissa_temp.begin(), abscissa_temp.end())   );
					four_difference_weight = -4;
					break;

				case SecondRight:
					index = std::distance( abscissa_temp.begin(),  std::max_element ( abscissa_temp.begin(), abscissa_temp.end()) );
					four_difference_weight = 1;
					break;

				case FirstLeft:
					index = std::distance(  abscissa_temp.begin(), std::min_element( abscissa_temp.begin(), abscissa_temp.end()) );
					four_difference_weight = -4;
					break;

				case SecondLeft:
					index = std::distance(  abscissa_temp.begin(), std::min_element( abscissa_temp.begin(), abscissa_temp.end()) );
					four_difference_weight = 1;
					break;

				case Multidimensional:
					index = -1;
					four_difference_weight = 0;
					break;

				default:
					break;
				}


				auto abscissa_tuple =
					HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(rule5_weight,rule7_weight, lambda, four_difference_weight, index),
						hydra::detail::arrayToTuple(abscissa_temp));

				container.push_back(abscissa_tuple);


			}while(std::next_permutation(abscissa_temp.begin(), abscissa_temp.end()));

		}


		template<typename T, size_t N>
		inline void add_abscissas( unsigned int order)
		{
		  typedef std::array<T, N> X_t;

		  switch( order )
		  {
		  case 0:
		  {

			  abscissa_t x;
			  HYDRA_EXTERNAL_NS::thrust::get<0>(x)= fRule5Weight1;
			  HYDRA_EXTERNAL_NS::thrust::get<1>(x)= fRule7Weight1;
			  HYDRA_EXTERNAL_NS::thrust::get<2>(x)= 1.0;
			  HYDRA_EXTERNAL_NS::thrust::get<4>(x)= N;
			  HYDRA_EXTERNAL_NS::thrust::get<3>(x)= 6;
			  fAbscissas.push_back(x);
			  break;
		  }
			  // order = 2
		  case 2:
		  {
			  auto x = X_t();
			  x[0]= 1;
			  permute_abscissas( SecondRight, fRule5Weight2, fRule7Weight2, fLambda2, x,  fAbscissas);
			  x[0]= -1;
			  permute_abscissas( SecondLeft,fRule5Weight2, fRule7Weight2, fLambda2, x,  fAbscissas);

			  break;
		  }
			  // order = 3
		  case 3:
		  {
			  auto  x = X_t();
			  x[0]= 1;
			  permute_abscissas( FirstRight, fRule5Weight3, fRule7Weight3, fLambda3, x,  fAbscissas);
			  x[0]= -1;
			  permute_abscissas( FirstLeft, fRule5Weight3, fRule7Weight3, fLambda3, x,  fAbscissas);

			  break;
		  }
			  // order = 4
		  case 4:
		  {
			  auto 	  x = X_t();
			  x[0]= 1; x[1]= 1;
			  permute_abscissas(Multidimensional, fRule5Weight4, fRule7Weight4, fLambda4, x,  fAbscissas);

			  x[0]= -1; x[1]= -1;
			  permute_abscissas(Multidimensional,fRule5Weight4, fRule7Weight4, fLambda4, x,  fAbscissas);

			  x[0]= 1; x[1]= -1;
			  permute_abscissas(Multidimensional, fRule5Weight4, fRule7Weight4,  fLambda4, x,  fAbscissas);

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
					  x[j]= T(j<i?-1:1);
				  }
				  permute_abscissas(Multidimensional, 0.0, fRule7Weight5, fLambda5, x,  fAbscissas);
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

} // namespace hydra

#endif /* GENZMALIKRULE_H_ */
