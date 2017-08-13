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
 * AddPdfFunctor.h
 *
 *  Created on: 30/11/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ADDPDFFUNCTOR_H_
#define ADDPDFFUNCTOR_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <thrust/tuple.h>
#include <hydra/Pdf.h>

namespace hydra {

namespace detail {

//evaluate a tuple of functors and return a tuple of results
template< typename Tup, size_t ... index>
inline auto get_functor_tuple_helper(Tup& pdfs, index_sequence<index...>)
-> decltype(thrust::make_tuple(thrust::get<index>(pdfs).GetFunctor()...))
{
	return thrust::make_tuple(thrust::get<index>(pdfs).GetFunctor() ...);
}

template< typename Tup>
inline auto get_functor_tuple(Tup& pdfs)
-> decltype(get_functor_tuple_helper(pdfs, make_index_sequence< thrust::tuple_size<Tup>::value> { }))
{
	constexpr size_t Size = thrust::tuple_size<Tup>::value;
	return get_functor_tuple_helper(pdfs, make_index_sequence<Size> { });
}

template<typename PDF1, typename PDF2, typename ...PDFs>
struct AddPdfFunctor
{

	typedef thrust::tuple<
			typename PDF1::functor_type,
			typename PDF2::functor_type,
			typename PDFs::functor_type...> functors_tuple_type;

	constexpr static size_t npdfs = sizeof...(PDFs)+2;

	AddPdfFunctor()=delete;

	AddPdfFunctor(thrust::tuple<typename PDF1::functor_type,
			typename PDF2::functor_type, typename PDFs::functor_type...> const& functors,
			const Parameter (&coeficients)[sizeof...(PDFs)+2],
			GReal_t coef_sum, GBool_t extended,	GBool_t fractioned ):
				fFunctors( functors ),
				fExtended(extended),
				fFractioned(fractioned),
				fCoefSum(coef_sum)
	{
		for(size_t i=0; i<sizeof...(PDFs)+2;i++)
			fCoeficients[i]=coeficients[i].GetValue();
	}

    __host__ __device__
	AddPdfFunctor(AddPdfFunctor< PDF1, PDF2, PDFs...> const& other ):
		fFunctors( other.GetFunctors() ),
		fExtended( other.IsExtended() ),
		fFractioned( other.IsFractioned() ),
		fCoefSum( other.GetCoefSum() )
	{
		for(size_t i=0; i<sizeof...(PDFs)+2;i++)
			fCoeficients[i]=other.GetCoeficients()[i];
	}


    __host__ __device__
    AddPdfFunctor< PDF1, PDF2, PDFs...>&
    operator=(AddPdfFunctor< PDF1, PDF2, PDFs...> const& other )
    {
    	this->fFunctors = other.GetFunctors() ;
    	this->fExtended = other.IsExtended() ;
    	this->fFractioned = other.IsFractioned() ;
    	this->fCoefSum = other.GetCoefSum() ;

    	for(size_t i=0; i<sizeof...(PDFs)+2;i++)
    		this->fCoeficients[i]=other.GetCoeficients()[i];

    	return *this;
    }

    __host__
    void PrintRegisteredParameters()
    {
    	HYDRA_CALLER ;
    	HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
    	HYDRA_MSG << "Coeficients: "<< HYDRA_ENDL;
    			for(size_t i=0;i< sizeof...(PDFs)+2;i++ )
    			{
    				HYDRA_MSG << "["<<i<<"]" <<	fCoeficients[i]<<HYDRA_ENDL;
    			}
    			detail::print_parameters_in_tuple(fFunctors);
    	HYDRA_MSG <<"Registered parameters end."<< HYDRA_ENDL;
    	HYDRA_MSG << HYDRA_ENDL;
    }
    __host__ __device__
	const GReal_t* GetCoeficients() const
	{
		return fCoeficients;
	}
    __host__ __device__
	GReal_t GetCoefSum() const
	{
		return fCoefSum;
	}
    __host__ __device__
	void SetCoefSum(GReal_t coefSum)
	{
		fCoefSum = coefSum;
	}
    __host__ __device__
	GBool_t IsExtended() const
	{
		return fExtended;
	}
    __host__ __device__
	void SetExtended(GBool_t extended)
	{
		fExtended = extended;
	}
    __host__ __device__
	GBool_t IsFractioned() const
	{
		return fFractioned;
	}
    __host__ __device__
	void SetFractioned(GBool_t fractioned)
	{
		fFractioned = fractioned;
	}
    __host__ __device__
	const functors_tuple_type& GetFunctors() const
	{
		return fFunctors;
	}
    __host__ __device__
	void SetFunctors(functors_tuple_type functors)
	{
		fFunctors = functors;
	}


	template<typename T> inline
	 __host__ __device__
	GReal_t operator()(T&& t )
	{

		auto pdf_res_tuple = detail::invoke<functors_tuple_type, T>( t, fFunctors);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		//printf("%f %f %f %f %f  \n", pdf_res_array[0], pdf_res_array[1], pdf_res_array[2], result, fCoefSum );

		return result*fCoefSum;
	}

	template<typename T1, typename T2>
	 __host__ __device__
	inline	GReal_t operator()(T1&& x, T2&& cache)
	{


		auto pdf_res_tuple = detail::invoke<GReal_t,functors_tuple_type, T1, T2>( x, cache, fFunctors);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		return result*fCoefSum;
	}

	template<typename T>
	 __host__ __device__
	inline	GReal_t operator()(GUInt_t n, T *x)
	{


		auto pdf_res_tuple = detail::invoke(x, fFunctors);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;

		for(size_t i=0; i< npdfs; i++)
		{
			result += fCoeficients[i]*pdf_res_array[i];
		}

		return result*fCoefSum;
	}


private:
	GReal_t fCoefSum;
	GReal_t fCoeficients[sizeof...(PDFs)+2];
	GBool_t fExtended;
	GBool_t fFractioned;
	functors_tuple_type fFunctors;
};

}  // namespace detail

}  // namespace hydra

#endif /* ADDPDFFUNCTOR_H_ */
