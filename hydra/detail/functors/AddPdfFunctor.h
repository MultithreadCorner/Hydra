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
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/Pdf.h>

namespace hydra {

namespace detail {

//evaluate a tuple of functors and return a tuple of results
template< typename Tup, size_t ... index>
inline auto get_functor_tuple_helper(Tup& pdfs, index_sequence<index...>)
-> decltype(HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(pdfs).GetFunctor()...))
{
	return HYDRA_EXTERNAL_NS::thrust::make_tuple(HYDRA_EXTERNAL_NS::thrust::get<index>(pdfs).GetFunctor() ...);
}

template< typename Tup>
inline auto get_functor_tuple(Tup& pdfs)
-> decltype(get_functor_tuple_helper(pdfs, make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value> { }))
{
	constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tup>::value;
	return get_functor_tuple_helper(pdfs, make_index_sequence<Size> { });
}

template<typename PDF1, typename PDF2, typename ...PDFs>
struct AddPdfFunctor
{

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<
			typename PDF1::functor_type,
			typename PDF2::functor_type,
			typename PDFs::functor_type...> functors_tuple_type;

	constexpr static size_t npdfs = sizeof...(PDFs)+2;

	AddPdfFunctor()=delete;

	AddPdfFunctor(HYDRA_EXTERNAL_NS::thrust::tuple<typename PDF1::functor_type,
			typename PDF2::functor_type, typename PDFs::functor_type...> const& functors,
			const Parameter (&coeficients)[sizeof...(PDFs)+2], GReal_t coef_sum):
				fFunctors( functors ),
				fCoefSum(coef_sum)
	{
		for(size_t i=0; i<sizeof...(PDFs)+2;i++)
			fCoeficients[i]=coeficients[i].GetValue();
	}

    __hydra_host__ __hydra_device__
	AddPdfFunctor(AddPdfFunctor< PDF1, PDF2, PDFs...> const& other ):
		fFunctors( other.GetFunctors() ),
		fCoefSum( other.GetCoefSum() )
	{
		for(size_t i=0; i<sizeof...(PDFs)+2;i++)
			fCoeficients[i]=other.GetCoeficients()[i];
	}


    __hydra_host__ __hydra_device__
    AddPdfFunctor< PDF1, PDF2, PDFs...>&
    operator=(AddPdfFunctor< PDF1, PDF2, PDFs...> const& other )
    {
    	this->fFunctors = other.GetFunctors() ;
    	this->fCoefSum = other.GetCoefSum() ;

    	for(size_t i=0; i<sizeof...(PDFs)+2;i++)
    		this->fCoeficients[i]=other.GetCoeficients()[i];

    	return *this;
    }

    __hydra_host__
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
    __hydra_host__ __hydra_device__
	const GReal_t* GetCoeficients() const
	{
		return fCoeficients;
	}
    __hydra_host__ __hydra_device__
	GReal_t GetCoefSum() const
	{
		return fCoefSum;
	}
    __hydra_host__ __hydra_device__
	void SetCoefSum(GReal_t coefSum)
	{
		fCoefSum = coefSum;
	}


	__hydra_host__ __hydra_device__  inline
	GReal_t GetNorm() const {
		return 1.0 ;
	}

    __hydra_host__ __hydra_device__
	const functors_tuple_type& GetFunctors() const
	{
		return fFunctors;
	}

    __hydra_host__ __hydra_device__
	void SetFunctors(functors_tuple_type functors)
	{
		fFunctors = functors;
	}


	template<typename T> inline
	 __hydra_host__ __hydra_device__
	GReal_t operator()(T&& t ) const
	{

		auto pdf_res_tuple = detail::invoke_normalized<functors_tuple_type, T>( t, fFunctors);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		//printf("%f %f %f %f %f  \n", pdf_res_array[0], pdf_res_array[1], pdf_res_array[2], result, fCoefSum );

		return result*fCoefSum;
	}

	template<typename T1, typename T2>
	 __hydra_host__ __hydra_device__
	inline	GReal_t operator()(T1&& x, T2&& cache) const
	{


		auto pdf_res_tuple = detail::invoke_normalized<GReal_t,functors_tuple_type, T1, T2>( x, cache, fFunctors);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		return result*fCoefSum;
	}

	template<typename T>
	 __hydra_host__ __hydra_device__
	inline	GReal_t operator()(GUInt_t , T *x)
	{


		auto pdf_res_tuple = detail::invoke_normalized(x, fFunctors);
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
	functors_tuple_type fFunctors;
};

}  // namespace detail

}  // namespace hydra

#endif /* ADDPDFFUNCTOR_H_ */
