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
 * PDFSumNonExtendable.h
 *
 *  Created on: 08/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PDFSUMNONEXTENDABLE_H_
#define PDFSUMNONEXTENDABLE_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/Pdf.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/functors/AddPdfFunctor.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace hydra {

/**
 * \ingroup fit
 * \brief Class representing a pdf object built summing a pdf adding other pdfs.
 *
 * Given N unnormalized pdfs \f$F_i\f$ , this class define a object representing the sum
 * \f[ F_t = \sum_i^N c_i \times F_i \f]
 * The coefficients of the pdfs can represent fractions or yields. If the number of coefficients is equal to
 * the number of pdfs, the coefficients are interpreted as yields. If the number of coefficients is \f$(N-1)\f$,
 * the coefficients are interpreted as fractions defined in the interval [0,1].
 * The coefficient of the last term is calculated as \f$ c_N=1 -\sum_i^{(N-1)} c_i \f$.
 */
template<typename PDF1, typename PDF2, typename ...PDFs>
class PDFSumNonExtendable: public detail::AddPdfBase<PDF1,PDF2,PDFs...>
{


public:
typedef void hydra_pdf_tag;
//this typedef is actually a check. If the PDFSumNonExtendable is not built with
	//hydra::pdf, AddPdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::AddPdfBase<PDF1,PDF2,PDFs...>::type base_type; //!< base class type

	constexpr static size_t npdfs = sizeof...(PDFs)+2; //!< number of pdfs

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<PDF1, PDF2, PDFs...> pdfs_tuple_type;//!< type of the tuple of pdfs

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<typename PDF1::functor_type,
			typename  PDF2::functor_type,
			typename  PDFs::functor_type...> functors_tuple_type;//!< type of the tuple of pdf::functors

	typedef detail::AddPdfFunctor< PDF1, PDF2, PDFs...> functor_type;

	//tag
	typedef void hydra_sum_pdf_tag; //!< tag

	/**
	 * \brief Ctor for used to build AddPdf usable in extended likelihood fits.
	 * \param pdf1 first pdf object,
	 * \param pdf2 second pdf object,
	 * \param pdfs remaining pdfs.
	 * \param coef arrary of Parameters, each parameter correspond to a coefficient
	 * \param extend build pdf to be used in extended fit. Default is true.
	 *
	 * Each component pdf is normalized properly before evaluation each time SetParameters(const std::vector<double>& parameters) is called.
	 * The sum is normalized also.
	 */
	PDFSumNonExtendable( PDF1 const& pdf1, PDF2 const& pdf2, PDFs const& ...pdfs,
			std::array<Parameter, npdfs-1>const& coef ):
			fPDFs(HYDRA_EXTERNAL_NS::thrust::make_tuple(pdf1,pdf2,pdfs...) ),
			fFunctors(HYDRA_EXTERNAL_NS::thrust::make_tuple(pdf1.GetFunctor(),pdf2.GetFunctor(),pdfs.GetFunctor() ...) ),
			fCoefSum(1.0),
			fCoefPartialSum(0.0)
	{
		double fraction=1.0;
		size_t i=0;
		for(Parameter var:coef){
			fCoeficients[i] = var;
			fraction -= var.GetValue();
			fCoefPartialSum += var.GetValue();
			i++;
		}

		fCoeficients[npdfs-1].Name("1.0 - @sum_i{c_i}")
							 .Value(fraction)
							 .Error(0.000)
							 .Limits(0.0, 1.0);
	}


	/**
	 * \brief Copy constructor.
	 */
	PDFSumNonExtendable(PDFSumNonExtendable<PDF1, PDF2, PDFs...> const& other ):
	fPDFs(other.GetPDFs() ),
	fFunctors(other.GetFunctors() ),
	fCoefSum(other.GetCoefSum()),
	fCoefPartialSum(other.GetCoefPartialSum())
	{
		for( size_t i=0; i< npdfs; i++ ){
			fCoeficients[i]=other.GetCoeficient(i);
			}
	}


	/**
	 * \brief Assignment operator.
	 * @param other
	 * @return
	 */
	PDFSumNonExtendable<PDF1, PDF2, PDFs...>&
	operator=( PDFSumNonExtendable<PDF1, PDF2, PDFs...> const& other )
	{
		this->fFunctors= other.GetFunctors() ;
		this->fPDFs = other.GetPDFs();
		this->fCoefSum= other.GetCoefSum();
		this->fCoefPartialSum = other.GetCoefPartialSum();

		for( size_t i=0; i< npdfs; i++ ){
			this->fCoeficients[i]=other.GetCoeficient(i);
		}

		return *this;
	}


	/**
	 * \brief Set the coefficients and parameters of all pdfs.
	 * This method sets the values of all coefficients and parameters of pdfs stored in the PDFSumNonExtendable object.
	 * User should ensure this method is called before the object evaluation.
	 * @param parameters std::vector<double> containing the list of parameters passed by ROOT::Minuit2.
	 */
	inline	void SetParameters(const std::vector<double>& parameters){

		fCoeficients[ npdfs-1] = 1.0;
		fCoefPartialSum=0.0;
		for(size_t i=0; i< npdfs-1; i++){
			      fCoeficients[i].Reset(parameters );
			      fCoeficients[ npdfs-1] -= fCoeficients[i].GetValue();
			      fCoefPartialSum += fCoeficients[i].GetValue();
		}

		detail::set_functors_in_tuple(fPDFs, parameters);

		fFunctors = get_functor_tuple(fPDFs);

		fCoefSum=0;

		for(size_t i=0; i< npdfs ; i++)
				fCoefSum+=fCoeficients[i];
	}

	/**
	 * \brief Print all registered parameters, including its value, range, name etc.
	 */
	inline	void PrintRegisteredParameters()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;

		for(size_t i=0; i< npdfs ; i++)
			HYDRA_MSG <<fCoeficients[i]<< HYDRA_ENDL;

		detail::print_parameters_in_tuple(fPDFs);
		HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
		return;
	}


	/**
	 * \brief Add pointers to the functor's parameters to a external list, that will be used later to
	 * build the hydra::UserParameters instance that will be passed to ROOT::Minuit2.
	 *
	 * @param user_parameters external std::vector<hydra::Parameter*> object holding the list of pointers
	 * to functor parameters.
	 */
	inline	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
	{
		for(size_t i=0; i< npdfs-1; i++)
			user_parameters.push_back(&fCoeficients[i]);

		detail::add_parameters_in_tuple(user_parameters, fPDFs);
	}


	inline	const Parameter& GetCoeficient(size_t i) const
	{
		return fCoeficients[i];
	}

	inline	Parameter& Coeficient(size_t i)
	{
		return fCoeficients[i];
	}

	inline detail::AddPdfFunctor< PDF1, PDF2, PDFs...>  GetFunctor() const
	{
		return detail::AddPdfFunctor<PDF1, PDF2, PDFs...>(fFunctors,fCoeficients,1.0/fCoefSum );
	}

	template<unsigned int I>
		typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I,pdfs_tuple_type>::type&
		PDF( hydra::placeholders::placeholder<I>  ){

			return HYDRA_EXTERNAL_NS::thrust::get<I>(fPDFs);
		}

	inline const functors_tuple_type& GetFunctors() const
	{
			return fFunctors;
	}


	inline const pdfs_tuple_type& GetPDFs() const
	{
		return fPDFs;
	}

	inline	GReal_t GetCoefSum() const
	{
		return fCoefSum;
	}

	inline	GReal_t GetCoefPartialSum() const
		{
			return fCoefPartialSum;
		}


	template<typename T1> inline
	GReal_t operator()(T1&& t ) const
	{
		auto pdf_res_tuple = detail::invoke( t, fPDFs);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		return result/fCoefSum;
	}


	template<typename T1, typename T2>
	inline	GReal_t operator()( T1&& t, T2&& cache) const
	{

		auto pdf_res_tuple = detail::invoke<GReal_t,pdfs_tuple_type, T1, T2>( t, cache, fPDFs);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		return result/fCoefSum;
	}

	template<typename T>
    inline	GReal_t operator()( T* x, T* p)
	{

		auto pdf_res_tuple = detail::invoke<GReal_t,pdfs_tuple_type, T*, T*>( x, p, fPDFs);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;

		for(size_t i=0; i< npdfs; i++)
		{
			result += fCoeficients[i]*pdf_res_array[i];
		}

		return result/fCoefSum;
	}



private:

    GReal_t      fCoefSum;
    GReal_t      fCoefPartialSum;
	Parameter    fCoeficients[npdfs];
	pdfs_tuple_type fPDFs;
	functors_tuple_type fFunctors;

};

/**
 * \ingroup fit
 * \brief Function to build up non-extendable pdfs models.
 *
 * Given N unnormalized pdfs \f$F_i\f$ , this class define a object representing the sum
 * \f[ F_t = \sum_i^N c_i \times F_i \f]
 * The coefficients of the pdfs can represent fractions or yields. If the number of coefficients is equal to
 * the number of pdfs, the coefficients are interpreted as yields. If the number of coefficients is \f$(N-1)\f$,
 * the coefficients are interpreted as fractions defined in the interval [0,1].
 * The coefficient of the last term is calculated as \f$ c_N=1 -\sum_i^{(N-1)} c_i \f$.
 *
 * @param var_list std::array<hydra::Parameter, N > holding the (N_{Pdf} -1) hydra::Parameter yields
 * @param pdf1 hydra::Pdf object one
 * @param pdf2 hydra::Pdf object two
 * @param pdfs ... more pdfs
 * @return
 */
template<typename PDF1, typename PDF2, typename ...PDFs>
PDFSumNonExtendable<PDF1, PDF2, PDFs...>
add_pdfs(std::array<Parameter,sizeof...(PDFs)+1 >const& var_list, PDF1 const& pdf1, PDF2 const& pdf2, PDFs const& ...pdfs )
{
	return PDFSumNonExtendable<PDF1, PDF2, PDFs...>(pdf1, pdf2, pdfs..., var_list);
}



}  // namespace hydra


#endif /* PDFSUMNONEXTENDABLE_H_ */
