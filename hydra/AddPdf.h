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
 * AddPdf.h
 *
 *  Created on: 03/09/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ADDPDF_H_
#define ADDPDF_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/Pdf.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/functors/AddPdfFunctor.h>
#include <thrust/tuple.h>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace hydra {

namespace detail {


template<typename PDF1, typename PDF2, typename ...PDFs>
class AddPdfChecker:  public all_true<
detail::is_hydra_pdf<PDF1>::value,
detail::is_hydra_pdf<PDF2>::value,
detail::is_hydra_pdf<PDFs>::value...>{} ;

template<typename PDF1, typename PDF2, typename ...PDFs>
class AddPdfBase: public std::enable_if<AddPdfChecker<PDF1,PDF2,PDFs...>::value>
{};

}  // namespace detail


/**
 * @class
 * @ingroup fit
 * @brief Build a pdf adding other pdfs.
 *
 * Given N unnormalized pdfs \f$F_i\f$ , this class define a object representing the sum
 * \f[ F_t = \sum_i^N c_i \times F_i \f]
 * The coefficients of the pdfs can represent fractions or yields. If the number of coefficients is equal to
 * the number of pdfs, the coefficients are interpreted as yields. If the number of coefficients is \f$(N-1)\f$,
 * the coefficients are interpreted as fractions defined in the interval [0,1].
 * The coefficient of the last term is calculated as \f$ c_N=1 -\sum_i^{(N-1)} c_i \f$.
 */
template<typename PDF1, typename PDF2, typename ...PDFs>
class AddPdf: public detail::AddPdfBase<PDF1,PDF2,PDFs...>
{


public:

//this typedef is actually a check. If the AddPdf is not built with
	//hydra::pdf, AddPdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::AddPdfBase<PDF1,PDF2,PDFs...>::type base_type; //!< base class type

	constexpr static size_t npdfs = sizeof...(PDFs)+2; //!< number of pdfs

	typedef thrust::tuple<PDF1, PDF2, PDFs...> pdfs_tuple_type;//!< type of the tuple of pdfs

	typedef thrust::tuple<typename PDF1::functor_type,
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
	AddPdf( PDF1 const& pdf1, PDF2 const& pdf2, PDFs const& ...pdfs,
			std::array<Parameter*, npdfs>const& coef, GBool_t extend=kTrue ):
			fPDFs(thrust::make_tuple(pdf1,pdf2,pdfs...) ),
			fFunctors(thrust::make_tuple(pdf1.GetFunctor(),pdf2.GetFunctor(),pdfs.GetFunctor() ...) ),
			fExtended(kTrue),
			fFractioned(kFalse),
			fCoefSum(0.0)
	{
		size_t i=0;
		for(Parameter* var:coef){
			fCoeficients[i] = *var;
			fCoefSum += var->GetValue();
			i++;
		}

	}

	/**
	 * \brief Ctor for used to build AddPdf __not-usable__ in extended likelihood fits.
	 * \param pdf1 first pdf object,
	 * \param pdf2 second pdf object,
	 * \param pdfs remaining pdfs.
	 * \param coef arrary of Parameters, each parameter correspond to a fractional coefficient
	 *
	 * Each component pdf is normalized properly before evaluation each time SetParameters(const std::vector<double>& parameters) is called.
	 * The sum is normalized also.
	 */
	AddPdf(PDF1 const& pdf1, PDF2 const& pdf2, PDFs const& ...pdfs,
			std::array<Parameter*, npdfs-1>const& coef):
			fPDFs(thrust::make_tuple(pdf1,pdf2,pdfs...) ),
			fFunctors(thrust::make_tuple(pdf1.GetFunctor(),pdf2.GetFunctor(),pdfs.GetFunctor() ...) ),
			fExtended(kFalse),
			fFractioned(kTrue),
			fCoefSum(0.0)
	{
		size_t i=0;
		for(Parameter* var:coef)
		{
			if(var->GetLowerLim()< 0.0 || var->GetUpperLim() > 1.0 ||
					var->GetValue()< 0.0 || var->GetValue() > 1.0)
			{
				HYDRA_LOG(WARNING,"Fraction out of bounds" )
				HYDRA_MSG << (*var) << HYDRA_ENDL;
				HYDRA_MSG << "Change the parameter limits to [0.0, 1.0] and value accordingly."<< HYDRA_ENDL;
				 abort();
			}

			fCoeficients[i]= *var;
			fCoefSum+= var->GetValue();
			i++;
		}

		fCoeficients[npdfs-1]= 1.0 - fCoefSum;
	}

	/**
	 * \brief Copy constructor.
	 */
	AddPdf(AddPdf<PDF1, PDF2, PDFs...> const& other ):
	fPDFs(other.GetPdFs() ),
	fFunctors(other.GetFunctors() ),
	fExtended(other.IsExtended()),
	fCoefSum(other.GetCoefSum()),
	fFractioned(other.IsFractioned())
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
	AddPdf<PDF1, PDF2, PDFs...>&
	operator=( AddPdf<PDF1, PDF2, PDFs...> const& other )
	{
		this->fFunctors= other.GetFunctors() ;
		this->fPDFs = other.GetPdFs();
		this->fExtended = other.IsExtended();
		this->fCoefSum= other.GetCoefSum();
		for( size_t i=0; i< npdfs; i++ ){
			this->fCoeficients[i]=other.GetCoeficient(i);
		}

		return *this;
	}


	/**
	 * \brief Set the coefficients and parameters of all pdfs.
	 * This method sets the values of all coefficients and parameters of pdfs stored in the AddPdf object.
	 * User should ensure this method is called before the object evaluation.
	 * @param parameters std::vector<double> containing the list of parameters passed by ROOT::Minuit2.
	 */
	inline	void SetParameters(const std::vector<double>& parameters){

		for(size_t i=0; i< npdfs-(fFractioned); i++)
			      fCoeficients[i].Reset(parameters );

		detail::set_functors_in_tuple(fPDFs, parameters);

		fFunctors = get_functor_tuple(fPDFs);

		fCoefSum=0;

		if(!fFractioned){
			for(size_t i=0; i< npdfs ; i++)
				fCoefSum+=fCoeficients[i];
		}
		else
		{
			for(size_t i=0; i< npdfs-1 ; i++)
							fCoefSum+=fCoeficients[i];

			fCoeficients[npdfs -1] = 1.0 - fCoefSum;
		}
	}

	/**
	 * \brief Print all registered parameters, including its value, range, name etc.
	 */
	inline	void PrintRegisteredParameters()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		if(!fFractioned){
			for(size_t i=0; i< npdfs ; i++)
				HYDRA_MSG <<fCoeficients[i]<< HYDRA_ENDL;
		}
		else
		{
			for(size_t i=0; i< npdfs-1 ; i++)
				HYDRA_MSG <<fCoeficients[i]<< HYDRA_ENDL;
		}
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
		if(!fFractioned){
			for(size_t i=0; i< npdfs ; i++)
				user_parameters.push_back(&fCoeficients[i]);
		}
		else
		{
			for(size_t i=0; i< npdfs-1 ; i++)
				user_parameters.push_back(&fCoeficients[i]);

		}

		detail::add_parameters_in_tuple(user_parameters, fPDFs);
	}


	inline	const Parameter& GetCoeficient(size_t i) const
	{
		return fCoeficients[i];
	}

	inline	GBool_t IsExtended() const
	{
		return fExtended;
	}




	inline detail::AddPdfFunctor< PDF1, PDF2, PDFs...>  GetFunctor() const
	{


		return detail::AddPdfFunctor<PDF1, PDF2, PDFs...>(fFunctors,
				fCoeficients,1.0/fCoefSum,fExtended,fFractioned );
	}


	inline const functors_tuple_type& GetFunctors() const
	{
			return fFunctors;
	}


	inline const pdfs_tuple_type& GetPdFs() const
	{
		return fPDFs;
	}

	inline	GReal_t GetCoefSum() const
	{
		return fCoefSum;
	}

	inline	GBool_t IsFractioned() const
	{
		return fFractioned;
	}

	void SetExtended(GBool_t extended)
	{
		fExtended = extended;
	}


	template<typename T1> inline
	GReal_t operator()(T1&& t )
	{
		auto pdf_res_tuple = detail::invoke<pdfs_tuple_type, T1>( t, fPDFs);
		GReal_t pdf_res_array[npdfs];
		detail::tupleToArray( pdf_res_tuple, pdf_res_array );

		GReal_t result = 0;
		for(size_t i=0; i< npdfs; i++)
			result += fCoeficients[i]*pdf_res_array[i];

		return result/fCoefSum;
	}


	template<typename T1, typename T2>
	inline	GReal_t operator()( T1&& t, T2&& cache)
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
    GReal_t     fCoefSum;
	Parameter    fCoeficients[npdfs];
	pdfs_tuple_type fPDFs;
	functors_tuple_type fFunctors;
	GBool_t fExtended;
	GBool_t fFractioned;

};

/**
 *\brief Convenience function to add pdfs without set template parameters explicitly.
 */
template<size_t N, typename PDF1, typename PDF2, typename ...PDFs>
AddPdf<PDF1, PDF2, PDFs...>
add_pdfs(std::array<Parameter*, N>const& var_list, PDF1 const& pdf1, PDF2 const& pdf2, PDFs const& ...pdfs )
{
	return AddPdf<PDF1, PDF2, PDFs...>(pdf1, pdf2, pdfs..., var_list);
}

template<unsigned int I, typename PDF1, typename PDF2, typename ...PDFs>
auto get_pdf( AddPdf<PDF1, PDF2, PDFs...> const& pdfs)
-> std::pair< Parameter, typename thrust::tuple_element<I,
typename AddPdf<PDF1, PDF2, PDFs...>::pdfs_tuple_type>::type  >
{
	return	std::make_pair( pdfs.GetCoeficient(I), thrust::get<I>(pdfs.GetPdFs()));
}

}  // namespace hydra


#endif /* ADDPDF_H_ */
