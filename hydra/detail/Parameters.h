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
 * Parameters.h
 *
 *  Created on: 22/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <assert.h>

namespace hydra {

namespace detail {

template<size_t N>
class Parameters{

public:
	static const size_t parameter_count =N;

	Parameters ()=delete;

	Parameters(std::initializer_list<hydra::Parameter> init_parameters)
	{
		assert(init_parameters.size()==N && "HYDRA MESSAGE: hydra::detail::Parameters -> init_parameters list need do have N parameters");
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, *(init_parameters.begin() + i));
	}

	Parameters(std::array<hydra::Parameter,N> const& init_parameters)
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, init_parameters[i]);
	}

	Parameters(hydra::Parameter(& init_parameters)[N])
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, init_parameters[i]);
	}



	__hydra_host__ __hydra_device__ inline
	Parameters(Parameters<N> const& other)
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, other.GetParameter(i));
	}

	__hydra_host__ __hydra_device__ inline
	Parameters<N>& operator=(Parameters<N> const& other)
	{
		if(this == &other) return *this;

		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, other.GetParameter(i));

		return *this;
	}


	/**
	 * @brief Print registered parameters.
	 */
	void PrintParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<  HYDRA_ENDL;
		HYDRA_MSG << "Parameters begin:" << HYDRA_ENDL;

		for(size_t i=0; i<N; i++ )
			HYDRA_MSG <<"  >> Parameter " << i <<") "<< fParameters[i] << HYDRA_ENDL;

		HYDRA_MSG <<"Parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}


	/**
	 * @brief Set parameters
	 * @param parameters
	 */
	__hydra_host__ inline
	void SetParameters(const std::vector<double>& parameters)
	{


		for(size_t i=0; i< N; i++){
			fParameters[i] = parameters[fParameters[i].GetIndex()];
		}

		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< N ; i++){
				stringStream << "Parameter["<< fParameters[i].GetIndex() <<"] :  "
						<< parameters[fParameters[i].GetIndex() ]
						              << "  " << fParameters[i] << "\n";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		return;
	}


	inline	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
	{

		for(size_t i=0; i<N; i++)
			user_parameters.push_back(&fParameters[i]);
	}

	size_t  GetParametersKey(){

		std::array<double,N> _temp;
		for(size_t i=0; i<N; i++) _temp[i]= fParameters[i];

		size_t key = detail::hash_range(_temp.begin(), _temp.end() );

		return key;
	}

	__hydra_host__ __hydra_device__ inline
	 size_t GetNumberOfParameters() const {
		return N;
	}

	__hydra_host__ __hydra_device__ inline
	const hydra::Parameter* GetParameters() const {
		return &fParameters[0];
	}

	template<typename Int,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	__hydra_host__ __hydra_device__ inline
	const hydra::Parameter& GetParameter(Int i) const {
		return fParameters[i];
	}

	__hydra_host__ inline
	const hydra::Parameter& GetParameter(const char* name) const {

		size_t i=0;

		for(i=0; i<N; i++)
			if (strcmp(fParameters[i].GetName(),name)==0) break;

		return fParameters[i] ;
	}

	template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	__hydra_host__ __hydra_device__ inline
	hydra::Parameter& Parameter(Int i) {
		return fParameters[i];
	}

	__hydra_host__ inline
	hydra::Parameter& Parameter(const char* name) {

		size_t i=0;

		for(i=0; i<N; i++)
			if (strcmp(fParameters[i].GetName(),name)==0) break;

		return fParameters[i] ;
	}

	template<typename Int,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	__hydra_host__ __hydra_device__ inline
	void SetParameter(Int i, hydra::Parameter const& value) {
		fParameters[i]=value;
	}

	template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	__hydra_host__ __hydra_device__ inline
	void SetParameter(Int i, double value) {
		fParameters[i]=value;
	}

	__hydra_host__ inline
	void SetParameter(const char* name, hydra::Parameter const& value) {

		size_t i=0;

		for(i=0; i<N; i++)
			if (strcmp(fParameters[i].GetName(),name)==0){
				fParameters[i]=value;
				break;
			}
	}

	__hydra_host__ inline
	void SetParameter(const char* name, double value) {

		size_t i=0;

		for(i=0; i<N; i++)
			if (strcmp(fParameters[i].GetName(),name)==0){
				fParameters[i]=value;
				break;
			}
	}

	template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	__hydra_host__ __hydra_device__  inline
	GReal_t operator[](Int i) const {
		return (GReal_t ) fParameters[i];
	}

private:

	hydra::Parameter fParameters[N];

};


/**
 * specialization for no-parametrized functor
 */
template<>
class Parameters<0>{

public:


	Parameters() = default;

	Parameters(std::initializer_list<Parameter>)
	{ }

	Parameters(std::array<Parameter,0> const& )
	{	}

	__hydra_host__ __hydra_device__
	Parameters(Parameters<0> const& )
	{	}

	__hydra_host__ __hydra_device__
	Parameters<0>& operator=(Parameters<0> const& )
	{	return *this;	}


	/**
	 * @brief Print registered parameters.
	 */
	void PrintParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<  HYDRA_ENDL;
		HYDRA_MSG <<"Parameters begin:" << HYDRA_ENDL;
		HYDRA_MSG <<" >>> No parameters to report." << HYDRA_ENDL;
		HYDRA_MSG <<"Parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}

	/**
	 * @brief Set parameters
	 * @param parameters
	 */
	__hydra_host__ inline
	void SetParameters(const std::vector<double>&){}

	inline	void AddUserParameters(std::vector<hydra::Parameter*>&  ){}

	__hydra_host__ __hydra_device__ inline
	size_t GetNumberOfParameters() const { 	return 0; 	}

};


}  // namespace detail

}  // namespace hydra



#endif /* PARAMETERS_H_ */
