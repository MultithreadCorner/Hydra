/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
#include <hydra/detail/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>

namespace hydra {

namespace detail {

template<size_t N>
class Parameters{

public:
	static const size_t parameter_count =N;

	 Parameters (){};

	Parameters(std::initializer_list<Parameter> init_parameters)
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, *(init_parameters.begin() + i));
	}

	Parameters(std::array<Parameter,N> const& init_parameters)
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, *(init_parameters.begin() + i));
	}

	__host__ __device__
	Parameters(Parameters<N> const& other)
	{
		for(unsigned int i=0; i<N; i++)
			this->SetParameter(i, other.GetParameter(i));
	}

	__host__ __device__
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
	__host__ inline
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

	__host__ __device__ inline
	 size_t GetNumberOfParameters() const {
		return N;
	}

	__host__ __device__ inline
	const Parameter* GetParameters() const {
		return &fParameters[0];
	}

	__host__ __device__ inline
	const Parameter& GetParameter(size_t i) const {
		return fParameters[i];
	}

	__host__ __device__ inline
	void SetParameter(size_t i, Parameter const& value) {
		fParameters[i]=value;
	}

	__host__ __device__ inline
	void SetParameter(size_t i, double value) {
		fParameters[i]=value;
	}


	__host__ __device__  inline
	GReal_t operator[](unsigned int i) const {
		return (GReal_t ) fParameters[i];
	}

private:

	Parameter fParameters[N];

};


/**
 * specialization for no-parametrized functor
 */
template<>
class Parameters<0>{

public:

	 Parameters (){};

	Parameters(std::initializer_list<Parameter> init_parameters)
	{ }

	Parameters(std::array<Parameter,0> const& init_parameters)
	{	}

	__host__ __device__
	Parameters(Parameters<0> const& other)
	{	}

	__host__ __device__
	Parameters<0>& operator=(Parameters<0> const& other)
	{	return *this;	}


	/**
	 * @brief Print registered parameters.
	 */
	void PrintParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<  HYDRA_ENDL;
		HYDRA_MSG << "Parameters begin:" << HYDRA_ENDL;
		HYDRA_MSG <<" >>> No parameters to report." << HYDRA_ENDL;
		HYDRA_MSG <<"Parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}

	/**
	 * @brief Set parameters
	 * @param parameters
	 */
	__host__ inline
	void SetParameters(const std::vector<double>& parameters){}

	inline	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters ){}

	__host__ __device__ inline
	size_t GetNumberOfParameters() const { 	return 0; 	}

};


}  // namespace detail

}  // namespace hydra



#endif /* PARAMETERS_H_ */
