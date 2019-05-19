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
 * UserParameters.h
 *
 *  Created on: 27/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef USERPARAMETERS_H_
#define USERPARAMETERS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/detail/Print.h>

#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnPrint.h"

#include <algorithm>
#include <vector>
#include <initializer_list>
#include <ostream>
#include <memory>

namespace hydra {

/**
 * \ingroup fit
 * \class UserParameters
 * \brief Class implementing a interface to ROOT::Minuit2::MnUserParameters
 *
 */
class UserParameters
{
public:


	UserParameters():
		fMnState(new ROOT::Minuit2::MnUserParameters()),
		fVariables( std::vector<Parameter*> ())
	{ }

	/**
	 * Copy constructor
	 * @param other
	 */
	UserParameters( UserParameters const& other):
		fMnState( new ROOT::Minuit2::MnUserParameters(*other.GetMnStatePtr())),
		fVariables(other.GetVariables())
	{	}

	/**
	 * Assignment operator
	 * @param other
	 * @return
	 */
	UserParameters& operator= ( UserParameters const& other)
	{
		if(this ==&other) return *this;

		std::unique_ptr<ROOT::Minuit2::MnUserParameters>
		temp(new ROOT::Minuit2::MnUserParameters(*other.GetMnStatePtr()));

		this->fMnState.swap(temp) ;
		this->fVariables = other.GetVariables();
		return *this;
	}


	/**
	 * Add a parameter to be tracked by ROOT::Minuit2
	 *
	 * @param param
	 * @param update_size
	 */
	void AddParameter( Parameter* param, GBool_t update_size=1 )
	{
		if(param->HasError() && param->IsLimited()){
			fMnState->Add(param->GetName(), param->GetValue(),
					param->GetError(), param->GetLowerLim(), param->GetUpperLim());
		}

		if(param->HasError() && !param->IsLimited()){
			fMnState->Add(param->GetName(), param->GetValue(), param->GetError());
		}

		if(!param->HasError() && !param->IsLimited()){
			fMnState->Add(param->GetName(), param->GetValue());
		}

		if( param->IsFixed())
			fMnState->Fix( param->GetName());

		param->SetIndex( fMnState->Index( param->GetName()) );

		if( update_size)fVariables.push_back(param);

	}



	/**
	 * Update model parameters with the values hold by an ROOT::Minuit2::FunctionMinimum object
	 * @param minimum
	 */
	void UpdateParameters(ROOT::Minuit2::FunctionMinimum const& minimum )
	{
		auto optimized_parameters =  minimum.UserParameters();
		for(Parameter* param: fVariables){
			param->SetValue( optimized_parameters.Value(param->GetName()));
			param->SetError( optimized_parameters.Error(param->GetName()));
		}


	}

	/**
	 * Update model parameters errors with the values hold by an ROOT::Minuit2::MinosError object
	 * @param minos_error
	 */
	void UpdateParameterLimits(ROOT::Minuit2::MinosError const& minos_error )
	{

		std::string name = minos_error.LowerState().Name( minos_error.Parameter());
		auto parameter = std::find_if(fVariables.begin(), fVariables.end(),
				[&](Parameter* const p) { return std::string(p->GetName()) == name; } );

		if (parameter == std::end(fVariables)) {
			HYDRA_LOG(WARNING, " Parameter :"<< name << " not found. Limits not set.\n\n")
			return;
		}
		(*parameter)->SetLowerLim(minos_error.Lower() );
		(*parameter)->SetUpperLim(minos_error.Lower() );

		return;
	}

	/**
	 * Print the parameters
	 */
	void PrintParameters(){

		if(!fVariables.size()){
			HYDRA_CALLER ;
			HYDRA_MSG <<"Parameters: "<< HYDRA_ENDL;
			HYDRA_MSG <<" UserParameters is empty \n\n" << HYDRA_ENDL;
			return;
		}
		else{
			HYDRA_CALLER ;
			HYDRA_MSG << "Parameters: "<< HYDRA_ENDL;
			for(size_t i=0; i<fVariables.size(); i++ )
				HYDRA_MSG <<"  Variable " << i <<") "<< *fVariables[i] << HYDRA_ENDL;
			HYDRA_MSG <<"Parameters end.\n\n" << HYDRA_ENDL;
		}
		return;
	}

	/**
	 * Print the parameters (ROOT::Minuit2)
	 */
	void PrintMinuitParameters(){
		std::cout<< this->GetMnState() << std::endl;
		return;
	}


	const std::vector<Parameter* >& GetVariables() const {
		return fVariables;
	}

	void SetVariables(const std::vector<Parameter*>& variables)
	{
		fVariables = variables;
		std::unique_ptr<ROOT::Minuit2::MnUserParameters>
				temp(new ROOT::Minuit2::MnUserParameters());
		this->fMnState.swap(temp) ;
		for(size_t i=0; i < fVariables.size(); i++)
			this->AddParameter( fVariables[i], 0);
	}

	/**
	 * Get an constant reference to the hold ROOT::Minuit2::MnUserParameters
	 * @return
	 */
	const ROOT::Minuit2::MnUserParameters& GetMnState() const
	{
		return *fMnState;
	}

	/**
	 * Get a constant reference to the hold ROOT::Minuit2::MnUserParameters
	 * @return
	 */
	ROOT::Minuit2::MnUserParameters& GetMnState()
	{
		return *fMnState;
	}

	/**
	 * Get a constant reference to the hold std::unique_ptr<ROOT::Minuit2::MnUserParameters>
	 * @return
	 */
	const std::unique_ptr<ROOT::Minuit2::MnUserParameters>& GetMnStatePtr() const
	{
		return fMnState;
	}

	/**
	 * Set the ROOT::Minuit2 state
	 * @param state ROOT::Minuit2::MnUserParameters object.
	 */
	void SetMnState( ROOT::Minuit2::MnUserParameters const& state)
	{
		std::unique_ptr<ROOT::Minuit2::MnUserParameters>
		temp(new ROOT::Minuit2::MnUserParameters(state ));

		fMnState.swap(temp) ;
	}



private:
	std::vector<Parameter*> fVariables;
    std::unique_ptr<ROOT::Minuit2::MnUserParameters> fMnState;

};

/**
 * Print the ROOT::Minuit2 state to stream
 * @param os std::ostream
 * @param par hydra::UserParameters
 * @return
 */
std::ostream& operator<<(std::ostream& os, UserParameters const& par){

	return os << par.GetMnState() ;
}


}  // namespace hydra

#endif /* USERPARAMETERS_H_ */
