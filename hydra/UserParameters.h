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
 * UserParameters.h
 *
 *  Created on: 27/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup fit
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


class UserParameters
{
public:


	UserParameters():
		fMnState(new ROOT::Minuit2::MnUserParameters()),
		fVariables( std::vector<Parameter*> ())
	{ }

	UserParameters( UserParameters const& other):
		fMnState( new ROOT::Minuit2::MnUserParameters(*other.GetMnStatePtr())),
		fVariables(other.GetVariables())
	{	}

	UserParameters& operator= ( UserParameters const& other)
	{
		if(this ==&other) return *this;

		std::unique_ptr<ROOT::Minuit2::MnUserParameters>
		temp(new ROOT::Minuit2::MnUserParameters(*other.GetMnStatePtr()));

		this->fMnState.swap(temp) ;
		this->fVariables = other.GetVariables();
		return *this;
	}


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

		param->SetIndex( fMnState->Index( param->GetName()) );

		if( update_size)fVariables.push_back(param);

	}




	void UpdateParameters(ROOT::Minuit2::FunctionMinimum const& minimum )
	{
		auto optimized_parameters =  minimum.UserParameters();
		for(Parameter* param: fVariables){
			param->SetValue( optimized_parameters.Value(param->GetName()));
			param->SetError( optimized_parameters.Error(param->GetName()));
		}


	}

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

	const ROOT::Minuit2::MnUserParameters& GetMnState() const
	{
		return *fMnState;
	}

	ROOT::Minuit2::MnUserParameters& GetMnState()
	{
		return *fMnState;
	}

	const std::unique_ptr<ROOT::Minuit2::MnUserParameters>& GetMnStatePtr() const
	{
		return fMnState;
	}

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

/*
__host__
std::ostream& operator<<(std::ostream& os, UserParameters const& par){

	return os << par.GetMnState() ;
}
*/

}  // namespace hydra

#endif /* USERPARAMETERS_H_ */
