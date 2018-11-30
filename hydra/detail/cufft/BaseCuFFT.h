/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * BaseCuFFT.h
 *
 *  Created on: Nov 26, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BASECUFFT_H_
#define BASECUFFT_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/Range.h>
#include <hydra/Tuple.h>
#include <hydra/detail/external/thrust/memory.h>

#include <cassert>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>

//FFTW
#include <cufft.h>

//Hydra wrappers
#include<hydra/cuda/CudaWrappers.h>
#include<hydra/detail/cufft/Wrappers.h>

namespace hydra {

template<typename InputType, typename OutputType, typename PlannerType >
class BaseCuFFT
{

protected:
	typedef typename PlannerType::plan_type plan_type;
	typedef std::unique_ptr<InputType,  detail::cufft::_Deleter> input_ptr_type;
	typedef std::unique_ptr<OutputType, detail::cufft::_Deleter> output_ptr_type;

public:

	BaseCuFFT()=delete;

	BaseCuFFT(int input_size, int output_size, unsigned flags=FFTW_ESTIMATE, int sign=0):
		fFlags(flags),
		fSign(sign),
		fNInput(input_size ),
		fNOutput(output_size),
		fInput(reinterpret_cast<InputType*>(hydra::cuda::malloc(sizeof(InputType)*input_size))),
		fOutput(reinterpret_cast<OutputType*>(hydra::cuda::malloc(sizeof(OutputType)*output_size)))
	{



		//------------------
		int logical_size = input_size > output_size ? input_size : output_size;

		fPlan =  fPlanner( logical_size, fInput.get(), fOutput.get(), flags, sign);

		if(fPlan==NULL){

			throw std::runtime_error("hydra::BaseCuFFT : can not allocate fftw_plan");
		}
	}

	BaseCuFFT( BaseCuFFT<InputType,OutputType,PlannerType>&& other):
		fFlags(other.GetFlags()),
		fSign(other.GetSign()),
		fNInput(other.GetNInput()),
		fNOutput(other.GetNOutput()),
		fInput(std::move(other.GetInput())),
		fOutput(std::move(other.GetOutput()))
	{
		fDestroyer(fPlan);

		fPlan = fPlanner( other.GetSize() , fInput.get(), fOutput.get(), fFlags, fSign);
	}

	BaseCuFFT<InputType,OutputType,PlannerType>&
	operator=(BaseCuFFT<InputType,OutputType,PlannerType>&& other)
	{
		if(this ==&other) return *this;

		fFlags  = other.GetFlags();
		fSign   = other.GetSign();
		fNInput = other.GetNInput();
		fNOutput= other.GetNOutput();
		fInput  = std::move(other.GetInput());
		fOutput = std::move(other.GetOutput());

		fDestroyer(fPlan);

		fPlan =  fPlanner( other.GetSize(), fInput.get(), fOutput.get(), fFlags, fSign);

		return *this;
	}

	template<typename Iterable,
	typename Type =	typename decltype(*std::declval<Iterable&>().begin())::value_type>
	inline typename std::enable_if<std::is_convertible<InputType, Type>::value
	                        && detail::is_iterable<Iterable>::value, void>::type
	LoadInputData( Iterable&& container)
	{

		using HYDRA_EXTERNAL_NS::thrust::raw_pointer_cast;

		LoadInput(std::forward<Iterable>(container).size(),
				reinterpret_cast<InputType*>(
						raw_pointer_cast(std::forward<Iterable>(container).data())));
	}

	inline void	LoadInputData(int size, const InputType* data)
	{
		LoadInput(size, data);
	}

	inline void Execute()
	{
		fExecutor(fPlan);
	}

	inline hydra::pair<InputType*, int>
	GetInputData()
	{
		return hydra::make_pair(&fInput.get()[0], fNInput );
	}

	inline hydra::pair<OutputType*, int>
	GetOutputData()
	{
		return hydra::make_pair(&fOutput.get()[0], fNOutput );
	}

	inline int GetSize() const
	{
		return  fNInput > fNOutput ? fNInput : fNOutput;
	}

	detail::cufft::_PlanDestroyer GetDestroyer() const
	{
		return fDestroyer;
	}

	const detail::cufft::_PlanExecutor GetExecutor() const
	{
		return fExecutor;
	}

	int GetNInput() const
	{
		return fNInput;
	}

	void SetNInput(int nInput)
	{
		fNInput = nInput;
	}

	int GetNOutput() const
	{
		return fNOutput;
	}

	void SetNOutput(int nOutput)
	{
		fNOutput = nOutput;
	}

	int GetSign() const
	{
		return fSign;
	}

	void SetSign(int sign)
	{
		fSign = sign;
	}

	unsigned GetFlags() const
	{
		return fFlags;
	}

	void SetFlags(unsigned  flags)
	{
		fFlags = flags;
	}
	PlannerType GetPlanner() const
	{
		return fPlanner;
	}

	void SetPlanner(PlannerType planner)
	{
		fPlanner = planner;
	}



	~BaseCuFFT(){
		fDestroyer(fPlan);
	}


private:

	inline input_ptr_type GetInput()
	{
		return std::move(fInput);
	}

	inline output_ptr_type	GetOutput()
	{
		return std::move(fOutput);
	}


	void LoadInput(int size, const InputType* data )
	{
		assert(size <= fNInput);

		hydra::cuda::memcpy(&fInput.get()[0], data, sizeof(InputType)*size);
		hydra::cuda::memset(&fInput.get()[size], 0, sizeof(InputType)*( fNInput-size  ));
	}

	unsigned fFlags;
	int fSign;
	int fNInput;
	int fNOutput;
	PlannerType  fPlanner;
	plan_type    fPlan ;
	detail::cufft::_PlanExecutor fExecutor;
	detail::cufft::_PlanDestroyer fDestroyer;
	input_ptr_type fInput;
	output_ptr_type	fOutput;
};

}  // namespace hydra






#endif /* BASECUFFT_H_ */
