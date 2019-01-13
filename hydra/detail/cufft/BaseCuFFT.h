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
#include <hydra/Complex.h>
#include <cassert>
#include <memory>
#include <utility>
#include <stdexcept>
#include <type_traits>


//FFTW
#include <cufft.h>

//Hydra wrappers
#include<hydra/detail/cufft/WrappersCuFFT.h>

namespace hydra {

template<typename InputType, typename OutputType, typename PlannerType >
class BaseCuFFT
{

protected:

	typedef typename PlannerType::plan_type plan_type;
	typedef std::unique_ptr<InputType,  detail::cufft::_Deleter> input_ptr_type;
	typedef std::unique_ptr<OutputType, detail::cufft::_Deleter> output_ptr_type;
	typedef HYDRA_EXTERNAL_NS::thrust::pointer<InputType, HYDRA_EXTERNAL_NS::thrust::cuda::tag> input_tagged_ptr_type;
	typedef HYDRA_EXTERNAL_NS::thrust::pointer<OutputType, HYDRA_EXTERNAL_NS::thrust::cuda::tag> output_tagged_ptr_type;

public:

	BaseCuFFT()=delete;

	BaseCuFFT(int input_size, int output_size, int sign=0):
		fSign(sign),
		fNInput(input_size ),
		fNOutput(output_size),
		fInput(reinterpret_cast<InputType*>(detail::cufft::malloc(sizeof(InputType)*input_size))),
		fOutput(reinterpret_cast<OutputType*>(detail::cufft::malloc(sizeof(OutputType)*output_size)))
	{



		//------------------
		int logical_size = input_size > output_size ? input_size : output_size;

		fPlan =  fPlanner( logical_size, 1);

		//	throw std::runtime_error("hydra::BaseCuFFT : can not allocate fftw_plan");

	}

	BaseCuFFT( BaseCuFFT<InputType,OutputType,PlannerType>&& other):
		fSign(other.GetSign()),
		fNInput(other.GetNInput()),
		fNOutput(other.GetNOutput()),
		fInput(std::move(other.GetInput())),
		fOutput(std::move(other.GetOutput()))
	{
		fDestroyer(fPlan);

		fPlan = fPlanner( other.GetSize() , 1);
	}

	BaseCuFFT<InputType,OutputType,PlannerType>&
	operator=(BaseCuFFT<InputType,OutputType,PlannerType>&& other)
	{
		if(this ==&other) return *this;

		fSign   = other.GetSign();
		fNInput = other.GetNInput();
		fNOutput= other.GetNOutput();
		fInput  = std::move(other.GetInput());
		fOutput = std::move(other.GetOutput());

		fDestroyer(fPlan);

		fPlan =  fPlanner( other.GetSize(), 1);

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


	inline void	LoadInputData(int size,	input_tagged_ptr_type data)
	{
		LoadInput(size, data.get());
	}

	inline void Execute()
	{
		fExecutor(fPlan, fInput.get(), fOutput.get(), fSign);
	}

	inline hydra::pair<input_tagged_ptr_type, int>
	GetInputData()
	{
		return hydra::make_pair( input_tagged_ptr_type(fInput.get()), fNInput );
	}

	inline hydra::pair<output_tagged_ptr_type, int>
	GetOutputData()
	{
		return hydra::make_pair( output_tagged_ptr_type(fOutput.get()),
				fNOutput );
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

	int GetNOutput() const
	{
		return fNOutput;
	}

	void Reset(int ninput, int noutput)
	{
		fNInput = ninput;
		fInput.reset(reinterpret_cast<InputType*>(detail::cufft::malloc(sizeof(InputType)*ninput)));

		fNOutput = noutput;
		fOutput.reset(reinterpret_cast<OutputType*>(detail::cufft::malloc(sizeof(OutputType)*ninput)));

		fDestroyer(fPlan);

		int logical_size = ninput > noutput ? ninput : noutput;

		fPlan =  fPlanner( logical_size, 1);
	}

	int GetSign() const
	{
		return fSign;
	}

	void SetSign(int sign)
	{
		fSign = sign;
	}


	PlannerType GetPlanner() const
	{
		return fPlanner;
	}

	void SetPlanner(PlannerType planner)
	{
		fPlanner = planner;
	}

	virtual void SetSize(int logical_size)=0;

	virtual ~BaseCuFFT(){
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

		detail::cufft::memcpy(&fInput.get()[0], data, sizeof(InputType)*size);
		detail::cufft::memset(&fInput.get()[size], 0, sizeof(InputType)*( fNInput-size  ));
	}

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
