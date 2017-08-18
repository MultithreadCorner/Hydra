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
 * Generate.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup phsp
 */

#ifndef _PHASESPACE_INL_
#define _PHASESPACE_INL_


namespace hydra {


template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(const GReal_t motherMass, const GReal_t (&daughtersMasses)[N]):
fSeed(1)
{
	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses[i];

	GReal_t fTeCmTm = 0.0;

	fTeCmTm = motherMass; // total energy in C.M. minus the sum of the masses

	for (size_t n = 0; n < N; n++)
		fTeCmTm -= daughtersMasses[n];

	if (fTeCmTm < 0.0) {
		std::cout << "Not enough energy for this decay. Exit." << std::endl;
		exit(1);
	}

}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(const GReal_t motherMass, const std::array<GReal_t,N>& daughtersMasses):
fSeed(1)
{
	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses[i];

	GReal_t fTeCmTm = 0.0;

	fTeCmTm = motherMass; // total energy in C.M. minus the sum of the masses

	for (size_t n = 0; n < N; n++)
		fTeCmTm -= daughtersMasses[n];

	if (fTeCmTm < 0.0) {
		std::cout << "Not enough energy for this decay. Exit." << std::endl;
		exit(1);
	}

}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(const GReal_t motherMass, const std::initializer_list<GReal_t>& daughtersMasses):
fSeed(1)
{
	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses.begin()[i];

	GReal_t fTeCmTm = 0.0;

	fTeCmTm = motherMass; // total energy in C.M. minus the sum of the masses

	for (size_t n = 0; n < N; n++)
		fTeCmTm -= daughtersMasses.begin()[n];

	if (fTeCmTm < 0.0) {
		std::cout << "Not enough energy for this decay. Exit." << std::endl;
		exit(1);
	}

}

template <size_t N, typename GRND>
template<typename FUNCTOR, hydra::detail::Backend BACKEND>
std::pair<GReal_t, GReal_t>
PhaseSpace<N,GRND>::AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy,
		Vector4R const& mother, FUNCTOR const& functor, size_t n){

	detail::AverageMother<N,GRND,FUNCTOR>
	reducer( mother,fMasses, fSeed,functor);

	thrust::counting_iterator<GLong_t> first(0);

	thrust::counting_iterator<GLong_t> last = first + n;

	detail::StatsPHSP result =
			detail::launch_reducer(policy,	first, last, reducer );
	return std::make_pair(result.fMean, sqrt(result.fM2) );

}

template <size_t N, typename GRND>
template<typename FUNCTOR,typename Iterator>
std::pair<GReal_t, GReal_t>
PhaseSpace<N,GRND>::AverageOn(Iterator begin, Iterator end, FUNCTOR const& functor) {

	detail::AverageMothers<N,GRND,FUNCTOR>	reducer( fMasses, fSeed,functor);

	detail::StatsPHSP result =
			detail::launch_reducer(begin, end, reducer );
	return std::make_pair(result.fMean, sqrt(result.fM2) );

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename Iterator>
void PhaseSpace<N,GRND>::Evaluate(Vector4R const& mother, Iterator begin, Iterator end,
		 FUNCTOR const& ...functors) {

	detail::EvalMother<N,GRND,FUNCTOR...> evaluator( mother, fMasses, fSeed, functors...);

	detail::launch_evaluator( begin, end, evaluator );

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename IteratorMother, typename Iterator>
void PhaseSpace<N,GRND>::Evaluate( IteratorMother mbegin,
		IteratorMother mend, Iterator begin, FUNCTOR const& ...functors) {

	detail::EvalMothers<N,GRND,FUNCTOR...> evaluator(fMasses, fSeed,functors... );

	detail::launch_evaluator( mbegin, mend, begin, evaluator );

}

template <size_t N, typename GRND>
template<typename Iterator>
void PhaseSpace<N,GRND>::Generate(Vector4R const& mother, Iterator begin, Iterator end){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the fourvector of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */

#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif


	detail::DecayMother<N,GRND> decayer(mother,fMasses, fSeed);
	detail::launch_decayer(begin, end, decayer );

}

template <size_t N, typename GRND>
template<typename Iterator1, typename Iterator2>
void PhaseSpace<N,GRND>::Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */

#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif


	detail::DecayMothers<N,GRND> decayer(fMasses, fSeed);
	detail::launch_decayer(begin, end, daughters_begin, decayer );



}

template <size_t N, typename GRND>
inline GInt_t PhaseSpace<N,GRND>::GetSeed() const	{
	return fSeed;
}

template <size_t N, typename GRND>
inline void PhaseSpace<N,GRND>::SetSeed(GInt_t _seed) 	{
	fSeed=_seed;
}


/**
 * PDK function
 */
template <size_t N, typename GRND>
inline GReal_t PhaseSpace<N,GRND>::PDK(const GReal_t a, const GReal_t b, const GReal_t c) const {
	//the PDK function
	GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
	x = sqrt(x) / (2 * a);
	return x;
}



}//namespace hydra

#endif /* _PHASESPACE_INL_ */
