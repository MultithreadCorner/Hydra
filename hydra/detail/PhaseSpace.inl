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
PhaseSpace<N,GRND>::PhaseSpace( const GReal_t (&daughtersMasses)[N]):
fSeed(1)
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses[i];

	fSeed *= detail::hash_range(&fMasses[0], &fMasses[0]+N);

}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace( const std::array<GReal_t,N>& daughtersMasses):
fSeed(1)
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses[i];

	fSeed *= detail::hash_range(&fMasses[0], &fMasses[0]+N);
}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace( const std::initializer_list<GReal_t>& daughtersMasses):
fSeed(1)
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= daughtersMasses.begin()[i];

	fSeed *= detail::hash_range(&fMasses[0], &fMasses[0]+N);
}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace( PhaseSpace<N,GRND> const& other):
fSeed(other.GetSeed())
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= other.GetMasses()[i];
}

template <size_t N, typename GRND>
template <typename GRND2>
PhaseSpace<N,GRND>::PhaseSpace( PhaseSpace<N,GRND2> const& other):
fSeed(other.GetSeed())
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= other.GetMasses()[i];
}


template <size_t N, typename GRND>
PhaseSpace<N,GRND> &
PhaseSpace<N,GRND>::operator=( PhaseSpace<N,GRND> const& other)
{
	if(this==&other) return *this;
	this->fSeed = other.GetSeed();
	for(size_t i=0;i<N;i++)
		this->ffMasses[i]= other.GetMasses()[i];

	return *this;
}

template <size_t N, typename GRND>
template <typename GRND2>
PhaseSpace<N,GRND> &
PhaseSpace<N,GRND>::operator=( PhaseSpace<N,GRND2> const& other)
{
	if(this==&other) return *this;
		this->fSeed = other.GetSeed();
		for(size_t i=0;i<N;i++)
			this->ffMasses[i]= other.GetMasses()[i];

		return *this;
}


template <size_t N, typename GRND>
template<typename FUNCTOR, hydra::detail::Backend BACKEND>
std::pair<GReal_t, GReal_t>
PhaseSpace<N,GRND>::AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy,
		Vector4R const& mother, FUNCTOR const& functor, size_t n){

	detail::StatsPHSP result;

	if (EnergyChecker( mother )){

		detail::AverageMother<N,GRND,FUNCTOR>
		reducer( mother,fMasses, fSeed,functor);

		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> first(0);

		HYDRA_EXTERNAL_NS::thrust::counting_iterator<GLong_t> last = first + n;

		result = 	detail::launch_reducer(policy,	first, last, reducer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}

	return std::make_pair(result.fMean, ::sqrt(result.fM2)/result.fW );

}

template <size_t N, typename GRND>
template<typename FUNCTOR,typename Iterator>
std::pair<GReal_t, GReal_t>
PhaseSpace<N,GRND>::AverageOn(Iterator begin, Iterator end, FUNCTOR const& functor) {

	detail::StatsPHSP result ;

	if (EnergyChecker( begin, end)){

		detail::AverageMothers<N,GRND,FUNCTOR>	reducer( fMasses, fSeed,functor);

		result =	detail::launch_reducer(begin, end, reducer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}

	return std::make_pair(result.fMean, ::sqrt(result.fM2)/ result.fW);
}

template <size_t N, typename GRND>
template<typename Iterator, typename ...FUNCTOR>
void PhaseSpace<N,GRND>::Evaluate(Vector4R const& mother, Iterator begin, Iterator end,
		 FUNCTOR const& ...functors) {

	if (EnergyChecker( mother )){

		detail::EvalMother<N,GRND,FUNCTOR...> evaluator( mother, fMasses, fSeed, functors...);

		detail::launch_evaluator( begin, end, evaluator );


	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename IteratorMother, typename Iterator>
void PhaseSpace<N,GRND>::Evaluate( IteratorMother mbegin,
		IteratorMother mend, Iterator begin, FUNCTOR const& ...functors) {

if (EnergyChecker( mbegin, mend)){


	detail::EvalMothers<N,GRND,FUNCTOR...> evaluator(fMasses, fSeed,functors... );

	detail::launch_evaluator( mbegin, mend, begin, evaluator );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}

}

//Evaluate range semantics interface ----------------------------


template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
			 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Evaluate(Vector4R const& mother, Iterable&& result,
		FUNCTOR const& ...functors) {

	if (EnergyChecker( mother )){

		detail::EvalMother<N,GRND,FUNCTOR...> evaluator( mother, fMasses, fSeed, functors...);

		detail::launch_evaluator( std::forward<Iterable>(result).begin(),
				std::forward<Iterable>(result).end(), evaluator );


	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}

	return make_range( std::forward<Iterable>(result).begin(),
			std::forward<Iterable>(result).end() );

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename IterableMother, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value &&
	hydra::detail::is_iterable<IterableMother>::value,
				 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Evaluate( IterableMother&& mothers, Iterable&& result, FUNCTOR const& ...functors) {

if (EnergyChecker( std::forward<IterableMother>(mothers).begin(),
		std::forward<IterableMother>(mothers).end())){


	detail::EvalMothers<N,GRND,FUNCTOR...> evaluator(fMasses, fSeed,functors... );

	detail::launch_evaluator( std::forward<IterableMother>(mothers).begin(),
			std::forward<IterableMother>(mothers).end(),
			std::forward<Iterable>(result).begin(), evaluator );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}

return make_range( std::forward<Iterable>(result).begin(),
			std::forward<Iterable>(result).end() );
}
//---------------------------------------------------------------

template <size_t N, typename GRND>
template<typename Iterator>
void PhaseSpace<N,GRND>::Generate(Vector4R const& mother, Iterator begin, Iterator end){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the fourvector of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */

	if (EnergyChecker( mother )){

	detail::DecayMother<N,GRND> decayer(mother,fMasses, fSeed);
	detail::launch_decayer(begin, end, decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}
}

template <size_t N, typename GRND>
template<typename Iterator1, typename Iterator2>
void PhaseSpace<N,GRND>::Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */
	if (EnergyChecker( begin, end)){

	detail::DecayMothers<N,GRND> decayer(fMasses, fSeed);
	detail::launch_decayer(begin, end, daughters_begin, decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}


}

template <size_t N, typename GRND>
template<typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
				 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Generate(Vector4R const& mother, Iterable&& events){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the fourvector of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */

	if (EnergyChecker( mother )){

	detail::DecayMother<N,GRND> decayer(mother,fMasses, fSeed);
	detail::launch_decayer(std::forward<Iterable>(events).begin(),
			std::forward<Iterable>(events).end(), decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}

	return make_range( std::forward<Iterable>(events).begin(),
			std::forward<Iterable>(events).end() );
}

template <size_t N, typename GRND>
template<typename IterableMothers, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value &&
		hydra::detail::is_iterable<IterableMothers>::value,
					 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Generate( IterableMothers&& mothers, Iterable&& daughters){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */
	if (EnergyChecker( std::forward<IterableMothers>(mothers).begin(),
			std::forward<IterableMothers>(mothers).end()))
	{

	detail::DecayMothers<N,GRND> decayer(fMasses, fSeed);
	detail::launch_decayer(std::forward<IterableMothers>(mothers).begin(),
			std::forward<IterableMothers>(mothers).end(),
			std::forward<Iterable>(daughters).begin(), decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}

	return make_range( std::forward<Iterable>(daughters).begin(),
				std::forward<Iterable>(daughters).end() );
}


//========================
template <size_t N, typename GRND>
template<typename Iterator, hydra::detail::Backend BACKEND>
void PhaseSpace<N,GRND>::Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy ,Vector4R const& mother, Iterator begin, Iterator end){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the fourvector of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */
/*
#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
*/
	if (EnergyChecker( mother )){

	detail::DecayMother<N,GRND> decayer(mother,fMasses, fSeed);
	detail::launch_decayer(exec_policy ,begin, end, decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the mass of the mother particle")
	}
}

template <size_t N, typename GRND>
template<typename Iterator1, typename Iterator2, hydra::detail::Backend BACKEND>
void PhaseSpace<N,GRND>::Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy , Iterator1 begin, Iterator1 end, Iterator2 daughters_begin){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */
/*
#if(THRUST_DEVICE_SYSTEM==THRUST_DEVICE_BACKEND_CUDA && (BACKEND==device))
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
*/
	if (EnergyChecker( begin, end)){

	detail::DecayMothers<N,GRND> decayer(fMasses, fSeed);
	detail::launch_decayer(exec_policy ,begin, end, daughters_begin, decayer );

	}
	else {
		HYDRA_LOG(WARNING, "Not enough energy to generate all decays.Check the masses of the mother particles")
	}


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
	x = ::sqrt(x) / (2 * a);
	return x;
}


template <size_t N, typename GRND>
template<typename Iterator>
inline bool PhaseSpace<N,GRND>::EnergyChecker(Iterator first, Iterator last) const {

	return  HYDRA_EXTERNAL_NS::thrust::all_of( first,  last,  detail::CheckEnergy<N>(  fMasses) );

}

template <size_t N, typename GRND>
inline bool PhaseSpace<N,GRND>::EnergyChecker( Vector4R const& mother) const {

	GReal_t fTeCmTm =  mother.mass();

	#pragma unroll N
			for (size_t n = 0; n < N; n++)
			{
				fTeCmTm -= fMasses[n];
			}

			return (bool) fTeCmTm > 0.0;
}



}//namespace hydra

#endif /* _PHASESPACE_INL_ */
