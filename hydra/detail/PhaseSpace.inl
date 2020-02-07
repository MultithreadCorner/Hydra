/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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

#include <hydra/detail/utility/Exception.h>

namespace hydra {


template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(double motherMass, const GReal_t (&daughtersMasses)[N]):
fSeed(1),
fMotherMass(motherMass),
fMaxWeight(0.),
fECM(0.)
{
	if(fMotherMass<=0.0){
		throw std::invalid_argument("[hydra::PhaseSpace]: Mass of mother particle is negative or zero. (fMotherMass <=0)");
	}

	for(size_t i=0;i<N;i++){

		fMasses[i]= daughtersMasses[i];

		if(fMasses[i]<=0.0){
			throw std::invalid_argument("[hydra::PhaseSpace]: Mass of daughter particle is negative or zero. (fMass[] <=0)");
		}

	}

	//should make a good enough seed
	fSeed *= (1+ detail::hash_range(&fMasses[0], &fMasses[0]+N));

	//compute maximum weight
	fECM = fMotherMass;

	for (size_t n = 0; n < N; n++)
	{
		fECM -= fMasses[n];
	}

	if(fECM<=0.0){
		throw std::runtime_error("[hydra::PhaseSpace]: Not enough energy to generate this decay. ( fECM <=0.") ;
	}

	double emmax = fECM + fMasses[0];
	double emmin = 0.0;
	double wtmax = 1.0;

	for (size_t n = 1; n < N; n++)
	{
		emmin += fMasses[n - 1];
		emmax += fMasses[n];
		wtmax *= PDK(emmax, emmin, fMasses[n]);
	}

	fMaxWeight = 1.0 / wtmax;
}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(double motherMass,  std::array<GReal_t,N>const& daughtersMasses):
fSeed(1),
fMotherMass(motherMass),
fMaxWeight(0.),
fECM(0.)
{
	if(fMotherMass<=0.0){
		throw std::invalid_argument("[hydra::PhaseSpace]: Mass of mother particle is negative or zero. (fMotherMass <=0)");
	}

	for(size_t i=0;i<N;i++){


		fMasses[i]= daughtersMasses[i];

		if(fMasses[i]<=0.0){
			throw std::invalid_argument("[hydra::PhaseSpace]: Mass of daughter particle is negative or zero. (fMass[] <=0)");
		}

	}

	//should make a good enough seed
	fSeed *= (1+ detail::hash_range(&fMasses[0], &fMasses[0]+N));

	//compute maximum weight
	fECM = fMotherMass;

	for (size_t n = 0; n < N; n++)
	{
		fECM -= fMasses[n];
	}

	if(fECM<=0.0){
		throw std::runtime_error("[hydra::PhaseSpace]: Not enough energy to generate this decay. ( fECM <=0.") ;
	}

	double emmax = fECM + fMasses[0];
	double emmin = 0.0;
	double wtmax = 1.0;

	for (size_t n = 1; n < N; n++)
	{
		emmin += fMasses[n - 1];
		emmax += fMasses[n];
		wtmax *= PDK(emmax, emmin, fMasses[n]);
	}

	fMaxWeight = 1.0 / wtmax;
}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace(double motherMass,  const std::initializer_list<GReal_t>& daughtersMasses):
fSeed(1),
fMotherMass(motherMass),
fMaxWeight(0.),
fECM(0.)
{
	if(fMotherMass<=0.0){
		throw std::invalid_argument("[hydra::PhaseSpace]: Mass of mother particle is negative or zero. (fMotherMass <=0)");
	}

	for(size_t i=0;i<N;i++){


		fMasses[i]= *(daughtersMasses.begin()+i);

		if(fMasses[i]<=0.0){
			throw std::invalid_argument("[hydra::PhaseSpace]: Mass of daughter particle is negative or zero. (fMass[] <=0)");
		}
	}

	//should make a good enough seed
	fSeed *= (1+ detail::hash_range(&fMasses[0], &fMasses[0]+N));

	//compute maximum weight
	fECM = fMotherMass;

	for (size_t n = 0; n < N; n++)
	{
		fECM -= fMasses[n];
	}

	if(fECM<=0.0){
		throw std::runtime_error("[hydra::PhaseSpace]: Not enough energy to generate this decay. ( fECM <=0.") ;
	}

	double emmax = fECM + fMasses[0];
	double emmin = 0.0;
	double wtmax = 1.0;

	for (size_t n = 1; n < N; n++)
	{
		emmin += fMasses[n - 1];
		emmax += fMasses[n];
		wtmax *= PDK(emmax, emmin, fMasses[n]);
	}

	fMaxWeight = 1.0 / wtmax;
}

template <size_t N, typename GRND>
PhaseSpace<N,GRND>::PhaseSpace( PhaseSpace<N,GRND> const& other):
fSeed(other.GetSeed()),
fMotherMass(other.GetMotherMass()),
fMaxWeight(other.GetMaxWeight()),
fECM(other.GetECM())
{

	for(size_t i=0;i<N;i++)
		fMasses[i]= other.GetMasses()[i];
}

template <size_t N, typename GRND>
template <typename GRND2>
PhaseSpace<N,GRND>::PhaseSpace( PhaseSpace<N,GRND2> const& other):
fSeed(other.GetSeed()),
fMotherMass(other.GetMotherMass()),
fMaxWeight(other.GetMaxWeight()),
fECM(other.GetECM())
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
	this->fMotherMass = other.other.GetMotherMass();
	this->fMaxWeight = other.GetMaxWeight();
	this->fECM = other.GetECM();

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
	this->fMotherMass = other.other.GetMotherMass();
	this->fMaxWeight = other.GetMaxWeight();
	this->fECM = other.GetECM();

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

	detail::AverageMother<N,GRND,FUNCTOR>reducer( mother, fMasses, fMaxWeight, fECM, fSeed,functor);

	hydra_thrust::counting_iterator<GLong_t> first(0);

	hydra_thrust::counting_iterator<GLong_t> last = first + n;

	result = 	detail::launch_reducer(policy,	first, last, reducer );

	return std::make_pair(result.fMean, ::sqrt(result.fM2)/result.fW );

}

template <size_t N, typename GRND>
template<typename FUNCTOR,typename Iterator>
std::pair<GReal_t, GReal_t>
PhaseSpace<N,GRND>::AverageOn(Iterator begin, Iterator end, FUNCTOR const& functor) {

	detail::StatsPHSP result ;

	detail::AverageMothers<N,GRND,FUNCTOR>	reducer( fMasses,  fMaxWeight, fECM, fSeed, functor);

	result =	detail::launch_reducer(begin, end, reducer );

	return std::make_pair(result.fMean, ::sqrt(result.fM2)/ result.fW);
}

template <size_t N, typename GRND>
template<typename Iterator, typename ...FUNCTOR>
void PhaseSpace<N,GRND>::Evaluate(Vector4R const& mother,
		Iterator begin, Iterator end, FUNCTOR const& ...functors) {

		detail::EvalMother<N,GRND,FUNCTOR...> evaluator( mother, fMasses, fMaxWeight, fECM, fSeed, functors...);

		detail::launch_evaluator( begin, end, evaluator );

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename IteratorMother, typename Iterator>
void PhaseSpace<N,GRND>::Evaluate( IteratorMother mbegin,
		IteratorMother mend, Iterator begin, FUNCTOR const& ...functors) {

	detail::EvalMothers<N,GRND,FUNCTOR...> evaluator( fMotherMass,fMasses,  fMaxWeight, fECM,fSeed,functors... );

	detail::launch_evaluator( mbegin, mend, begin, evaluator );

}

//Evaluate range semantics interface ----------------------------


template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Evaluate(Vector4R const& mother, Iterable&& result,
		FUNCTOR const& ...functors) {

	detail::EvalMother<N,GRND,FUNCTOR...> evaluator( mother, fMasses, fMaxWeight, fECM, fSeed, functors...);

	detail::launch_evaluator( std::forward<Iterable>(result).begin(),
			std::forward<Iterable>(result).end(), evaluator );

	return make_range( std::forward<Iterable>(result).begin(),
			std::forward<Iterable>(result).end() );

}

template <size_t N, typename GRND>
template<typename ...FUNCTOR, typename IterableMother, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value &&
	hydra::detail::is_iterable<IterableMother>::value,
				 hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
PhaseSpace<N,GRND>::Evaluate( IterableMother&& mothers, Iterable&& result, FUNCTOR const& ...functors) {


	detail::EvalMothers<N,GRND,FUNCTOR...> evaluator(fMasses,  fMaxWeight, fECM,fSeed,functors... );

	detail::launch_evaluator( std::forward<IterableMother>(mothers).begin(),
			std::forward<IterableMother>(mothers).end(),
			std::forward<Iterable>(result).begin(),
			evaluator );

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

	detail::DecayMother<N,GRND> decayer(mother,fMasses, fMaxWeight, fECM, fSeed);
	detail::launch_decayer(begin, end, decayer );

}

template <size_t N, typename GRND>
template<typename Iterator1, typename Iterator2>
void PhaseSpace<N,GRND>::Generate( Iterator1 begin, Iterator1 end, Iterator2 daughters_begin){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */

	detail::DecayMothers<N,GRND> decayer(fMasses, fMaxWeight, fECM, fSeed);
	detail::launch_decayer(begin, end, daughters_begin, decayer );

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

	detail::DecayMother<N,GRND> decayer(mother,fMasses,  fMaxWeight, fECM,fSeed);

	detail::launch_decayer(std::forward<Iterable>(events).begin(),
			std::forward<Iterable>(events).end(), decayer );

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

	detail::DecayMothers<N,GRND> decayer(fMasses,  fMaxWeight, fECM,fSeed);

	detail::launch_decayer(std::forward<IterableMothers>(mothers).begin(),
			std::forward<IterableMothers>(mothers).end(),
			std::forward<Iterable>(daughters).begin(), decayer );

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

	detail::DecayMother<N,GRND> decayer(mother,fMasses,  fMaxWeight, fECM, fSeed);
	detail::launch_decayer(exec_policy ,begin, end, decayer );

}

template <size_t N, typename GRND>
template<typename Iterator1, typename Iterator2, hydra::detail::Backend BACKEND>
void PhaseSpace<N,GRND>::Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy , Iterator1 begin, Iterator1 end, Iterator2 daughters_begin){
	/**
	 * Run the generator and calculate the maximum weight. It takes as input the device vector with the four-vectors of the mother particle
	 * in any system of reference. The daughters will be generated in this system.
	 */


	detail::DecayMothers<N,GRND> decayer(fMasses,  fMaxWeight, fECM,  fSeed);
	detail::launch_decayer(exec_policy ,begin, end, daughters_begin, decayer );

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


}//namespace hydra

#endif /* _PHASESPACE_INL_ */
