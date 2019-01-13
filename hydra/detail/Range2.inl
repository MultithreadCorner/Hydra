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
 * Range.h
 *
 *  Created on: 29/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANGE2_INL_
#define RANGE2_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/FunctorTraits.h>
#include <utility>

namespace hydra {

template<typename Iterator, typename Functor>
class Range<Iterator,Functor>{

public:

	typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator<Functor, Iterator, typename Functor::return_type> iterator;

	Range()=delete;

	Range(Iterator begin, Iterator end, Functor functor):
		fBegin( begin),
		fEnd( end ),
		fFunctor(functor)
		{}

	Range(Iterator begin,  size_t last, Functor functor):
			fBegin( begin),
			fEnd( begin + last ),
			fFunctor(functor)
			{}

	Range(Range<Iterator,Functor> const& other):
			fBegin( other.GetBegin()),
			fEnd( other.GetEnd() ),
			fFunctor(other.GetFunctor())
			{}


	inline Range<Iterator,Functor>&
	operator=(Range<Iterator,Functor> const& other){

		if(this==&other) return this;

		fBegin = other.GetBegin();
		fEnd = other.GetEnd();
		fFunctor = other.GetFunctor();

		return this;
	}


	inline iterator begin(){ return iterator(fBegin, fFunctor); };

	inline iterator   end(){ return iterator(fEnd, fFunctor); };

	inline iterator begin()const{ return iterator(fBegin, fFunctor); };

	inline iterator   end()const{ return iterator(fEnd, fFunctor); };

	inline void resize(size_t size){	}

	inline 	size_t size() { return hydra::distance(fBegin, fEnd);}

	inline Functor const& GetFunctor() const { return fFunctor;};

	inline Functor& GetFunctor(){ return fFunctor;};


	inline Iterator GetBegin() const {
		return fBegin;
	}

	inline void SetBegin(Iterator begin) {
		fBegin = begin;
	}

	inline Iterator GetEnd() const {
		return fEnd;
	}

	inline void SetEnd(Iterator end) {
		fEnd = end;
	}

	inline typename  iterator::reference operator[](size_t i)
	{
	 return begin()[i];
	}


private:
	Iterator fBegin;
	Iterator fEnd;
	Functor  fFunctor;
};

template<typename Iterator, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< detail::is_hydra_functor<Functor>::value , Range<Iterator, Functor> >::type
make_range(Iterator begin, Iterator end, Functor const& functor ){
	return Range<Iterator, Functor>( begin, end, functor);
}

template<typename Iterator, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_hydra_functor<Functor>::value ,
    Range<HYDRA_EXTERNAL_NS::thrust::reverse_iterator<Iterator>, Functor> >::type
make_reverse_range(Iterator begin, Iterator end,Functor const& functor ){

	typedef HYDRA_EXTERNAL_NS::thrust::reverse_iterator<Iterator> reverse_iterator_type;

		return Range<reverse_iterator_type>(
				reverse_iterator_type(end),
				reverse_iterator_type(begin) );

}


template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	detail::is_iterable<Iterable>::value &&
	detail::is_hydra_functor<Functor>::value,
 Range<decltype(std::declval<Iterable>().begin()), Functor> >::type
make_range(Iterable const& iterable,Functor const& functor ){

	typedef decltype(hydra::begin(iterable)) iterator_type;
	return Range<iterator_type, Functor>( hydra::begin(iterable),
			hydra::end(iterable), functor);
}

template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	detail::is_iterable<Iterable>::value &&
	detail::is_hydra_functor<Functor>::value,
 Range<decltype(std::declval<Iterable>().begin()), Functor> >::type
make_range(Iterable&& iterable,Functor const& functor ){

	typedef decltype(hydra::begin(std::forward<Iterable>(iterable))) iterator_type;
	return Range<iterator_type, Functor>( hydra::begin(std::forward<Iterable>(iterable)),
			hydra::end( std::forward<Iterable>(iterable) ), functor);
}



template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_reverse_iterable<Iterable>::value &&
    detail::is_hydra_functor<Functor>::value ,
    Range<decltype(std::declval<Iterable>().rbegin()), Functor> >::type
make_reverse_range(Iterable const& iterable,Functor const& functor ){

	typedef decltype(hydra::rbegin(iterable)) iterator_reverse_type;
	return Range<iterator_reverse_type, Functor>( hydra::rbegin(iterable),
			hydra::rend(iterable), functor);
}


template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_reverse_iterable<Iterable>::value &&
    detail::is_hydra_functor<Functor>::value ,
    Range<decltype(std::declval<Iterable>().rbegin()), Functor> >::type
make_reverse_range(Iterable&& iterable,Functor const& functor ){

	typedef decltype(hydra::rbegin( std::forward<Iterable>(iterable) )) iterator_reverse_type;
	return Range<iterator_reverse_type, Functor>( hydra::rbegin( std::forward<Iterable>(iterable) ),
			hydra::rend( std::forward<Iterable>(iterable) ), functor);
}


template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_iterable<Iterable>::value &&
    detail::is_hydra_functor<Functor>::value ,
    Range<decltype(std::declval< const Iterable>().begin()), Functor> >::type
operator|(Iterable const& iterable, Functor const& functor){

	typedef decltype(hydra::begin(iterable)) iterator_type;
		return Range<iterator_type, Functor>( hydra::begin(iterable),
				hydra::end(iterable), functor);
}


template<typename Iterable, typename Functor>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_iterable<Iterable>::value &&
    detail::is_hydra_functor<Functor>::value ,
    Range<decltype(std::declval<Iterable>().begin()), Functor> >::type
operator|(Iterable&& iterable, Functor const& functor){

	typedef decltype(hydra::begin( std::forward<Iterable>(iterable) )) iterator_type;

		return Range<iterator_type, Functor>( hydra::begin( std::forward<Iterable>(iterable) ),
				hydra::end( std::forward<Iterable>(iterable) ), functor);
}

template<typename Iterable>
typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
    detail::is_iterable<Iterable>::value ,
    Range<
     HYDRA_EXTERNAL_NS::thrust::reverse_iterator<
     	 decltype(std::declval<Iterable>().begin()) >>>::type
reverse(Iterable&& iterable) {

	typedef decltype(hydra::begin(iterable)) iterator_type;
	typedef HYDRA_EXTERNAL_NS::thrust::reverse_iterator<iterator_type> reverse_iterator_type;

		return Range<reverse_iterator_type>(
				reverse_iterator_type(hydra::end( std::forward<Iterable>(iterable) )),
				reverse_iterator_type(hydra::begin( std::forward<Iterable>(iterable) )) );

}


}  // namespace hydra



#endif /* GENERICRANGE_H_ */
