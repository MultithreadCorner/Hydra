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
 * Evaluate.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef EVALUATE_INL_
#define EVALUATE_INL_

namespace hydra {

namespace detail {

template<typename ReturnType,  typename T >
struct process
{

	process(T const& f):
		fTuple(f)
	{}

	template<typename ArgType  >
	__host__ __device__ ReturnType operator()(ArgType&& x)
	{
		return detail::invoke<T,ArgType>(x,fTuple);
	}

	template<typename ArgType1, typename ArgType2  >
	__host__ __device__ ReturnType operator()(ArgType1&& x, ArgType2&& y )
	{
		return detail::invoke<T,ArgType1,ArgType2>(x,y,fTuple);
	}

	T fTuple;

};



template<typename T, template<typename, typename...> class V,  size_t N>
inline 	size_t get_size(std::array<V<T>*, N>const & Array)
{
	bool same_size=true;

	for ( size_t n=1; n< N; n++ )
	{
		if( Array[n-1]->size() != Array[n]->size() )
		{
			same_size = false;
			break;
		}
	}
	return same_size ? Array[0]->size() : 0;
}

}/* namespace detail */

/*
template< typename Iterator,typename ...Iterators, typename ...Functors,
typename thrust::detail::enable_if<
hydra::detail::are_all_same<typename Range<Iterator>::system ,
typename Range<Iterators>::system...>::value, int>::type=0>
inline auto Eval(thrust::tuple<Functors...> const& t,Range<Iterator>const& range, Range<Iterators>const&...  ranges) ->
typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
typename EvalReturnType<mc_device_vector,typename Functors::return_type ...>::type,
typename EvalReturnType<mc_host_vector,typename Functors::return_type ...>::type>::type
{
	typedef typename detail::if_then_else<
			std::is_same<thrust::device_system_tag,typename  Range<Iterator>::system>::value,
			mc_device_vector< thrust::tuple<typename Functors::return_type ...> >,
			mc_host_vector< thrust::tuple<typename Functors::return_type ...> > >::type container;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(range.begin(), ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(range.end(), ranges.end()...) );

	container Table( thrust::distance(begin, end) );

	thrust::transform(begin, end ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;
}
*/
/*
template< typename ...Iterators, typename ...Functors,
typename U =typename thrust::detail::enable_if<
hydra::detail::are_all_same< thrust::host_system_tag,typename Range<Iterators>::system...>::value,	void>::type>
inline auto Eval(thrust::tuple<Functors...> const& t, Range<Iterators> const&... ranges)
-> typename EvalReturnType<mc_host_vector,typename Functors::return_type ...>::type
{
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(ranges.end()...) );

	mc_host_vector< thrust::tuple<typename Functors::return_type ...> > Table( thrust::distance(begin, end) );

	thrust::transform(begin, end ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;
}*/

/*
template< template <class, class...> class V, typename T,  size_t N, typename ...Functors>
inline auto Eval( std::array<V<T>*,N>const & Array, thrust::tuple<Functors...> const& t)
->	typename EvalReturnType<V, typename Functors::return_type ...>::type
{


	size_t entries = detail::get_size<T, V, N>(Array);
	V< thrust::tuple<typename Functors::return_type ...> > Table(entries);

	std::array<typename V<T>::iterator, N> Array_Begin;
	std::array<typename V<T>::iterator, N> Array_End;

	for(size_t i=0;i<N;i++)
	{
		Array_Begin[i] = Array[i]->begin();
		Array_End[i]   = Array[i]->end();

	}

	auto begin = thrust::make_zip_iterator( detail::arrayToTuple(Array_Begin));
	auto end   = thrust::make_zip_iterator( detail::arrayToTuple(Array_End));

	thrust::transform(begin, end ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;


}

template< template <class, class...> class V, typename T,   size_t N, typename ...Functors>
inline void Eval( std::array<V<T>*, N>const & Array, thrust::tuple<Functors...> const& t,
		V< thrust::tuple<typename Functors::return_type ...> >& Table)
{


	size_t entries = detail::get_size<T, V, N>(Array);


	std::array<typename V<T>::iterator, N> Array_Begin;
	std::array<typename V<T>::iterator, N> Array_End;

	for(size_t i=0;i<N;i++)
	{
		Array_Begin[i] = Array[i]->begin();
		Array_End[i]   = Array[i]->end();

	}

	auto begin = thrust::make_zip_iterator( detail::arrayToTuple(Array_Begin));
	auto end   = thrust::make_zip_iterator( detail::arrayToTuple(Array_End));

	thrust::transform(begin, end , Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>,
			thrust::tuple<Functors...>>(t) );


}


template< template <class, class...> class V, typename T,typename Tuple,  size_t N, typename ...Functors>
inline auto Eval( std::array<V<T>*,N>const & Array,
		V<Tuple> const& Cache, thrust::tuple<Functors...> const& t)
->	typename EvalReturnType<V, typename Functors::return_type ...>::type
{


	size_t entries = detail::get_size<T, V, N>(Array);
	V< thrust::tuple<typename Functors::return_type ...> > Table(entries);

	std::array<typename V<T>::iterator, N> Array_Begin;
	std::array<typename V<T>::iterator, N> Array_End;

	for(size_t i=0;i<N;i++)
	{
		Array_Begin[i] = Array[i]->begin();
		Array_End[i]   = Array[i]->end();

	}

	auto begin = thrust::make_zip_iterator( detail::arrayToTuple(Array_Begin));
	auto end   = thrust::make_zip_iterator( detail::arrayToTuple(Array_End));

	thrust::transform(begin, end , Cache.begin(), Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;


}

template< template <class, class...> class V, typename T,  typename Tuple, size_t N, typename ...Functors>
inline void Eval( std::array<V<T>*, N>const & Array, V<Tuple> const& Cache,	thrust::tuple<Functors...> const& t,
		V< thrust::tuple<typename Functors::return_type ...> >& Table)
{


	size_t entries = detail::get_size<T, V, N>(Array);


	std::array<typename V<T>::iterator, N> Array_Begin;
	std::array<typename V<T>::iterator, N> Array_End;

	for(size_t i=0;i<N;i++)
	{
		Array_Begin[i] = Array[i]->begin();
		Array_End[i]   = Array[i]->end();

	}

	auto begin = thrust::make_zip_iterator( detail::arrayToTuple(Array_Begin));
	auto end   = thrust::make_zip_iterator( detail::arrayToTuple(Array_End));

	thrust::transform(begin, end ,Cache.begin(), Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>,
			thrust::tuple<Functors...>>(t) );


}

*/

}// namespace hydra

#endif /* EVALUATE_INL_ */
