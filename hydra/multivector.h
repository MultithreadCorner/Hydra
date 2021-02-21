/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * multivector.h
 *
 *  Created on: 10/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_H_
#define MULTIVECTOR_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/Caster.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Placeholders.h>
#include <hydra/Range.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/detail/ZipIteratorUtility.h>
#include <hydra/detail/TupleTraits.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/logical.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <array>


namespace hydra {



template<typename T, typename BACKEND>
class multivector;




/**
 * @brief This class implements storage in SoA layouts for
 * table where all elements have the same type.
 */
template<typename ...T, hydra::detail::Backend BACKEND>
class multivector< hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	constexpr static size_t N = sizeof...(T);

	//Useful aliases
	template<typename Type>
	using _vector = typename system_t::template container<Type>;

	template<typename Type>
	using _pointer = typename system_t::template container<Type>::pointer;

	template<typename Type>
	using _iterator	= typename system_t::template container<Type>::iterator;

	template<typename Type>
	using _const_iterator = typename system_t::template container<Type>::const_iterator;

	template<typename Type>
	using _reverse_iterator = typename system_t::template container<Type>::reverse_iterator;

	template<typename Type>
	using _const_reverse_iterator = typename system_t::template container<Type>::const_reverse_iterator;

	template<typename Type>
	using _reference = typename system_t::template container<Type>::reference;

	template<typename Type>
	using _const_reference = typename system_t::template container<Type>::const_reference;

	template<typename Type>
	using _value_type = typename system_t::template container<Type>::value_type;

	typedef hydra_thrust::tuple<T...> tuple_type;

public:

	//tuples...
	typedef hydra_thrust::tuple< _vector<T>...    >	   storage_tuple;
	typedef hydra_thrust::tuple< _pointer<T>...   >	   pointer_tuple;
	typedef hydra_thrust::tuple< _value_type<T>...>	value_type_tuple;

	typedef hydra_thrust::tuple< _iterator<T>...  >              iterator_tuple;
	typedef hydra_thrust::tuple< _const_iterator<T>...   > const_iterator_tuple;

	typedef hydra_thrust::tuple< _reverse_iterator<T>... >            reverse_iterator_tuple;
	typedef hydra_thrust::tuple< _const_reverse_iterator<T>...>	const_reverse_iterator_tuple;

	typedef hydra_thrust::tuple< _reference<T>...		 > 	      reference_tuple;
	typedef hydra_thrust::tuple< _const_reference<T>...  > 	const_reference_tuple;

	//zip iterator
	typedef hydra_thrust::zip_iterator<iterator_tuple>		     		 iterator;
	typedef hydra_thrust::zip_iterator<const_iterator_tuple>	 		 const_iterator;
	typedef hydra_thrust::zip_iterator<reverse_iterator_tuple>	 		 reverse_iterator;
	typedef hydra_thrust::zip_iterator<const_reverse_iterator_tuple>	 const_reverse_iterator;

	 //stl-like typedefs
	 typedef size_t size_type;
	 typedef typename hydra_thrust::iterator_traits<iterator>::reference             reference;
	 typedef typename hydra_thrust::iterator_traits<const_iterator>::reference const_reference;

	 typedef typename hydra_thrust::iterator_traits<iterator>::value_type               value_type;
	 typedef typename hydra_thrust::iterator_traits<iterator>::iterator_category iterator_category;

	 template<typename Functor>
	 using caster_iterator = hydra_thrust::transform_iterator< Functor,
			 iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	 template<typename Functor>
	 using caster_reverse_iterator = hydra_thrust::transform_iterator< Functor,
			 reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	 template<typename Iterators,  unsigned int I1, unsigned int I2,unsigned int ...IN>
	 using columns_iterator = hydra_thrust::zip_iterator< hydra_thrust::tuple<
	 			typename hydra_thrust::tuple_element< I1, Iterators >::type,
	 			typename hydra_thrust::tuple_element< I2, Iterators >::type,
	 			typename hydra_thrust::tuple_element< IN, Iterators >::type...> >;

	/**
	 * Default constructor. This constructor creates an empty \p multivector.
	 */
	multivector() = default;


	/**
	 * Constructor initializing the \p multivector with \p n entries.
	 * @param n The number of elements to initially create.
	 */
	multivector(size_t n){
		__resize(n);
	};

    /**
     * Constructor initializing the multivector with \p n copies of \p value .
     * @param n number of elements
     * @param value object to copy from (hydra::tuple or convertible to hydra::tuple)
     */
	multivector(size_t n, value_type const& value){
		__resize(n);
		hydra_thrust::fill(begin(), end(), value );
	};

	/**
	 * Constructor initializing the multivector with \p n copies of \p value .
	 * @param pair hydra::pair<size_t, hydra::tuple<T...> >  object to copy from.
	 */
	template<typename Int, typename = typename hydra_thrust::detail::enable_if<std::is_integral<Int>::value>::type >
	multivector(hydra::pair<Int, hydra_thrust::tuple<T...> > const& pair)
	{
		__resize(pair.first);
		hydra_thrust::fill(begin(), end(), pair.second );
	}


	/**
	 * Copy constructor
	 * @param other
	 */
	multivector(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other )
	{
		__resize(other.size());
		hydra_thrust::copy(other.begin(), other.end(), begin());
	}

	/**
	 * Move constructor
	 * @param other
	 */
	multivector(multivector<hydra_thrust::tuple<T...> ,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.__move())
	{}

	/**
	 * Copy constructor for containers allocated in different backends.
	 * @param other
	 */
	template< hydra::detail::Backend BACKEND2>
	multivector(multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND2>> const& other )
	{
	  __resize(other.size());

		hydra_thrust::copy(other.begin(), other.end(), begin());
	}

	/*! This constructor builds a \p multivector from a range.
	 *  \param first The beginning of the range.
	 *  \param last The end of the range.
	 */
	template< typename Iterator>
	multivector(Iterator first, Iterator last )
	{
		__resize( hydra_thrust::distance(first, last) );

		hydra_thrust::copy(first, last, begin());

	}
	template< typename Iterable,
	          typename = typename std::enable_if<
	           (detail::is_iterable<Iterable>::value) &&
	          !(detail::is_iterator<Iterable>::value) &&
	           (std::is_convertible<decltype(*std::declval<Iterable>().begin()), value_type>::value)
	          >::type >
	multivector(Iterable&& other )
	{
		__resize( hydra_thrust::distance(
				std::forward<Iterable>(other).begin(),
				std::forward<Iterable>(other).end() ) );

		hydra_thrust::copy(std::forward<Iterable>(other).begin(),
				           std::forward<Iterable>(other).end(),
				           begin());
	}
	/**
	 * Assignment operator
	 * @param other
	 * @return
	 */
	multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(this==&other) return *this;

		hydra_thrust::copy(other.begin(), other.end(), begin());

		return *this;
	}

	/**
	 * Move-assignment operator
	 * @param other
	 * @return
	 */
	multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this == &other) return *this;
		this->fData = other.__move();
		return *this;
	}

	/**
	 * Assignment operator
	 * @param other
	 * @return
	 */
	template< hydra::detail::Backend BACKEND2>
	multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<hydra_thrust::tuple<T...>,detail::BackendPolicy<BACKEND2> > const& other )
	{
		hydra_thrust::copy(other.begin(), other.end(), begin());
		return *this;
	}


    /*! This method erases the last element of this \p multivector, invalidating
     *  all iterators and references to it.
     */
	inline void pop_back()
	{
		__pop_back();
	}

    /*! This method appends the given element to the end of this \p multivector.
     *  \param x The element to append.
     */
	inline void	push_back(value_type const& value)
	{
		__push_back( value );
	}

    /*! This method appends the given element to the end of this \p multivector.
     *  \param x The element to append.
     *  \param functor Functor to convert the element to a value_type tuple
     */
	template<typename Functor, typename Obj>
	inline void	push_back(Functor  const& functor, Obj const& obj)
	{
		__push_back( functor(obj) );
	}

	/*!
	 *  Returns the number of elements in this \p multivector.
	 */
	inline size_type size() const
	{
		return hydra_thrust::distance( begin(), end() );
	}

	/*! Returns the number of elements which have been reserved in this
	 *  \p multivector.
	 */
	inline size_type capacity() const
	{
		return hydra_thrust::get<0>(fData).capacity();
	}

    /*! This method returns true iff size() == 0.
     *  \return true if size() == 0; false, otherwise.
     */
	inline bool empty() const
	{
		return hydra_thrust::get<0>(fData).empty();
	}

	/*! \brief Resizes this \p multivector to the specified number of elements.
	 *  \param new_size Number of elements this \p multivector should contain.
	 *  \throw std::length_error If n exceeds max_size().
	 *
	 *  This method will resize this \p multivector to the specified number of
	 *  elements.  If the number is smaller than this \p multivector's current
	 *  size this \p multivector is truncated, otherwise this \p multivector is
	 *  extended and new default initialized elements are populated.
	 */
	inline void resize(size_type new_size)
	{
		__resize(new_size);
	}


    /*! This method resizes this \p multivector to 0.
     */
	inline void clear()
	{
		__clear();
	}

    /*! This method shrinks the capacity of this \p multivector to exactly
     *  fit its elements.
     */
	inline void shrink_to_fit()
	{
		__shrink_to_fit();
	}


	/*! \brief If n is less than or equal to capacity(), this call has no effect.
	 *         Otherwise, this method is a request for allocation of additional memory. If
	 *         the request is successful, then capacity() is greater than or equal to
	 *         n; otherwise, capacity() is unchanged. In either case, size() is unchanged.
	 *  \throw std::length_error If n exceeds max_size().
	 */
	inline void reserve(size_type size)
	{
		__reserve(size);
	}

    /*! This method removes the element at position pos.
     *  \param pos The position of the element of interest.
     *  \return An iterator pointing to the new location of the element that followed the element
     *          at position pos.
     */
	inline iterator erase(iterator pos)
	{
		size_type position = hydra_thrust::distance(begin(), pos);
		return __erase(position);
	}

    /*! This method removes the range of elements [first,last) from this \p multivector.
     *  \param first The beginning of the range of elements to remove.
     *  \param last The end of the range of elements to remove.
     *  \return An iterator pointing to the new location of the element that followed the last
     *          element in the sequence [first,last).
     */
	inline iterator erase(iterator first, iterator last)
	{
		size_type first_position = hydra_thrust::distance(begin(), first);
		size_type last_position = hydra_thrust::distance(begin(), last);

		return __erase( first_position, last_position);
	}

    /*! This method inserts a single copy of a given exemplar value at the
     *  specified position in this \p multivector.
     *  \param position The insertion position.
     *  \param x The exemplar element to copy & insert.
     *  \return An iterator pointing to the newly inserted element.
     */
	inline iterator insert(iterator pos, const value_type &x)
	{
		size_type position = hydra_thrust::distance(begin(), pos);
		return __insert(position, x);
	}

	/*! This method inserts a copy of an exemplar value to a range at the
	 *  specified position in this \p multivector.
	 *  \param position The insertion position
	 *  \param n The number of insertions to perform.
	 *  \param x The value to replicate and insert.
	 */
	inline void insert(iterator pos, size_type n, const value_type &x)
	{
		size_type position = hydra_thrust::distance(begin(), pos);
		__insert(position, n, x);
	}


    /*! This method inserts a copy of an input range at the specified position
     *  in this \p multivector.
     *  \param position The insertion position.
     *  \param first The beginning of the range to copy.
     *  \param last  The end of the range to copy.
     *
     *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html>Input Iterator</a>,
     *                        and \p InputIterator's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>.
     */
	template< typename InputIterator>
	inline 	typename hydra_thrust::detail::enable_if<
	detail::is_zip_iterator<InputIterator>::value &&
	std::is_convertible<typename hydra_thrust::iterator_traits<InputIterator>::value_type,
	value_type>::value, void>::type
	insert(iterator pos, InputIterator first, InputIterator last)
	{
		size_type position = hydra_thrust::distance(begin(), pos);
		__insert(position, first, last);
	}


    /*! This method returns a const_reference referring to the first element of this
     *  \p multivector.
     *  \return The first element of this \p multivector.
     */
	inline reference front()
	{
		return __front();
	}

	/*! This method returns a reference pointing to the first element of this
	 *  \p multivector.
	 *  \return The first element of this \p multivector.
	 */
	inline const_reference front() const
	{
	   return __front();
	}

    /*! This method returns a reference referring to the last element of
     *  this vector_dev.
     *  \return The last element of this \p multivector.
     */
	inline reference back()
	{
		return __back();
	}

	/*! This method returns a const reference pointing to the last element of
	 *  this \p multivector.
	 *  \return The last element of this \p multivector.
	 */
	inline const_reference back() const
	{
		return __back();
	}

    /*! This method returns an iterator pointing to the beginning of
     *  this \p multivector.
     *  \return iterator
     */
	inline 	iterator begin()
	{
		return __begin();
	}

	/*! This method returns a const_iterator pointing to one element past the
	 *  last of this \p multivector.
	 *  \return begin() + size().
	 */
	inline iterator end()
	{
		return __end();
	}

	//constant automatic conversion access
	template<typename Functor>
	inline caster_iterator<Functor> begin( Functor const& caster )
	{
		return this->__caster_begin(caster);
	}


	template<typename Functor>
	inline caster_iterator<Functor> end( Functor const& caster )
	{
		return this->__caster_end(caster);
	}

	//constant automatic conversion access
	template<typename Functor>
	inline caster_reverse_iterator<Functor> rbegin( Functor const& caster )
	{
		return this->__caster_rbegin(caster);
	}

	template<typename Functor>
	inline caster_reverse_iterator<Functor> rend( Functor const& caster )
	{
		return this->__caster_rend(caster);
	}

	//non-constant access
	inline reverse_iterator rbegin()
	{
		return this->__rbegin();
	}

	inline reverse_iterator rend()
	{
		return this->__rend();
	}

	//constant access
	inline const_iterator begin() const
	{
		return this->__begin();
	}

	inline const_iterator end() const
	{
		return this->__end();
	}

	inline const_reverse_iterator rbegin() const
	{
		return this->__rbegin();
	}

	inline const_reverse_iterator rend() const
	{
		return this->__rend();
	}

	inline const_iterator cbegin() const
	{
		return this->__cbegin() ;
	}

	inline const_iterator cend() const
	{
		return this->	__cend() ;
	}

	inline 	const_reverse_iterator crbegin() const
	{
		return this->__crbegin() ;
	}

	inline 	const_reverse_iterator crend() const
	{
		return 	this->__crend() ;
	}

	//non-constant access
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< iterator_tuple, I1, I2,IN...>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< iterator_tuple, I1, I2,IN...>
	end(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, iterator_tuple>::type
	begin(placeholders::placeholder<I> )
	{
		return hydra_thrust::get<I>(fData).begin();
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, iterator_tuple>::type
	end(placeholders::placeholder<I>  )
	{
		return hydra_thrust::get<I>(fData).end();
	}


	template<typename Type>
	inline typename hydra_thrust::tuple_element<
		detail::index_in_tuple<Type, tuple_type>::value ,
		iterator_tuple >::type 	begin()
	{
		return hydra_thrust::get< detail::index_in_tuple<Type, tuple_type>::value >(fData).begin();
	}


	template<typename Type>
	inline typename hydra_thrust::tuple_element<
		detail::index_in_tuple<Type, tuple_type>::value ,
		iterator_tuple >::type	end()
	{

		return hydra_thrust::get< detail::index_in_tuple<Type, tuple_type>::value >(fData).end();
	}


	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_tuple, I1, I2,IN...>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_tuple, I1, I2,IN...>
	rend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __rend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, reverse_iterator_tuple>::type
	rbegin(placeholders::placeholder<I>  )
	{
		return hydra_thrust::get<I>(fData).rbegin();
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, reverse_iterator_tuple>::type
	rend(placeholders::placeholder<I>  )
	{
		return hydra_thrust::get<I>(fData).rend();
	}


	template<typename Type>
	inline typename hydra_thrust::tuple_element<
	detail::index_in_tuple<Type, tuple_type>::value ,
	reverse_iterator_tuple>::type
	rbegin()
	{
		return hydra_thrust::get< detail::index_in_tuple<Type, tuple_type>::value >(fData).rbegin();
	}


	template<typename Type>
	inline typename hydra_thrust::tuple_element<
	detail::index_in_tuple<Type, tuple_type>::value ,
	reverse_iterator_tuple >::type
	rend()
	{
		return hydra_thrust::get< detail::index_in_tuple<Type, tuple_type>::value >(fData).rend();
	}

	template<typename Type>
	inline typename hydra_thrust::tuple_element<detail::index_in_tuple<Type, tuple_type>::value ,
	storage_tuple>::type&
	column()
	{
		return hydra_thrust::get<detail::index_in_tuple<Type, tuple_type>::value>(fData);
	}

	template<unsigned int I>
	inline typename hydra_thrust::tuple_element<I, storage_tuple>::type&
	column(placeholders::placeholder<I>   )
	{
		return hydra_thrust::get<I>(fData);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline hydra::Range<columns_iterator< iterator_tuple, I1, I2,IN...>>
	column(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2, placeholders::placeholder<IN> ...cn)
	{
		return hydra::make_range( this->begin(c1,c2, cn...), this->end(c1, c2, cn...) );
	}

	//constant access
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	end(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline	typename hydra_thrust::tuple_element<I, const_iterator_tuple>::type
	begin(placeholders::placeholder<I> ) const
	{
		return hydra_thrust::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 inline	typename hydra_thrust::tuple_element<I, const_iterator_tuple>::type
	end(placeholders::placeholder<I> ) const
	{
		return hydra_thrust::get<I>(fData).cend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	cbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __cbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	cend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __cend(c1, c2, cn...);
	}

	template<unsigned int I>
	inline typename hydra_thrust::tuple_element<I, const_iterator_tuple>::type
	cbegin(placeholders::placeholder<I> )  const
	{
		return hydra_thrust::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 	inline typename hydra_thrust::tuple_element<I, const_iterator_tuple>::type
	cend(placeholders::placeholder<I>  ) const
	{
		return hydra_thrust::get<I>(fData).cend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator<  const_reverse_iterator_tuple, I1, I2,IN...>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	rend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __rend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, const_reverse_iterator_tuple>::type
	rbegin(placeholders::placeholder<I>  ) const
	{
		return hydra_thrust::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, const_reverse_iterator_tuple>::type
	rend(placeholders::placeholder<I>  ) const
	{
		return hydra_thrust::get<I>(fData).crend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	crbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __crbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	crend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __crend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, const_reverse_iterator_tuple>::type
	crbegin(placeholders::placeholder<I>  ) const
	{
		return hydra_thrust::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline typename hydra_thrust::tuple_element<I, const_reverse_iterator_tuple>::type
	crend(placeholders::placeholder<I>  ) const
	{
		return hydra_thrust::get<I>(fData).crend();
	}

	template<unsigned int I>
	 inline const typename hydra_thrust::tuple_element<I, storage_tuple>::type&
	column(placeholders::placeholder<I>   ) const
	{
		return hydra_thrust::get<I>(fData);
	}


	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline hydra::Range<columns_iterator< const_iterator_tuple, I1, I2,IN...>>
	column(placeholders::placeholder<I1> c1, placeholders::placeholder<I2> c2, placeholders::placeholder<IN> ...cn) const
	{
		return hydra::make_range( this->cbegin(c1,c2, cn...), this->cend(c1, c2, cn...) );
	}


	template<typename ...Iterables>
	auto meld( Iterables&& ...iterables)
	-> typename std::enable_if<detail::all_true<detail::is_iterable<Iterables>::value...>::value,
	hydra::Range<decltype(detail::meld_iterators(begin(), std::forward<Iterables>(iterables).begin()... ))>>::type
	{
			auto first = detail::meld_iterators(begin(), std::forward<Iterables>(iterables).begin()... );
			auto last  = detail::meld_iterators(end(), std::forward<Iterables>(iterables).end()... );

			return hydra::make_range(first, last);
	}


	template<typename Functor>
	 inline caster_iterator<Functor> operator[](Functor const& caster)
	{	return this->begin(caster) ;	}

	//
	template<unsigned int I>
	inline	typename hydra_thrust::tuple_element<I, iterator_tuple>::type
	operator[](placeholders::placeholder<I>  index)
	{	return begin(index) ;	}

	template<unsigned int I>
	inline typename hydra_thrust::tuple_element<I, const_iterator_tuple>::type
	operator[](placeholders::placeholder<I> index) const
	{	return cbegin(index); }


	/*! \brief Subscript access to the data contained in this vector_dev.
	 *  \param n The index of the element for which data should be accessed.
	 *  \return Read/write reference to data.
	 *
	 *  This operator allows for easy, array-style, data access.
	 *  Note that data access with this operator is unchecked and
	 *  out_of_range lookups are not defined.
	 */
	inline	reference operator[](size_t n)
	{	return begin()[n] ;	}

	/*! \brief Subscript read access to the data contained in this vector_dev.
	     *  \param n The index of the element for which data should be accessed.
	     *  \return Read reference to data.
	     *
	     *  This operator allows for easy, array-style, data access.
	     *  Note that data access with this operator is unchecked and
	     *  out_of_range lookups are not defined.
	     */
	inline const_reference operator[](size_t n) const
	{	return cbegin()[n]; }



private:


	//__________________________________________
	// caster accessors
	template<typename Functor>
	 inline caster_iterator<Functor> __caster_begin( Functor const& caster )
	{
		return hydra_thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->begin(), caster);
	}

	template<typename Functor>
	 inline caster_iterator<Functor> __caster_end( Functor const& caster )
	{
		return hydra_thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->end(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rbegin( Functor const& caster )
	{
		return hydra_thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rbegin(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rend( Functor const& caster )
	{
		return hydra_thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rend(), caster);
	}
	//__________________________________________
	// pop_back
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__pop_back(){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__pop_back()
	{
		 hydra_thrust::get<I>(fData).pop_back();
		__pop_back<I + 1>();

	}

	//__________________________________________
	// resize
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__resize(size_type){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__resize(size_type n)
	{
		hydra_thrust::get<I>(fData).resize(n);
		__resize<I + 1>(n);
	}

	//__________________________________________
	// push_back
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__push_back( value_type const& ){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__push_back( value_type const& value )
	{
		hydra_thrust::get<I>(fData).push_back( hydra_thrust::get<I>(value) );
		__push_back<I + 1>( value );
	}

	//__________________________________________
	// clear
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__clear(){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__clear( )
	{
		hydra_thrust::get<I>(fData).clear();
		__clear<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__shrink_to_fit(){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__shrink_to_fit( )
	{
		hydra_thrust::get<I>(fData).shrink_to_fit();
		__shrink_to_fit<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__reserve(size_type ){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__reserve(size_type size )
	{
		hydra_thrust::get<I>(fData).reserve(size);
		__reserve<I + 1>(size);
	}

	//__________________________________________
	// erase
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ){ }

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void>::type
	__erase_helper(size_type position )
	{
		hydra_thrust::get<I>(fData).erase(
					hydra_thrust::get<I>(fData).begin()+position);
		__erase_helper<I+1>(position);
	}

	 inline iterator __erase(size_type position )
	{
		__erase_helper(position);

		return begin() + position;
	}

	//__________________________________________
	// erase

	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ,  size_type ){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void>::type
	__erase_helper( size_type first_position,  size_type last_position)
	{
		hydra_thrust::get<I>(fData).erase(
				hydra_thrust::get<I>(fData).begin() + first_position,
				hydra_thrust::get<I>(fData).begin() + last_position );

		__erase_helper<I+1>(first_position, last_position);
	}

	 inline iterator __erase( size_type first_position,  size_type last_position )
	{
	    __erase_helper( first_position, last_position );

	    return begin() + first_position;
	}

	//__________________________________________
	// insert
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type ,  const value_type&){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position,  const value_type &x)
	{
		hydra_thrust::get<I>(fData).insert(
				hydra_thrust::get<I>(fData).begin()+position,
				hydra_thrust::get<I>(x) );

		__insert_helper<I+1>(position,x);
	}

	 inline iterator __insert(size_type position,  const value_type &x )
	{
		 __insert_helper(position, x );

		 return begin()+position;
	}

	//__________________________________________
	// insert
	template<size_t I>
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type, size_type, const value_type & ){}

	template<size_t I=0>
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position, size_type n, const value_type &x)
	{
		hydra_thrust::get<I>(fData).insert(
				hydra_thrust::get<I>(fData).begin() + position, n,
				hydra_thrust::get<I>(x) );
	}

	 inline iterator __insert(size_t position, size_type n,  const value_type &x )
	{
		__insert_helper(position, x  ,  n);
		return begin()+position+n;
	}


	//__________________________________________
	// insert
	template<size_t I,typename InputIterator >
	 inline typename hydra_thrust::detail::enable_if<(I == N), void >::type
	__insert(size_type , InputIterator , InputIterator  ){}

	template<size_t I=0,typename InputIterator >
	 inline typename hydra_thrust::detail::enable_if<(I < N), void >::type
	__insert(size_type position, InputIterator first, InputIterator last  )
	{
		hydra_thrust::get<I>(fData).insert(hydra_thrust::get<I>(fData).begin() + position,
				hydra_thrust::get<I>(first.get_iterator_tuple()),
				hydra_thrust::get<I>(last.get_iterator_tuple()) );

		__insert<I + 1>( position,  first, last );
	}

	//__________________________________________
	//front
	inline reference __front()
	{
		return *(begin());
	}

	inline const_reference __front() const
	{
		return *(cbegin());
	}

	//__________________________________________
	//back
	inline reference __back()
	{
		return *(end()-1);
	}

	inline const_reference __back() const
	{
		return *(cend()-1);
	}


	// _____________ Begin ______________
	//begin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< iterator_tuple, I1, I2,IN...>
	__begin(placeholders::placeholder<I1> , placeholders::placeholder<I2>,
			placeholders::placeholder<IN>...) {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).begin(),
						hydra_thrust::get<I2>(fData).begin(),
						hydra_thrust::get<IN>(fData).begin()...));
			}

	template<size_t ...I>
	 inline iterator __begin_helper( detail::index_sequence<I...> ){

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
					hydra_thrust::get<I>(fData).begin()...) );
	}

	inline iterator __begin(){
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	__begin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).begin(),
						hydra_thrust::get<I2>(fData).begin(),
						hydra_thrust::get<IN>(fData).begin()...));
	}

	template<size_t ...I>
	 inline const_iterator __begin_helper( detail::index_sequence<I...> ) const {

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).begin()... ) );
	}

	inline const_iterator __begin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	__cbegin(placeholders::placeholder<I1>, placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).cbegin(),
						hydra_thrust::get<I2>(fData).cbegin(),
						hydra_thrust::get<IN>(fData).cbegin()...));
	}

	template<size_t ...I>
	 inline const_iterator __cbegin_helper( detail::index_sequence<I...> ) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).cbegin() ... )	);
	}

	 inline const_iterator __cbegin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ End ______________
	//end
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline columns_iterator< iterator_tuple, I1, I2,IN...>
	 __end(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...) {
		 return hydra_thrust::make_zip_iterator(
				 hydra_thrust::make_tuple(
						 hydra_thrust::get<I1>(fData).end(),
						 hydra_thrust::get<I2>(fData).end(),
						 hydra_thrust::get<IN>(fData).end()...));
	 }

	template<size_t ...I>
	 inline iterator __end_helper( detail::index_sequence<I...> ){

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).end()...) );
	}

	inline iterator __end(){
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
	__end(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).end(),
						hydra_thrust::get<I2>(fData).end(),
						hydra_thrust::get<IN>(fData).end()...));
	}

	template<size_t ...I>
	 inline const_iterator __end_helper( detail::index_sequence<I...> ) const {

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).end()... ) );
	}

	inline const_iterator __end() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
		inline columns_iterator< const_iterator_tuple, I1, I2,IN...>
		__cend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
				placeholders::placeholder<IN>...) const {
			return hydra_thrust::make_zip_iterator(
					hydra_thrust::make_tuple(
							hydra_thrust::get<I1>(fData).cend(),
							hydra_thrust::get<I2>(fData).cend(),
							hydra_thrust::get<IN>(fData).cend()...));
		}

	template<size_t ...I>
	 inline const_iterator __cend_helper( detail::index_sequence<I...> ) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).cend() ... )	);
	}

	 inline const_iterator __cend() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse Begin ______________
	//rbegin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline columns_iterator< reverse_iterator_tuple, I1, I2,IN...>
	 __rbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2>,
			 placeholders::placeholder<IN>...) {
		 return hydra_thrust::make_zip_iterator(
				 hydra_thrust::make_tuple(
						 hydra_thrust::get<I1>(fData).rbegin(),
						 hydra_thrust::get<I2>(fData).rbegin(),
						 hydra_thrust::get<IN>(fData).rbegin()...));
	 }

	template<size_t ...I>
	 inline reverse_iterator __rbegin_helper( detail::index_sequence<I...> ){

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
					hydra_thrust::get<I>(fData).rbegin()...) );
	}

	inline reverse_iterator __rbegin(){
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	__rbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).rbegin(),
						hydra_thrust::get<I2>(fData).rbegin(),
						hydra_thrust::get<IN>(fData).rbegin()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __rbegin_helper( detail::index_sequence<I...> ) const {

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).rbegin()... ) );
	}

	inline const_reverse_iterator __rbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//crbegin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	__crbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).crbegin(),
						hydra_thrust::get<I2>(fData).crbegin(),
						hydra_thrust::get<IN>(fData).crbegin()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __crbegin_helper( detail::index_sequence<I...> ) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).crbegin() ... )	);
	}

	inline const_reverse_iterator __crbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse End ______________
	//rend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_tuple, I1, I2,IN...>
	__rend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).rend(),
						hydra_thrust::get<I2>(fData).rend(),
						hydra_thrust::get<IN>(fData).rend()...));
	}

	template<size_t ...I>
	 inline reverse_iterator __rend_helper( detail::index_sequence<I...> ){

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).rend()...) );
	}

	inline reverse_iterator __rend(){
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	__rend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).rend(),
						hydra_thrust::get<I2>(fData).rend(),
						hydra_thrust::get<IN>(fData).rend()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __rend_helper( detail::index_sequence<I...> ) const {

		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).rend()... ) );
	}

	inline const_reverse_iterator __rend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//crend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_tuple, I1, I2,IN...>
	__crend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I1>(fData).crend(),
						hydra_thrust::get<I2>(fData).crend(),
						hydra_thrust::get<IN>(fData).crend()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __crend_helper( detail::index_sequence<I...> ) const {
		return hydra_thrust::make_zip_iterator(
				hydra_thrust::make_tuple(
						hydra_thrust::get<I>(fData).crend() ... )	);
	}

	 inline const_reverse_iterator __crend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}



	 inline storage_tuple  __move()
	{
		return std::move(fData);
	}

	storage_tuple fData;


};

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
get(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.column(placeholders::placeholder<I>{}))
{
	return other.column(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
begin(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
end(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
begin(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
end(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}



template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rbegin(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rend(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rbegin(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rend(multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}

template<hydra::detail::Backend BACKEND, typename ...T, typename ...U>
inline hydra::Range<     hydra_thrust::zip_iterator< typename detail::tuple_cat_type<
typename multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>::iterator_tuple,
typename multivector<hydra_thrust::tuple<U...>, detail::BackendPolicy<BACKEND>>::iterator_tuple>::type> >
meld( multivector<hydra_thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& left,
	  multivector<hydra_thrust::tuple<U...>, detail::BackendPolicy<BACKEND>>& right)
{
	if( left.size() != right.size() ){
	     throw std::invalid_argument("[ hydra::join ]: containers have different size.");
	}

	auto first = hydra_thrust::make_zip_iterator( hydra_thrust::tuple_cat(
			left.begin().get_iterator_tuple(), right.begin().get_iterator_tuple()) );

	auto  last = hydra_thrust::make_zip_iterator( hydra_thrust::tuple_cat(
			left.end().get_iterator_tuple(), right.end().get_iterator_tuple()) );

	return hydra::make_range(first, last);

}



template<typename ...T, hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator==(const multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__(
			hydra_thrust::tuple< hydra_thrust::tuple<T...>,
				hydra_thrust::tuple<T...> > const& values)
	{
			return hydra_thrust::get<0>(values)== hydra_thrust::get<1>(values);

	};

	return hydra_thrust::all_of(
			hydra_thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			hydra_thrust::make_zip_iterator(lhs.end()  , rhs.end()  ), comparison);
}


template<typename ...T, hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator!=(const multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__( hydra_thrust::tuple< hydra_thrust::tuple<T...>,
			hydra_thrust::tuple<T...>	> const& values)
	{

		return hydra_thrust::get<0>(values)== hydra_thrust::get<1>(values);

	};

	return !(hydra_thrust::all_of(
			hydra_thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			hydra_thrust::make_zip_iterator(lhs.end(), rhs.end())
	, comparison));
}

template<hydra::detail::Backend BACKEND, typename ...T, unsigned int...I>
auto columns( multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>const& other, placeholders::placeholder<I>...cls)
-> Range<decltype(std::declval<multivector<hydra_thrust::tuple<T...>,
		hydra::detail::BackendPolicy<BACKEND>> const&>().begin(placeholders::placeholder<I>{}...))>
{

	typedef decltype( other.begin(cls...)) iterator_type;
	return Range<iterator_type>( other.begin(cls...), other.end(cls...));
}

template<hydra::detail::Backend BACKEND, typename ...T, unsigned int...I>
auto columns( multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>& other, placeholders::placeholder<I>...cls)
-> Range<decltype(std::declval<multivector<hydra_thrust::tuple<T...>,
		hydra::detail::BackendPolicy<BACKEND>>&>().begin(placeholders::placeholder<I>{}...))>
{

	typedef decltype( other.begin(cls...)) iterator_type;
	return Range<iterator_type>( other.begin(cls...), other.end(cls...));
}

template<typename Type, hydra::detail::Backend BACKEND, typename ...T>
auto columns( multivector<hydra_thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>& other)
-> Range<decltype(
std::declval<multivector<hydra_thrust::tuple<T...>,
hydra::detail::BackendPolicy<BACKEND>>&>().begin(
		placeholders::placeholder<detail::index_in_tuple<Type, hydra_thrust::tuple<T...> >::value>{})) >
{

	constexpr size_t I = detail::index_in_tuple<Type, hydra_thrust::tuple<T...> >::value;

	typedef decltype( other.begin( placeholders::placeholder<I>{} )) iterator_type;
	return Range<iterator_type>( other.begin( placeholders::placeholder<I>{}),
			other.end( placeholders::placeholder<I>{} ));
}



}  // namespace hydra


#endif /* MULTIVECTOR_H_ */
