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
#include <hydra/Placeholders.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/logical.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <array>


namespace hydra {



template<typename T, typename BACKEND>
class multivector;




/**
 * @brief This class implements storage in SoA layouts for
 * table where all elements have the same type.
 */
template<typename ...T, hydra::detail::Backend BACKEND>
class multivector< HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	constexpr static size_t N = sizeof...(T);

	//Useful aliases
	template<typename Type>
	using vector = typename system_t::template container<Type>;

	template<typename Type>
	using pointer_v    = typename system_t::template container<Type>::pointer;

	template<typename Type>
	using iterator_v   = typename system_t::template container<Type>::iterator;

	template<typename Type>
	using const_iterator_v   = typename system_t::template container<Type>::const_iterator;

	template<typename Type>
	using reverse_iterator_v   = typename system_t::template container<Type>::reverse_iterator;

	template<typename Type>
	using const_reverse_iterator_v   = typename system_t::template container<Type>::const_reverse_iterator;

	template<typename Type>
	using reference_v = typename system_t::template container<Type>::reference;

	template<typename Type>
	using const_reference_v = typename system_t::template container<Type>::const_reference;

	template<typename Type>
	using value_type_v = typename system_t::template container<Type>::value_type;

	typedef HYDRA_EXTERNAL_NS::thrust::tuple<T...> tuple_type;

public:

	typedef HYDRA_EXTERNAL_NS::thrust::tuple< vector<T>...	  	 > 	storage_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< pointer_v<T>... 	 > 	pointer_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< value_type_v<T>... > 	value_type_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< iterator_v<T>...   >  iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_iterator_v<T>...  > 	const_iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< reverse_iterator_v<T>...> 	reverse_iterator_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_reverse_iterator_v<T>... > 	const_reverse_iterator_t;

	typedef HYDRA_EXTERNAL_NS::thrust::tuple< reference_v<T>...		 > 	reference_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple< const_reference_v<T>...> 	const_reference_t;

	//zip iterator
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<iterator_t>		     		 iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_iterator_t>	 		 const_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<reverse_iterator_t>	 		 reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_reverse_iterator_t>	 const_reverse_iterator;

	 //stl-like typedefs
	 typedef size_t size_type;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_iterator>::reference const_reference;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	 typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::iterator_category iterator_category;

	 template<typename Functor>
	 using caster_iterator = HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
			 iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	 template<typename Functor>
	 using caster_reverse_iterator = HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
			 reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >;

	 template<typename Iterators,  unsigned int I1, unsigned int I2,unsigned int ...IN>
	 using columns_iterator = HYDRA_EXTERNAL_NS::thrust::zip_iterator< HYDRA_EXTERNAL_NS::thrust::tuple<
	 			typename HYDRA_EXTERNAL_NS::thrust::tuple_element< I1, Iterators >::type,
	 			typename HYDRA_EXTERNAL_NS::thrust::tuple_element< I2, Iterators >::type,
	 			typename HYDRA_EXTERNAL_NS::thrust::tuple_element< IN, Iterators >::type...> >;

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
		HYDRA_EXTERNAL_NS::thrust::fill(begin(), end(), value );
	};

	/**
	 * Constructor initializing the multivector with \p n copies of \p value .
	 * @param pair hydra::pair<size_t, hydra::tuple<T...> >  object to copy from.
	 */
	template<typename Int, typename = typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<std::is_integral<Int>::value>::type >
	multivector(hydra::pair<Int, HYDRA_EXTERNAL_NS::thrust::tuple<T...> > const& pair)
	{
		__resize(pair.first);
		HYDRA_EXTERNAL_NS::thrust::fill(begin(), end(), pair.second );
	}


	/**
	 * Copy constructor
	 * @param other
	 */
	multivector(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other )
	{
		__resize(other.size());
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
	}

	/**
	 * Move constructor
	 * @param other
	 */
	multivector(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...> ,detail::BackendPolicy<BACKEND>>&& other ):
	fData(other.__move())
	{}

	/**
	 * Copy constructor for containers allocated in different backends.
	 * @param other
	 */
	template< hydra::detail::Backend BACKEND2>
	multivector(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND2>> const& other )
	{
	  __resize(other.size());

		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
	}

	/*! This constructor builds a \p multivector from a range.
	 *  \param first The beginning of the range.
	 *  \param last The end of the range.
	 */
	template< typename Iterator>
	multivector(Iterator first, Iterator last )
	{
		__resize( HYDRA_EXTERNAL_NS::thrust::distance(first, last) );

		HYDRA_EXTERNAL_NS::thrust::copy(first, last, begin());

	}

	/**
	 * Assignment operator
	 * @param other
	 * @return
	 */
	multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(this==&other) return *this;

		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());

		return *this;
	}

	/**
	 * Move-assignment operator
	 * @param other
	 * @return
	 */
	multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND> >&& other )
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
	multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND>>&
	operator=(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>,detail::BackendPolicy<BACKEND2> > const& other )
	{
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(), other.end(), begin());
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
		return HYDRA_EXTERNAL_NS::thrust::distance( begin(), end() );
	}

	/*! Returns the number of elements which have been reserved in this
	 *  \p multivector.
	 */
	inline size_type capacity() const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<0>(fData).capacity();
	}

    /*! This method returns true iff size() == 0.
     *  \return true if size() == 0; false, otherwise.
     */
	inline bool empty() const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<0>(fData).empty();
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
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
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
		size_type first_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), first);
		size_type last_position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), last);

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
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
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
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
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
	inline 	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::tuple,
		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference< InputIterator >::type >::type >::value ||
	detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references,
		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference< InputIterator >::type>::type >::value, void>::type
	insert(iterator pos, InputIterator first, InputIterator last)
	{
		size_type position = HYDRA_EXTERNAL_NS::thrust::distance(begin(), pos);
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
	inline columns_iterator< iterator_t, I1, I2,IN...>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< iterator_t, I1, I2,IN...>
	end(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, iterator_t>::type
	begin(placeholders::placeholder<I> )
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin();
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, iterator_t>::type
	end(placeholders::placeholder<I>  )
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_t, I1, I2,IN...>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_t, I1, I2,IN...>
	rend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn)
	{
		return __rend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, reverse_iterator_t>::type
	rbegin(placeholders::placeholder<I>  )
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin();
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, reverse_iterator_t>::type
	rend(placeholders::placeholder<I>  )
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend();
	}

	//constant access
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	begin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __begin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	end(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __end(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	begin(placeholders::placeholder<I> ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 inline	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	end(placeholders::placeholder<I> ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	cbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __cbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	cend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __cend(c1, c2, cn...);
	}

	template<unsigned int I>
	inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	cbegin(placeholders::placeholder<I> )  const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin();
	}

	template<unsigned int I>
	 	inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
	cend(placeholders::placeholder<I>  ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator<  const_reverse_iterator_t, I1, I2,IN...>
	rbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __rbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	rend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __rend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	rbegin(placeholders::placeholder<I>  ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	rend(placeholders::placeholder<I>  ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend();
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	crbegin(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __crbegin(c1, c2, cn...);
	}

	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	crend(placeholders::placeholder<I1> c1,	placeholders::placeholder<I2> c2,  placeholders::placeholder<IN> ...cn) const
	{
		return __crend(c1, c2, cn...);
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	crbegin(placeholders::placeholder<I>  ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin();
	}

	template<unsigned int I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_reverse_iterator_t>::type
	crend(placeholders::placeholder<I>  ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend();
	}

	template<unsigned int I>
	 inline const typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, storage_t>::type&
	column(placeholders::placeholder<I>   ) const
	{
		return HYDRA_EXTERNAL_NS::thrust::get<I>(fData);
	}


	template<typename Functor>
	 inline caster_iterator<Functor> operator[](Functor const& caster)
	{	return this->begin(caster) ;	}

	//
	template<unsigned int I>
	inline	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, iterator_t>::type
	operator[](placeholders::placeholder<I>  index)
	{	return begin(index) ;	}

	template<unsigned int I>
	inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I, const_iterator_t>::type
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
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->begin(), caster);
	}

	template<typename Functor>
	 inline caster_iterator<Functor> __caster_end( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->end(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rbegin( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rbegin(), caster);
	}

	template<typename Functor>
	 inline caster_reverse_iterator<Functor> __caster_rend( Functor const& caster )
	{
		return HYDRA_EXTERNAL_NS::thrust::transform_iterator< Functor,
				reverse_iterator, typename std::result_of<Functor(tuple_type&)>::type >(this->rend(), caster);
	}
	//__________________________________________
	// pop_back
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__pop_back(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__pop_back()
	{
		 HYDRA_EXTERNAL_NS::thrust::get<I>(fData).pop_back();
		__pop_back<I + 1>();

	}

	//__________________________________________
	// resize
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__resize(size_type){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__resize(size_type n)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).resize(n);
		__resize<I + 1>(n);
	}

	//__________________________________________
	// push_back
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back( value_type const& ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back( value_type const& value )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).push_back( HYDRA_EXTERNAL_NS::thrust::get<I>(value) );
		__push_back<I + 1>( value );
	}

	//__________________________________________
	// clear
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__clear(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__clear( )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).clear();
		__clear<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__shrink_to_fit(){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__shrink_to_fit( )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).shrink_to_fit();
		__shrink_to_fit<I + 1>();
	}

	//__________________________________________
	// shrink_to_fit
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__reserve(size_type ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__reserve(size_type size )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).reserve(size);
		__reserve<I + 1>(size);
	}

	//__________________________________________
	// erase
	template<size_t I>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ){ }

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void>::type
	__erase_helper(size_type position )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).erase(
					HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()+position);
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
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void>::type
	__erase_helper( size_type ,  size_type ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void>::type
	__erase_helper( size_type first_position,  size_type last_position)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).erase(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + first_position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + last_position );

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
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type ,  const value_type&){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position,  const value_type &x)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()+position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );

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
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type, size_type, const value_type & ){}

	template<size_t I=0>
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type position, size_type n, const value_type &x)
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(
				HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + position, n,
				HYDRA_EXTERNAL_NS::thrust::get<I>(x) );
	}

	 inline iterator __insert(size_t position, size_type n,  const value_type &x )
	{
		__insert_helper(position, x  ,  n);
		return begin()+position+n;
	}


	//__________________________________________
	// insert
	template<size_t I,typename InputIterator >
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert(size_type , InputIterator , InputIterator  ){}

	template<size_t I=0,typename InputIterator >
	 inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert(size_type position, InputIterator first, InputIterator last  )
	{
		HYDRA_EXTERNAL_NS::thrust::get<I>(fData).insert(HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin() + position,
				HYDRA_EXTERNAL_NS::thrust::get<I>(first),
				HYDRA_EXTERNAL_NS::thrust::get<I>(last) );

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
	inline columns_iterator< iterator_t, I1, I2,IN...>
	__begin(placeholders::placeholder<I1> , placeholders::placeholder<I2>,
			placeholders::placeholder<IN>...) {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).begin(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).begin(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).begin()...));
			}

	template<size_t ...I>
	 inline iterator __begin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
					HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()...) );
	}

	inline iterator __begin(){
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	__begin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).begin(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).begin(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).begin()...));
	}

	template<size_t ...I>
	 inline const_iterator __begin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).begin()... ) );
	}

	inline const_iterator __begin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	//const begin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	__cbegin(placeholders::placeholder<I1>, placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).cbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).cbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).cbegin()...));
	}

	template<size_t ...I>
	 inline const_iterator __cbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cbegin() ... )	);
	}

	 inline const_iterator __cbegin() const {
		return __begin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ End ______________
	//end
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline columns_iterator< iterator_t, I1, I2,IN...>
	 __end(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			 placeholders::placeholder<IN>...) {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).end(),
						 HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).end(),
						 HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).end()...));
	 }

	template<size_t ...I>
	 inline iterator __end_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end()...) );
	}

	inline iterator __end(){
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_iterator_t, I1, I2,IN...>
	__end(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).end(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).end(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).end()...));
	}

	template<size_t ...I>
	 inline const_iterator __end_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).end()... ) );
	}

	inline const_iterator __end() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	//const end
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
		inline columns_iterator< const_iterator_t, I1, I2,IN...>
		__cend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
				placeholders::placeholder<IN>...) const {
			return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(
							HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).cend(),
							HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).cend(),
							HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).cend()...));
		}

	template<size_t ...I>
	 inline const_iterator __cend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).cend() ... )	);
	}

	 inline const_iterator __cend() const {
		return __end_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse Begin ______________
	//rbegin
	 template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	 inline columns_iterator< reverse_iterator_t, I1, I2,IN...>
	 __rbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2>,
			 placeholders::placeholder<IN>...) {
		 return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				 HYDRA_EXTERNAL_NS::thrust::make_tuple(
						 HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).rbegin(),
						 HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).rbegin(),
						 HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).rbegin()...));
	 }

	template<size_t ...I>
	 inline reverse_iterator __rbegin_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
					HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin()...) );
	}

	inline reverse_iterator __rbegin(){
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//const rbegin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	__rbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).rbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).rbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).rbegin()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __rbegin_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rbegin()... ) );
	}

	inline const_reverse_iterator __rbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	//crbegin
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	__crbegin(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).crbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).crbegin(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).crbegin()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __crbegin_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crbegin() ... )	);
	}

	inline const_reverse_iterator __crbegin() const {
		return __rbegin_helper(detail::make_index_sequence<N> { });
	}

	// _____________ Reverse End ______________
	//rend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< reverse_iterator_t, I1, I2,IN...>
	__rend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).rend(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).rend(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).rend()...));
	}

	template<size_t ...I>
	 inline reverse_iterator __rend_helper( detail::index_sequence<I...> ){

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend()...) );
	}

	inline reverse_iterator __rend(){
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//const rend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	__rend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).rend(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).rend(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).rend()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __rend_helper( detail::index_sequence<I...> ) const {

		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).rend()... ) );
	}

	inline const_reverse_iterator __rend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}

	//crend
	template<unsigned int I1, unsigned int I2,unsigned int ...IN >
	inline columns_iterator< const_reverse_iterator_t, I1, I2,IN...>
	__crend(placeholders::placeholder<I1> , placeholders::placeholder<I2> ,
			placeholders::placeholder<IN>...) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I1>(fData).crend(),
						HYDRA_EXTERNAL_NS::thrust::get<I2>(fData).crend(),
						HYDRA_EXTERNAL_NS::thrust::get<IN>(fData).crend()...));
	}

	template<size_t ...I>
	 inline const_reverse_iterator __crend_helper( detail::index_sequence<I...> ) const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(
						HYDRA_EXTERNAL_NS::thrust::get<I>(fData).crend() ... )	);
	}

	 inline const_reverse_iterator __crend() const {
		return __rend_helper(detail::make_index_sequence<N> { });
	}



	 inline storage_t  __move()
	{
		return std::move(fData);
	}

	storage_t fData;


};

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
get(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.column(placeholders::placeholder<I>{}))
{
	return other.column(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
begin(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
end(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
begin(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.begin(placeholders::placeholder<I>{}))
{
	return other.begin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
end(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.end(placeholders::placeholder<I>{}))
{
	return other.end(placeholders::placeholder<I>{});
}



template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rbegin(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rend(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>> const& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}


template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rbegin(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rbegin(placeholders::placeholder<I>{}))
{
	return other.rbegin(placeholders::placeholder<I>{});
}

template<unsigned int I,  hydra::detail::Backend BACKEND, typename ...T>
inline auto
rend(multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, detail::BackendPolicy<BACKEND>>& other  )
-> decltype(other.rend(placeholders::placeholder<I>{}))
{
	return other.rend(placeholders::placeholder<I>{});
}

template<typename ...T, hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator==(const multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__(
			HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<T...>,
				HYDRA_EXTERNAL_NS::thrust::tuple<T...> > const& values)
	{
			return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	return HYDRA_EXTERNAL_NS::thrust::all_of(
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end()  , rhs.end()  ), comparison);
}


template<typename ...T, hydra::detail::Backend BACKEND1, hydra::detail::Backend BACKEND2>
bool operator!=(const multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND1>>& lhs,
                const multivector<HYDRA_EXTERNAL_NS::thrust::tuple<T...>, hydra::detail::BackendPolicy<BACKEND2>>& rhs){

	auto comparison = []__hydra_host__ __hydra_device__(
			HYDRA_EXTERNAL_NS::thrust::tuple< HYDRA_EXTERNAL_NS::thrust::tuple<T...>,
			HYDRA_EXTERNAL_NS::thrust::tuple<T...>	> const& values){
		return HYDRA_EXTERNAL_NS::thrust::get<0>(values)== HYDRA_EXTERNAL_NS::thrust::get<1>(values);

	};

	return !(HYDRA_EXTERNAL_NS::thrust::all_of(
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.begin(), rhs.begin()),
			HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(lhs.end(), rhs.end())
	, comparison));
}




}  // namespace hydra


#endif /* MULTIVECTOR_H_ */
