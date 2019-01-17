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
 * DenseHistogram.h
 *
 *  Created on: 22/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DENSEHISTOGRAM_H_
#define DENSEHISTOGRAM_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/cpp/System.h>
#include <hydra/Types.h>
#include <hydra/detail/Dimensionality.h>
#include <hydra/detail/functors/GetBinCenter.h>
#include <hydra/Range.h>
#include <hydra/Algorithm.h>

#include <hydra/detail/external/thrust/iterator/counting_iterator.h>

#include <type_traits>
#include <utility>
#include <array>


namespace hydra {

/**
 * \ingroup histogram
 */
template< typename T, size_t N, typename BACKEND, typename = typename detail::dimensionality<N>::type,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
class DenseHistogram;

/**
 * \ingroup histogram
 * \brief Class representing multidimensional dense histograms.
 * \tparam T type of data to histogram
 * \tparam N number of dimensions
 * \tparam BACKEND memory space where histogram is allocated
 */
template<typename T, size_t N , hydra::detail::Backend BACKEND>
class DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
{
	typedef hydra::detail::BackendPolicy<BACKEND>    system_t;

	typedef typename system_t::template container<T> storage_t;
	typedef typename storage_t::iterator iterator;
	typedef typename storage_t::const_iterator const_iterator;
	typedef typename storage_t::reference reference;
	typedef typename storage_t::pointer pointer;
	typedef typename storage_t::value_type value_type;

public:

	//tag
	typedef   void hydra_dense_histogram_tag;

	DenseHistogram()=delete;


	explicit DenseHistogram( std::array<size_t, N> grid,
			std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins *=grid[i];
		}

		fContents.resize(fNBins +2 );
	}

	explicit DenseHistogram( size_t (&grid)[N],
			T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins*=grid[i];
		}

		fContents.resize(fNBins  +2);
	}

	template<typename Int, typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	DenseHistogram( std::array<Int, N> grid, std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits):
					fNBins(1)
		{
			for( size_t i=0; i<N; i++){
				fGrid[i]=grid[i];
				fLowerLimits[i]=lowerlimits[i];
				fUpperLimits[i]=upperlimits[i];
				fNBins *=grid[i];
			}

			fContents.resize(fNBins +2 );
		}

	template<typename Int, typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	DenseHistogram( Int (&grid)[N],	T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
		{
			for( size_t i=0; i<N; i++){
				fGrid[i]=grid[i];
				fLowerLimits[i]=lowerlimits[i];
				fUpperLimits[i]=upperlimits[i];
				fNBins*=grid[i];
			}

			fContents.resize(fNBins  +2);
		}




	DenseHistogram(DenseHistogram< T, N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> const& other ):
			fContents(other.GetContents())
		{
			for( size_t i=0; i<N; i++){
				fGrid[i] = other.GetGrid(i);
				fLowerLimits[i] = other.GetLowerLimits(i);
				fUpperLimits[i] = other.GetUpperLimits(i);
			}

			fNBins= other.GetNBins();
		}

	DenseHistogram(DenseHistogram< T, N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&& other ):
			fContents(std::move(other.GetContents()))
		{
			for( size_t i=0; i<N; i++){
				fGrid[i] = other.GetGrid(i);
				fLowerLimits[i] = other.GetLowerLimits(i);
				fUpperLimits[i] = other.GetUpperLimits(i);
			}

			fNBins= other.GetNBins();
		}

	DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(DenseHistogram<T, N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}


	DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(DenseHistogram<T, N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&& other )
	{
		if(this==&other) return *this;

		fContents = std::move(other.GetContents());
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}



	template<hydra::detail::Backend BACKEND2>
	DenseHistogram(DenseHistogram< T, N, hydra::detail::BackendPolicy<BACKEND2>, detail::multidimensional> const& other ):
			fContents(other.GetContents())
		{
			for( size_t i=0; i<N; i++){
				fGrid[i] = other.GetGrid(i);
				fLowerLimits[i] = other.GetLowerLimits(i);
				fUpperLimits[i] = other.GetUpperLimits(i);
			}

			fNBins= other.GetNBins();
		}

	template<hydra::detail::Backend BACKEND2>
	DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(DenseHistogram<T, N, hydra::detail::BackendPolicy<BACKEND2>, detail::multidimensional> const& other )
	{
		fContents = other.GetContents();
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}


	 inline const storage_t& GetContents() const {
		return fContents;
	}

	 inline void SetContents(storage_t histogram) {
		fContents = histogram;
	}

	 inline size_t GetGrid(size_t i) const {
		return fGrid[i];
	}

	 inline 	T GetLowerLimits(size_t i) const {
		return fLowerLimits[i];
	}

	 inline T GetUpperLimits(size_t i) const {
		return fUpperLimits[i];
	}

	 inline size_t GetNBins() const {
		return fNBins;
	}

	 inline 	size_t GetBin( size_t  (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	 inline size_t GetBin( std::array<size_t,N> const&  bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	 template<typename Int,
	 		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline 	size_t GetBin( Int  (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	 template<typename Int,
	 		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline size_t GetBin( std::array<Int,N> const&  bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	 inline void GetIndexes(size_t globalbin,  size_t  (&bins)[N]){

		get_indexes(globalbin, bins);
	}

	 inline void GetIndexes(size_t globalbin, std::array<size_t,N>&  bins){

		get_indexes(globalbin, bins);
	}

	 template<typename Int,
	 			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline void GetIndexes(size_t globalbin,  Int  (&bins)[N]){

		get_indexes(globalbin, bins);
	}

	 template<typename Int,
	 			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline void GetIndexes(size_t globalbin, std::array<Int,N>&  bins){

		get_indexes(globalbin, bins);
	}



	 inline double GetBinContent( size_t (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return  ( bin < (fNBins) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

	 inline double GetBinContent( std::array<size_t, N> const& bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return  ( bin < (fNBins) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

	 template<typename Int,
	 			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline double GetBinContent( Int (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return  ( bin < (fNBins) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

	 template<typename Int,
	 			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	 inline double GetBinContent( std::array<Int, N> const& bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return  ( bin < (fNBins) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

	 inline double GetBinContent( size_t  bin){

		return ( bin<= (fNBins+1) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

    inline Range<const_iterator> GetBinsContents() const {

    	return make_range(begin(), end());
    }

    inline Range<iterator> GetBinsContents() {

    	return make_range(begin(), end());
    }

    inline Range<HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,N>,
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t>  > >
    GetBinsCenters() {

    	HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,N>,
    			HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> > first(
    					HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t>(0),
    					detail::GetBinCenter<T,N>( fGrid, fLowerLimits, fUpperLimits) );



    	return make_range( first , first+fNBins);
    }

	//stl range interface

    inline pointer data(){
		return fContents.data();
	}

    inline iterator begin(){
		return fContents.begin();
	}

    inline 	iterator end(){
		return fContents.end();
	}

    inline const_iterator begin() const {
		return fContents.begin();
	}

    inline const_iterator end() const {
		return fContents.end();
	}

    inline reference operator[](size_t i) {
		return *(fContents.begin()+i);
	}

    inline  value_type operator[](size_t i) const {
		return fContents.begin()[i];
	}

    inline size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(fContents.begin(), fContents.end() );
	}

	template<typename Iterator>
	 inline DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	 Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	 inline DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	 Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
	 DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&>::type
	Fill(Iterable& container){
		return this->Fill( container.begin(), container.end());
	}

	template<typename Iterable1, typename Iterable2>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value
	&&  hydra::detail::is_iterable<Iterable2>::value,
	 DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>& >::type
	Fill(Iterable1& container, Iterable2& wbegin){
		return this->Fill( container.begin(), container.end(), wbegin.begin());
	}

	template<hydra::detail::Backend BACKEND2, typename Iterator >
	inline  DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator begin, Iterator end);

	template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
	inline  DenseHistogram<T,N, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator1 begin, Iterator1 end, Iterator2 wbegin);



private:

	//k = i_1*(dim_2*...*dim_n) + i_2*(dim_3*...*dim_n) + ... + i_{n-1}*dim_n + i_n

	template<typename Int,size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I== N) && std::is_integral<Int>::value, void>::type
	get_global_bin(const Int (&)[N], size_t&){ }

	template<typename Int,size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N) && std::is_integral<Int>::value, void>::type
	get_global_bin(const Int (&indexes)[N], size_t& index)
	{
		size_t prod =1;
		for(size_t i=N-1; i>I; i--)
			prod *=fGrid[i];
		index += prod*indexes[I];

		get_global_bin<Int,I+1>( indexes, index);
	}

	template<typename Int,size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I== N) && std::is_integral<Int>::value, void>::type
	get_global_bin( std::array<Int,N> const& , size_t&){ }

	template<typename Int,size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N) && std::is_integral<Int>::value, void>::type
	get_global_bin( std::array<Int,N> const& indexes, size_t& index)
	{
		size_t prod =1;

		for(size_t i=N-1; i>I; i--)
			prod *=fGrid[i];

		index += prod*indexes[I];

		get_global_bin<Int, I+1>( indexes, index);
	}

	/*
	 *  conversion of one-dimensional index to multidimensional one
	 * ____________________________________________________________
	 */

	//----------------------------------------
	// multiply  std::array elements
	//----------------------------------------
	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply( std::array<size_t, N> const&, size_t& )
	{ }

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply( std::array<size_t, N> const&  obj, size_t& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply<I+1>( obj, result );
	}

	//----------------------------------------
	// multiply static array elements
	//----------------------------------------
	template< size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply( size_t (&)[N] , size_t& )
	{ }

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply( size_t (&obj)[N], size_t& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply<I+1>( obj, result );
	}


	//-------------------------
	// std::array version
	//-------------------------
	//end of recursion
	template<typename Int, size_t I,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t ,  std::array<Int,N>& )
	{}

	//begin of the recursion
	template<typename Int, size_t I=0,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index, std::array<Int,N>& indexes)
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< Int,I+1>(next_index,indexes );
	}

	//-------------------------
	// static array version
	//-------------------------
	//end of recursion
	template<typename Int, size_t I,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t, Int (&)[N])
	{}

	//begin of the recursion
	template<typename Int, size_t I=0,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index, Int (&indexes)[N] )
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< Int, I+1>(next_index, indexes );
	}



	T fUpperLimits[N];
	T fLowerLimits[N];
	size_t   fGrid[N];
	size_t   fNBins;
	storage_t fContents;
	system_t fSystem;

};


/*
 * 1D dimension specialization
 */
/**
 * \ingroup histogram
 * \brief Class representing one-dimensional dense histogram.
 */
template< typename T, hydra::detail::Backend BACKEND >
class DenseHistogram<T, 1,  hydra::detail::BackendPolicy<BACKEND>,   detail::unidimensional >{

	typedef hydra::detail::BackendPolicy<BACKEND>    system_t;

	typedef typename system_t::template container<T> storage_t;

	typedef typename storage_t::iterator iterator;
	typedef typename storage_t::const_iterator const_iterator;
	typedef typename storage_t::reference reference;
	typedef typename storage_t::pointer pointer;
	typedef typename storage_t::value_type value_type;

public:

	DenseHistogram()=delete;


	DenseHistogram( size_t grid, T lowerlimits, T upperlimits):
		fGrid(grid),
		fLowerLimits(lowerlimits),
		fUpperLimits(upperlimits),
		fNBins(grid),
		fContents( grid+2 )
	{}


	DenseHistogram(DenseHistogram< T,1,  hydra::detail::BackendPolicy<BACKEND>,detail::unidimensional > const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}

	DenseHistogram(DenseHistogram< T,1,  hydra::detail::BackendPolicy<BACKEND>,detail::unidimensional >&& other ):
			fContents(std::move(other.GetContents())),
			fGrid(other.GetGrid()),
			fLowerLimits(other.GetLowerLimits()),
			fUpperLimits(other.GetUpperLimits()),
			fNBins(other.GetNBins())
		{}



	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(DenseHistogram<T, 1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		fGrid = other.GetGrid();
		fLowerLimits = other.GetLowerLimits();
		fUpperLimits = other.GetUpperLimits();
		fNBins= other.GetNBins();

		return *this;
	}

	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(DenseHistogram<T, 1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&& other )
	{
		if(this==&other) return *this;

		fContents = std::move(other.GetContents());
		fGrid = other.GetGrid();
		fLowerLimits = other.GetLowerLimits();
		fUpperLimits = other.GetUpperLimits();
		fNBins= other.GetNBins();

		return *this;
	}

	template<hydra::detail::Backend BACKEND2>
	DenseHistogram(DenseHistogram< T,1,  hydra::detail::BackendPolicy<BACKEND2>,detail::unidimensional > const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}

	template<hydra::detail::Backend BACKEND2>
	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&
	operator=(DenseHistogram<T, 1, hydra::detail::BackendPolicy<BACKEND2>, detail::unidimensional> const& other )
	{
		fContents = other.GetContents();
		fGrid = other.GetGrid();
		fLowerLimits = other.GetLowerLimits();
		fUpperLimits = other.GetUpperLimits();
		fNBins= other.GetNBins();

		return *this;
	}

	const storage_t& GetContents()const  {
		return fContents;
	}

	void SetContents(storage_t histogram) {
		fContents = histogram;
	}

	size_t GetGrid() const {
		return fGrid;
	}

	T GetLowerLimits() const {
		return fLowerLimits;
	}

	T GetUpperLimits() const {
		return fUpperLimits;
	}

	size_t GetNBins() const {
		return fNBins;
	}

	double GetBinContent(size_t i){

		return (i<=fNBins+1) ?
				fContents.begin()[i] :
					std::numeric_limits<double>::max();
	}

	inline Range<HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,1>,
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t>  > >
	GetBinsCenters() const {


		HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,1>,
		HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> > first(HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t>(0),
				detail::GetBinCenter<T,1>( fGrid, fLowerLimits, fUpperLimits) );



		return make_range( first , first+fNBins);
	}

	inline Range<const_iterator> GetBinsContents() const {

	    	return make_range(begin(), end());
	}

	inline Range<iterator> GetBinsContents()  {

		    	return make_range(begin(), end());
		}


	//stl interface
	pointer data(){
		return fContents.data();
	}

	iterator begin(){
		return fContents.begin();
	}

	iterator end(){
		return fContents.end();
	}

	const_iterator begin() const {
		return fContents.begin();
	}

	const_iterator end() const {
		return fContents.end();
	}

   reference operator[](size_t i) {
    	return *(fContents.begin()+i);
    }

   value_type  operator[](size_t i) const {
        return fContents.begin()[i];
    }

	size_t size() const	{
	 return  HYDRA_EXTERNAL_NS::thrust::distance(fContents.begin(), fContents.end() );
	}

	template<typename Iterator>
	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&
	 Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&
	Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>& >::type
	Fill(Iterable&& container){
		return this->Fill( std::forward<Iterable>(container).begin(), std::forward<Iterable>(container).end());
	}

	template<typename Iterable1, typename Iterable2>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value
	&&  hydra::detail::is_iterable<Iterable2>::value,
	DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>& >::type
	Fill(Iterable1&& container, Iterable2&& wbegin){
		return this->Fill( container.begin(), container.end(), wbegin.begin());
	}


	template<hydra::detail::Backend BACKEND2, typename Iterator>
	inline DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,Iterator begin, Iterator end);

	template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
	inline DenseHistogram<T,1, hydra::detail::BackendPolicy<BACKEND>, detail::unidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,Iterator1 begin, Iterator1 end, Iterator2 wbegin);



private:

	T fUpperLimits;
	T fLowerLimits;
	size_t   fGrid;
	size_t   fNBins;
	storage_t fContents;
	system_t fSystem;

};

/**
 * \ingroup histogram
 * \brief Function to make a N-dimensional dense histogram.
 *
 * @param backend
 * @param grid  std::array storing the bins per dimension.
 * @param lowerlimits std::array storing the lower limits per dimension.
 * @param upperlimits  std::array storing the upper limits per dimension.
 * @param first Iterator pointing to the begin of the data range.
 * @param end Iterator pointing to the end of the data range.
 * @return
 */
template<typename Iterator, typename T, size_t N , hydra::detail::Backend BACKEND>
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits,
		Iterator first, Iterator end);

/**
 * \ingroup histogram
 * \brief Function to make a N-dimensional dense histogram.
 *
 * @param backend
 * @param grid  std::array storing the bins per dimension.
 * @param lowerlimits std::array storing the lower limits per dimension.
 * @param upperlimits  std::array storing the upper limits per dimension.
 * @param data Iterable storing the data to histogram.
 * @return
 */
template<typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<T, N> lowerlimits,   std::array<T, N> upperlimits,	Iterable&& data);




/**
 * \ingroup histogram
 * \brief Function to make a N-dimensional dense histogram.
 *
 * @param backend
 * @param grid  std::array storing the bins per dimension.
 * @param lowerlimits std::array storing the lower limits per dimension.
 * @param upperlimits  std::array storing the upper limits per dimension.
 * @param first Iterator pointing to the begin of the data range.
 * @param end Iterator pointing to the end of the data range.
 * @return
 */
template<typename Iterator, typename T, hydra::detail::Backend BACKEND>
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid, T lowerlimits, T upperlimits,
		Iterator first, Iterator end);


}  // namespace hydra

#include <hydra/detail/DenseHistogram.inl>


#endif /* DENSEHISTOGRAM_H_ */
