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
 * SparseHistogram.h
 *
 *  Created on: 29/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPARSEHISTOGRAM_H_
#define SPARSEHISTOGRAM_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Dimensionality.h>
#include <hydra/detail/functors/GetBinCenter.h>
#include <hydra/Range.h>
#include <hydra/Algorithm.h>

#include <type_traits>
#include <utility>
#include <array>


#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/find.h>

namespace hydra {

/**
 * \ingroup histogram
 */
template<typename T, size_t N,  typename BACKEND, typename = typename detail::dimensionality<N>::type,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
class SparseHistogram;

/**
 * \ingroup histogram
 * Class representing multidimensional sparse histogram.
 */
template<typename T, size_t N, hydra::detail::Backend BACKEND >
class SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
{

	typedef hydra::detail::BackendPolicy<BACKEND>    system_t;

	typedef typename system_t::template container<double> storage_data_t;
	typedef typename system_t::template container<size_t> storage_keys_t;

	typedef typename storage_data_t::iterator data_iterator;
	typedef typename storage_data_t::const_iterator data_const_iterator;
	typedef typename storage_data_t::reference data_reference;
	typedef typename storage_data_t::pointer data_pointer;
	typedef typename storage_data_t::value_type data_value_type;

	typedef typename storage_keys_t::iterator keys_iterator;
	typedef typename storage_keys_t::const_iterator keys_const_iterator;
	typedef typename storage_keys_t::reference keys_reference;
	typedef typename storage_keys_t::pointer keys_pointer;
	typedef typename storage_keys_t::value_type keys_value_type;

	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_iterator, data_iterator>> iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_const_iterator, data_const_iterator> > const_iterator;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	typedef std::pair<size_t*, double*> pointer_pair;

public:

	//tag
	typedef   void hydra_sparse_histogram_tag;

	SparseHistogram()=delete;

	explicit SparseHistogram( std::array<size_t , N> const& grid,
			std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins *=grid[i];
		}

	}

	explicit SparseHistogram( size_t (&grid)[N],
			T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins*=grid[i];
		}

	}

	template<typename Int, typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	SparseHistogram( std::array<Int , N> const& grid,
			std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins *=grid[i];
		}

	}

	template<typename Int, typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	SparseHistogram( Int (&grid)[N],
			T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins*=grid[i];
		}

	}

	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		fBins = other.GetBins();

		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}

	SparseHistogram(SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional> const& other ):
			fContents(other.GetContents()),
			fBins(other.GetBins())
		{
			for( size_t i=0; i<N; i++){
				fGrid[i] = other.GetGrid(i);
				fLowerLimits[i] = other.GetLowerLimits(i);
				fUpperLimits[i] = other.GetUpperLimits(i);
			}

			fNBins= other.GetNBins();
		}

	template<hydra::detail::Backend BACKEND2>
	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	operator=(SparseHistogram<T, N,  detail::BackendPolicy<BACKEND2>, detail::multidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		fBins = other.GetBins();

		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}

	template<hydra::detail::Backend BACKEND2>
	SparseHistogram(SparseHistogram<T, N,  detail::BackendPolicy<BACKEND2>, detail::multidimensional> const& other ):
		fContents(other.GetContents()),
		fBins(other.GetBins())
	{
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
	}


	inline const storage_data_t& GetContents() const {
		return fContents;
	}

	inline const storage_keys_t& GetBins() const
	{
		return fBins;
	}

	inline void SetBins(storage_keys_t bins)
	{
		fBins = bins;
	}

	inline void SetContents(storage_data_t histogram) {
		fContents = histogram;
	}

	inline size_t GetGrid(size_t i) const {
		return fGrid[i];
	}

	T GetLowerLimits(size_t i) const {
		return fLowerLimits[i];
	}

	inline T GetUpperLimits(size_t i) const {
		return fUpperLimits[i];
	}

	inline size_t GetNBins() const {
		return fNBins;
	}

	template<typename Int,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline 	size_t GetBin( Int  (&bins)[N]){

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
	inline size_t GetBin( std::array<Int,N> const&  bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
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

	template<typename Int,
				typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline double GetBinContent( Int  bins[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		size_t index = std::distance(fBins.begin(),
								std::find(fBins.begin(),fBins.end(), bin));

				return  (index < fBins.size() ) ?
								fContents.begin()[index] : 0.0;
	}

	inline double GetBinContent(std::array<size_t, N> const& bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		size_t index = HYDRA_EXTERNAL_NS::thrust::distance(fBins.begin(),
				HYDRA_EXTERNAL_NS::thrust::find(fSystem , fBins.begin(),fBins.end(), bin));

		return (index < fBins.size() ) ? fContents.begin()[index] : 0.0;
	}


	template<typename Int,
					typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline double GetBinContent(std::array<Int, N> const& bins){

			size_t bin=0;

			get_global_bin( bins,  bin);

			size_t index = HYDRA_EXTERNAL_NS::thrust::distance(fBins.begin(),
					HYDRA_EXTERNAL_NS::thrust::find(fSystem , fBins.begin(),fBins.end(), bin));

			return (index < fBins.size() ) ? fContents.begin()[index] : 0.0;
		}



	inline double GetBinContent( size_t  bin){

		size_t index = std::distance(fBins.begin(),
				std::find(fBins.begin(),fBins.end(), bin));

		return  (index < fBins.size() ) ?
				fContents.begin()[index] : 0.0;
	}

	inline Range<data_iterator> GetBinsContents() const {

		return make_range( fContents.begin(), fContents.begin() + fNBins);
	}

	inline Range< HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,N>, keys_iterator> >
	GetBinsCenters() {

		HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,N>, keys_iterator> first( fBins.begin(),
				detail::GetBinCenter<T,N>( fGrid, fLowerLimits, fUpperLimits) );

		return make_range( first , first + fNBins);
	}


	//stl interface

	pointer_pair data(){
		return std::make_pair(fBins.data() , fContents.data());
	}

	iterator begin(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
						HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.begin() ,  fContents.begin()) );
	}

	iterator end(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	const_iterator begin() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.cbegin() ,  fContents.cbegin() ) );
	}

	const_iterator end() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	reference operator[](size_t i) {
		return *(begin()+i);
	}

    value_type operator[](size_t i) const {
		return begin()[i];
	}

	size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(begin(), end() );
	}

	template<typename Iterator>
	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);


	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>& >::type
	Fill(Iterable&& container){
		return this->Fill( std::forward<Iterable>(container).begin(),
				std::forward<Iterable>(container).end());
	}

	template<typename Iterable1, typename Iterable2>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value
	&&  hydra::detail::is_iterable<Iterable2>::value,
	SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>& >::type
	Fill(Iterable1&& container, Iterable2&& wbegin){
		return this->Fill( std::forward<Iterable1>(container).begin(),
				std::forward<Iterable1>(container).end(), std::forward<Iterable2>(wbegin).begin());
	}


	template<hydra::detail::Backend BACKEND2,typename Iterator>
	inline SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator begin, Iterator end);

	template<hydra::detail::Backend BACKEND2,typename Iterator1, typename Iterator2>
	inline SparseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator1 begin, Iterator1 end, Iterator2 wbegin);



private:

	//k = i_1*(dim_2*...*dim_n) + i_2*(dim_3*...*dim_n) + ... + i_{n-1}*dim_n + i_n

	template<typename Int,size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I== N) && std::is_integral<Int>::value, void>::type
	get_global_bin(const Int (&)[N], size_t& ){ }

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
	get_global_bin( std::array<Int,N> const& , size_t& ){ }

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
	multiply( std::array<size_t, N> const& , size_t&  )
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
	multiply( size_t (&)[N] , size_t&  )
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
	get_indexes(size_t,  std::array<Int,N>& )
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
	get_indexes(size_t , Int (&)[N])
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
	storage_data_t fContents;
	storage_keys_t fBins;
	system_t fSystem;

};

/**
 * \ingroup histogram
 * Class representing one-dimensional sparse histogram.
 */
template< typename T, hydra::detail::Backend BACKEND >
class SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,  detail::unidimensional >{

	typedef hydra::detail::BackendPolicy<BACKEND>    system_t;

	typedef std::vector<double> storage_data_t;
	typedef std::vector<size_t> storage_keys_t;

	typedef typename storage_data_t::iterator data_iterator;
	typedef typename storage_data_t::const_iterator data_const_iterator;
	typedef typename storage_data_t::reference data_reference;
	typedef typename storage_data_t::pointer data_pointer;
	typedef typename storage_data_t::value_type data_value_type;

	typedef typename storage_keys_t::iterator keys_iterator;
	typedef typename storage_keys_t::const_iterator keys_const_iterator;
	typedef typename storage_keys_t::reference keys_reference;
	typedef typename storage_keys_t::pointer keys_pointer;
	typedef typename storage_keys_t::value_type keys_value_type;


	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_iterator, data_iterator>> iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_const_iterator, data_const_iterator> > const_iterator;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	typedef std::pair<size_t*, double*> pointer_pair;

public:

	SparseHistogram()=delete;


	SparseHistogram( size_t grid, T lowerlimits, T upperlimits):
		fGrid(grid),
		fLowerLimits(lowerlimits),
		fUpperLimits(upperlimits),
		fNBins(grid)
	{}


	SparseHistogram(SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional > const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}

	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	operator=(SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional > const& other )
	{
		if(this==&other) return *this;
		fContents = other.GetContents();
		fGrid = other.GetGrid();
		fLowerLimits = other.GetLowerLimits();
		fUpperLimits = other.GetUpperLimits();
		fNBins = other.GetNBins();
		return *this;
	}

	template<hydra::detail::Backend BACKEND2>
	SparseHistogram(SparseHistogram<T,1, detail::BackendPolicy<BACKEND2>,detail::unidimensional > const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}

	template<hydra::detail::Backend BACKEND2>
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	operator=(SparseHistogram<T,1, detail::BackendPolicy<BACKEND2>,detail::unidimensional > const& other )
	{
		if(this==&other) return *this;
		fContents = other.GetContents();
		fGrid = other.GetGrid();
		fLowerLimits = other.GetLowerLimits();
		fUpperLimits = other.GetUpperLimits();
		fNBins = other.GetNBins();
		return *this;
	}

	const storage_data_t& GetContents()const  {
		return fContents;
	}

	void SetContents(storage_data_t histogram) {
		fContents = histogram;
	}

	const storage_keys_t& GetBins() const
	{
		return fBins;
	}

	void SetBins(storage_keys_t bins)
	{
		fBins = bins;
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


	double GetBinContent( size_t  bin) {

		size_t index = std::distance(fBins.begin(),
				std::find(fBins.begin(),fBins.end(), bin));

		return  ( bin< fBins.size() ) ?
				fContents.begin()[bin] : 0.0;
	}

	inline Range<HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,1>, keys_iterator> >
	GetBinsCenters() {

		HYDRA_EXTERNAL_NS::thrust::transform_iterator<detail::GetBinCenter<T,1>, keys_iterator >
		first( fBins.begin(), detail::GetBinCenter<T,1>( fGrid, fLowerLimits, fUpperLimits) );

		return make_range( first , first+fNBins);
	}

	inline Range<iterator> GetBinsContents()  {

	  	return make_range(begin(),begin()+fNBins );
	}

	//stl interface

	pointer_pair data(){
		return std::make_pair(fBins.data() , fContents.data());
	}

	iterator begin(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
						HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.begin() ,  fContents.begin()) );
	}

	iterator end(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	const_iterator begin() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.cbegin() ,  fContents.cbegin() ) );
	}

	const_iterator end() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	reference operator[](size_t i) {
		return *(begin()+i);
	}

    value_type operator[](size_t i) const {
		return begin()[i];
	}

	size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(begin(), end() );
	}

	template<typename Iterator>
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);


	template<typename Iterable>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >& >::type
	Fill(Iterable&& container){
		return this->Fill( std::forward<Iterable>(container).begin(),
				std::forward<Iterable>(container).end());
	}

	template<typename Iterable1, typename Iterable2>
	inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value
	&&  hydra::detail::is_iterable<Iterable2>::value,
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >& >::type
	Fill(Iterable1&& container, Iterable2&& wbegin){
		return this->Fill( std::forward<Iterable1>(container).begin(),
				std::forward<Iterable1>(container).end(), std::forward<Iterable2>(wbegin).begin());
	}


	template<hydra::detail::Backend BACKEND2,typename Iterator>
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator begin, Iterator end);

	template<hydra::detail::Backend BACKEND2,typename Iterator1, typename Iterator2>
	SparseHistogram<T,1, detail::BackendPolicy<BACKEND>,detail::unidimensional >&
	Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator1 begin, Iterator1 end, Iterator2 wbegin);



private:



	T fUpperLimits;
	T fLowerLimits;
	size_t   fGrid;
	size_t   fNBins;
	storage_data_t fContents;
	storage_keys_t fBins;
	system_t fSystem;
};

/**
 * \ingroup histogram
 * \brief Function to make a N-dimensional sparse histogram.
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
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits,
		Iterator first, Iterator end);

/**
 * \ingroup histogram
 * \brief Function to make a N-dimensional sparse histogram.
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
SparseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid, T lowerlimits, T upperlimits,
		Iterator first, Iterator end);


}  // namespace hydra

#include <hydra/detail/SparseHistogram.inl>

#endif /* SPARSEHISTOGRAM_H_ */
