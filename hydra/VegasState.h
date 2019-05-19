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
 * VegasState.h
 *
 *  Created on: 19/07/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef VEGASSTATE_H_
#define VEGASSTATE_H_


#include <iostream>

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Containers.h>
#include <hydra/Types.h>

#include <vector>
#include <hydra/detail/external/thrust/copy.h>
#include <chrono>


namespace hydra {

enum {

	MODE_IMPORTANCE = 1,
	MODE_IMPORTANCE_ONLY = 0,
	MODE_STRATIFIED = -1,
	BINS_MAX = 50
};

/**
 * \ingroup numerical_integration
 * \brief Class to hold resources and state of hydra::Vegas integration
 * algorithm.
 * \tparam N number of parameters.
 * \tparam BACKEND can be hydra::omp::sys , hydra::cuda::sys , hydra::tbb::sys , hydra::cpp::sys ,hydra::host::sys and hydra::device::sys.
 */
template<size_t N , typename  BACKEND>
class VegasState ;

/**
 * \ingroup numerical_integration
 * \brief Class to hold resources and state of hydra::Vegas integration
 * algorithm.
 * \tparam N number of parameters.
 * \tparam hydra::detail::BackendPolicy<BACKEND> can be hydra::omp::sys ,
 *  hydra::cuda::sys , hydra::tbb::sys , hydra::cpp::sys ,hydra::host::sys and hydra::device::sys.
 */
template<size_t N , hydra::detail::Backend BACKEND>
class VegasState<N, hydra::detail::BackendPolicy<BACKEND>>
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<GReal_t>  rvector_backend;
	typedef typename system_t::template container<GUInt_t>  uvector_backend;
	typedef typename std::vector<GReal_t>        rvector_std;

	typedef typename rvector_backend::iterator rvector_iterator;
	typedef typename uvector_backend::iterator uvector_iterator;
	typedef typename rvector_std::iterator  rvector_std_iterator;

public:


	VegasState() = delete;

	/**
	 * @brief Constructor
	 * @param xlower std::array<GReal_t,N> with the lower limits of the integration region.
	 * @param xupper std::array<GReal_t,N>  with the upper limits of the integration region.
	 */
	VegasState(std::array<GReal_t,N> const& xlower,
			std::array<GReal_t,N> const& xupper);
	/**
	 * @brief Constructor
	 * @param xlower flat array with the lower limits of the integration region.
	 * @param xupper flat array with the upper limits of the integration region.
	 */
	VegasState(const GReal_t xlower[N], const GReal_t xupper[N]);

	/**
	 * @brief Copy constructor for a state in the same backend.
	 * @param state
	 */
	VegasState(const VegasState<N,hydra::detail::BackendPolicy<BACKEND>> &state);

	/**
	 * @brief Copy constructor for a state in a different backend.
	 * @param state
	 * @tparam BACKEND2 different backend.
	 */
	template<hydra::detail::Backend BACKEND2>
	VegasState(const VegasState<N,hydra::detail::BackendPolicy<BACKEND2>> &state);


	VegasState<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=(const VegasState<N,hydra::detail::BackendPolicy<BACKEND>> &state);

	/**
	 * @brief Copy constructor for a state in a different backend.
	 * @param state
	 * @tparam BACKEND2 different backend.
	 */
	template<hydra::detail::Backend BACKEND2>
	VegasState<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=(const VegasState<N,hydra::detail::BackendPolicy<BACKEND2>> &state);



    /**
     * @brief  Clear results of previously stored iterations.
     */
	void ClearStoredIterations();


	inline GReal_t GetAlpha() const { return fAlpha; }

	inline void SetAlpha(GReal_t alpha)	{ fAlpha = alpha;	}
	//-----------------------------
	//Calls

	inline size_t GetCalls() const {return fCalls;}

	inline size_t GetCalls(GBool_t training) const
	{ return  training ? fTrainingCalls : fCalls;}

	inline void SetCalls(size_t calls) {fCalls = calls;}

	inline void SetCalls(GBool_t training, size_t calls)
	{ if(training) fTrainingCalls = calls;
	else fCalls = calls;}

	//-----------------------------
	//CallsPerBox

	inline size_t GetCallsPerBox() const {return fCallsPerBox;}

	inline void SetCallsPerBox(size_t callsPerBox) {fCallsPerBox = callsPerBox;}

	//-----------------------------
	//ChiSquare

	inline GReal_t GetChiSquare() const {return fChiSquare;}

	inline void SetChiSquare(GReal_t chiSquare) { fChiSquare = chiSquare;}

	//-----------------------------
	//ChiSum

	inline GReal_t GetChiSum() const {return fChiSum;}

	inline void SetChiSum(GReal_t chiSum) { fChiSum = chiSum;}

	//-----------------------------
	//CumulatedResult

	inline const std::vector<GReal_t>& GetCumulatedResult() const {return fCumulatedResult;}

	inline void SetCumulatedResult(const std::vector<GReal_t>& cumulatedResult) {fCumulatedResult = cumulatedResult; }

	//-----------------------------
	//CumulatedSigma

	inline const std::vector<GReal_t>& GetCumulatedSigma() const {return fCumulatedSigma;}

	inline void SetCumulatedSigma(const std::vector<GReal_t>& cumulatedSigma){ fCumulatedSigma = cumulatedSigma;}

	//----------------
	//DeltaX

	inline const std::vector<GReal_t>& GetDeltaX() const {return fDeltaX;	}

	inline void SetDeltaX(const std::vector<GReal_t>& deltaX) {fDeltaX = deltaX;}

	inline void SetDeltaX(GUInt_t i, GReal_t dx) {fDeltaX[i] = dx;}

	//----------------
	//Distribution

	inline const std::vector<GReal_t>& GetDistribution() const {return fDistribution;}

	inline  std::vector<GReal_t>& GetDistribution() {return fDistribution;}

	inline void SetDistribution(const std::vector<GReal_t>& distribution) {fDistribution = distribution; }

	inline void SetDistribution(GUInt_t i, GReal_t x) {fDistribution[i] = x;}

	inline void SetDistribution(GUInt_t bin, GUInt_t dim, GReal_t x) {fDistribution[bin*N+dim] = x;}

	//----------------
	//IterationDuration

	inline const std::vector<GReal_t>& GetIterationDuration() const {return fIterationDuration;}

	inline void SetIterationDuration(const std::vector<GReal_t>& iterationDuration) {fIterationDuration = iterationDuration;}

	//----------------
	//IterationResult

	inline const std::vector<GReal_t>& GetIterationResult() const {return fIterationResult;}

	inline void SetIterationResult(const std::vector<GReal_t>& iterationResult){fIterationResult = iterationResult;}

	//----------------
	//Iterations

	inline GUInt_t GetIterations() const {return fIterations;	}

	inline void SetIterations(GUInt_t iterations) {fIterations = iterations;}

	//----------------
	//IterationSigma

	inline const std::vector<GReal_t>& GetIterationSigma() const {return fIterationSigma;}

	inline void SetIterationSigma(const std::vector<GReal_t>& iterationSigma) {fIterationSigma = iterationSigma;}

	//----------------
	//ItNum

	inline GUInt_t GetItNum() const {return fItNum;}

	inline void SetItNum(GUInt_t itNum) {fItNum = itNum;}

	//----------------
	//ItStart

	inline GUInt_t GetItStart() const {return fItStart;}

	inline void SetItStart(GUInt_t itStart) {fItStart = itStart;	}

	//----------------
	//Jacobian

	inline GReal_t GetJacobian() const {return fJacobian;	}


	inline void SetJacobian(GReal_t jacobian) {fJacobian = jacobian;}

	//----------------
	//MaxError

	inline GReal_t GetMaxError() const {return fMaxError;	}

	inline void SetMaxError(GReal_t maxError) {fMaxError = maxError;	}

	//----------------
	//Mode

	inline GInt_t GetMode() const {return fMode;	}

	inline void SetMode(GInt_t mode) {fMode = mode;}

	//----------------
	//NBins

	inline size_t GetNBins() const {return fNBins;}

	inline void SetNBins(size_t nBins) { fNBins = nBins;	}

	//----------------
	//NBinsMax

	inline size_t GetNBinsMax() const {	return fNBinsMax;}


	inline void SetNBinsMax(size_t nBinsMax) {fNBinsMax = nBinsMax;}

	//----------------
	//NBoxes

	inline size_t GetNBoxes() const {return fNBoxes;}

	inline void SetNBoxes(size_t nBoxes) {fNBoxes = nBoxes;}

	//----------------
	//NDimensions

	inline size_t GetNDimensions() const {return fNDimensions;}


	inline void SetNDimensions(size_t nDimensions) {
		fNDimensions = nDimensions;
	}

	//----------------
	//OStream

	inline std::ostream& GetOStream()  {return fOStream;}

	//----------------
	//Result

	inline GReal_t GetResult() const {return fResult;}

	inline void SetResult(GReal_t result) {fResult = result;}

	//----------------
	//Samples


	inline GUInt_t GetSamples() const {return fSamples;}

	inline void SetSamples(GUInt_t samples) {fSamples = samples;}

	//----------------
	//Sigma

	inline GReal_t GetSigma() const {return fSigma;}

	inline void SetSigma(GReal_t sigma) {fSigma = sigma;}

	//----------------
	//Stage

	inline GInt_t GetStage() const {return fStage;	}

	inline void SetStage(GInt_t stage) {fStage = stage;}

	//----------------
	//SumOfWeights

	inline GReal_t GetSumOfWeights() const {return fSumOfWeights;}

	inline void SetSumOfWeights(GReal_t sumOfWeights) {fSumOfWeights = sumOfWeights;}

	//----------------
	//UseRelativeError

	inline GBool_t IsUseRelativeError() const {return fUseRelativeError;}

	inline void SetUseRelativeError(GBool_t useRelativeError) {fUseRelativeError = useRelativeError;	}

	//----------------
	//Verbose

	inline GInt_t GetVerbose() const {return fVerbose;}

	inline void SetVerbose(GInt_t verbose) {fVerbose = verbose;}

	//----------------
	//Volume

	inline GReal_t GetVolume() const {return fVolume;}

	inline void SetVolume(GReal_t volume) {fVolume = volume;}

	//----------------
	//Weight


	inline const std::vector<GReal_t>& GetWeight() const {return fWeight;}

	inline void SetWeight(const std::vector<GReal_t>& weight) {fWeight = weight;}

	inline void SetWeight(GUInt_t i, GReal_t weight) {fWeight[i] = weight;}

	//----------------
	//WeightedIntSum

	inline GReal_t GetWeightedIntSum() const { return fWeightedIntSum;}

	inline void SetWeightedIntSum(GReal_t weightedIntSum) {fWeightedIntSum = weightedIntSum;}

	//----------------
	//Xi

	inline const std::vector<GReal_t>& GetXi() const {return fXi;}

	inline void SetXi(const std::vector<GReal_t>& xi) {fXi = xi;}

	inline void SetXi(GInt_t i, GReal_t xi) {fXi[i] = xi;}

	//----------------
	//	Xin

	inline const std::vector<GReal_t>& GetXin() const {return fXin;}

	inline void SetXin(const std::vector<GReal_t>& xin) {fXin = xin;}

	inline void SetXin(GUInt_t i, GReal_t xin) {fXin[i] = xin;}

	//----------------
	//Store...

	inline void StoreIterationResult(const GReal_t integral, const GReal_t sigma)
	{
		fIterationResult.push_back(integral);
		fIterationSigma.push_back(sigma);
	}


	inline void StoreCumulatedResult(const GReal_t integral, const GReal_t sigma)
	{
		fCumulatedResult.push_back(integral);
		fCumulatedSigma.push_back(sigma);
		fResult=integral;
		fSigma=sigma;
	}


	inline void StoreIterationDuration(const GReal_t timing) { fIterationDuration.push_back(timing);}
	inline void StoreFunctionCallsDuration(const GReal_t timing) { fFunctionCallsDuration.push_back(timing);}

	inline const std::vector<GReal_t>& GetXLow() const { return fXLow; }

	inline void SetXLow(const std::vector<GReal_t>& xLow) {fXLow = xLow;}

	inline const std::vector<GReal_t>& GetXUp() const {return fXUp;}

	inline void SetXUp(const std::vector<GReal_t>& xUp) {fXUp = xUp;}


	inline void CopyStateToDevice()
	{
		HYDRA_EXTERNAL_NS::thrust::copy(fXi.begin(), fXi.end(), fBackendXi.begin());
		//HYDRA_EXTERNAL_NS::thrust::copy( fDistribution.begin(), fDistribution.end(), fBackendDistribution.begin());

	}

	inline void CopyStateToHost()
	{/*	HYDRA_EXTERNAL_NS::thrust::copy(
			fBackendDistribution.begin(),
			fBackendDistribution.end(),
			fDistribution.begin());*/
	}


	inline void SendGridToBackend()
	{
		HYDRA_EXTERNAL_NS::thrust::copy(fDeltaX.begin(),
				fDeltaX.end(),
				fBackendDeltaX.begin());

		//checar
		HYDRA_EXTERNAL_NS::thrust::copy(fXLow.begin(),
				fXLow.end(),
				fBackendXLow.begin());
	}

//
	rvector_backend& GetBackendDeltaX() {return fBackendDeltaX;	}

	const rvector_backend& GetBackendDeltaX() const {return fBackendDeltaX;	}

	void SetBackendDeltaX(const rvector_backend& deviceDeltaX) {fBackendDeltaX = deviceDeltaX;}

//
	rvector_backend& GetBackendXi() {return fBackendXi;}

	const rvector_backend& GetBackendXi() const {return fBackendXi;}

	void SetBackendXi(const rvector_backend& deviceXi) {fBackendXi = deviceXi;}

//
	rvector_backend& GetBackendXLow() { return fBackendXLow;}

	const rvector_backend& GetBackendXLow() const { return fBackendXLow;}

	void SetBackendXLow(const rvector_backend& deviceXLow) {fBackendXLow = deviceXLow;}



	size_t GetTrainingCalls() const {
		return fTrainingCalls;
	}

	void SetTrainingCalls(size_t trainingCalls) {
		fTrainingCalls = trainingCalls;
	}

	GUInt_t GetTrainingIterations() const {
		return fTrainingIterations;
	}

	void SetTrainingIterations(GUInt_t trainingIterations) {
		fTrainingIterations = trainingIterations;
	}

	std::vector<GReal_t> const& GetFunctionCallsDuration() const {
		return fFunctionCallsDuration;
	}

	void SetFunctionCallsDuration(std::vector<GReal_t> functionCallsDuration) {
		fFunctionCallsDuration = functionCallsDuration;
	}

	GBool_t IsTrainedGridFrozen() const {
		return fTrainedGridFrozen;
	}

	void SetTrainedGridFrozen(GBool_t trainedGridFrozen) {
		fTrainedGridFrozen = trainedGridFrozen;
	}

	//const rvector_backend& GetBackendDistribution() const {	return fBackendDistribution;}


private:

	GInt_t fVerbose;
	std::ostream &fOStream;

	/* grid */
	size_t fNDimensions;
	size_t fNBinsMax;
	size_t fNBins;
	size_t fNBoxes; /* these are both counted along the axes */

	//host


	//backend
	//rvector_backend fBackendDistribution;
	rvector_backend fBackendXLow;//initgrid
	rvector_backend fBackendXi;//CopyStateToDevice
	rvector_backend fBackendDeltaX;//initgrid


	//std
	std::vector<GReal_t> fXUp;
	std::vector<GReal_t> fXLow;
	std::vector<GReal_t> fXi;
	std::vector<GReal_t> fXin;
	std::vector<GReal_t> fDeltaX;
	std::vector<GReal_t> fWeight;
	std::vector<GReal_t> fDistribution;
	std::vector<GReal_t> fIterationResult; ///< vector with the result per iteration
	std::vector<GReal_t> fIterationSigma; ///< vector with the result per iteration
	std::vector<GReal_t> fCumulatedResult; ///< vector of cumulated results per iteration
	std::vector<GReal_t> fCumulatedSigma; ///< vector of cumulated sigmas per iteration
	std::vector<GReal_t> fIterationDuration; ///< vector with the time per iteration
	std::vector<GReal_t> fFunctionCallsDuration; ///< vector with the time per iteration

	//mc_host_vector<GUInt_t> fBox;



	GReal_t fVolume;
	/* control variables */
	GReal_t fAlpha;
	GInt_t fMode;
	GUInt_t fIterations;
	GInt_t fStage;
	GBool_t fTrainedGridFrozen;

	/* scratch variables preserved between calls to vegas1/2/3  */
	GReal_t fJacobian;
	GReal_t fWeightedIntSum;
	GReal_t fSumOfWeights;
	GReal_t fChiSum;
	GReal_t fChiSquare;
	GReal_t fResult;
	GReal_t fSigma;

	GUInt_t fTrainingIterations;
	GUInt_t fItStart;
	GUInt_t fItNum;
	GUInt_t fSamples;
	size_t  fCallsPerBox; ///< number of call per box
	size_t  fCalls;
	size_t  fTrainingCalls;
	GReal_t fMaxError; ///< max error
	GBool_t fUseRelativeError; ///< use relative error as convergence criteria

};


}
#endif /* VEGASSTATE_H_ */

#include <hydra/detail/VegasState.inl>
