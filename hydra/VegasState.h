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
 * VegasState.h
 *
 *  Created on: 19/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */

#ifndef VEGASSTATE_H_
#define VEGASSTATE_H_


#include <iostream>

#include <hydra/detail/Config.h>
#include <hydra/Containers.h>
#include <hydra/Types.h>


#include <thrust/copy.h>
#include <chrono>
#include <mutex>

namespace hydra {


enum {

	MODE_IMPORTANCE = 1,
	MODE_IMPORTANCE_ONLY = 0,
	MODE_STRATIFIED = -1,
	BINS_MAX = 50
};

template<size_t N >
class VegasState {

public:

	VegasState(std::array<GReal_t,N> const& xlower,
			std::array<GReal_t,N> const& xupper);

	VegasState(const VegasState &state);

	void ResetState();

	inline GReal_t GetAlpha() const {
		return fAlpha;
	}

	inline void SetAlpha(GReal_t alpha) {
		fAlpha = alpha;
	}
	//-----------------------------
	//Calls

	inline size_t GetCalls() const {
		return fCalls;
	}

	inline void SetCalls(size_t calls) {
		fCalls = calls;
	}

	//-----------------------------
	//CallsPerBox

	inline size_t GetCallsPerBox() const {
		return fCallsPerBox;
	}

	inline void SetCallsPerBox(size_t callsPerBox) {
		fCallsPerBox = callsPerBox;
	}

	//-----------------------------
	//ChiSquare

	inline GReal_t GetChiSquare() const {
		return fChiSquare;
	}

	inline void SetChiSquare(GReal_t chiSquare) {
		fChiSquare = chiSquare;
	}

	//-----------------------------
	//ChiSum

	inline GReal_t GetChiSum() const {
		return fChiSum;
	}


	inline void SetChiSum(GReal_t chiSum) {
		fChiSum = chiSum;
	}

	//-----------------------------
	//CumulatedResult

	inline const mc_host_vector<GReal_t>& GetCumulatedResult() const {
		return fCumulatedResult;
	}

	inline void SetCumulatedResult(const mc_host_vector<GReal_t>& cumulatedResult) {
		fCumulatedResult = cumulatedResult;
	}

	//-----------------------------
	//CumulatedSigma

	inline const mc_host_vector<GReal_t>& GetCumulatedSigma() const {
		return fCumulatedSigma;
	}

	inline void SetCumulatedSigma(const mc_host_vector<GReal_t>& cumulatedSigma) {
		fCumulatedSigma = cumulatedSigma;
	}

	//----------------
	//DeltaX

	inline const mc_host_vector<GReal_t>& GetDeltaX() const {
		return fDeltaX;
	}

	inline void SetDeltaX(const mc_host_vector<GReal_t>& deltaX) {
		fDeltaX = deltaX;
	}

	inline void SetDeltaX(GUInt_t i, GReal_t dx) {
			fDeltaX[i] = dx;
	}

	//----------------
	//Distribution

	inline const mc_host_vector<GReal_t>& GetDistribution() const {
		return fDistribution;
	}

	inline void SetDistribution(const mc_host_vector<GReal_t>& distribution) {
		fDistribution = distribution;
	}

	inline void SetDistribution(GUInt_t i,  GReal_t x) {
			fDistribution[i] = x;
	}

	inline void SetDistribution(GUInt_t bin, GUInt_t dim,  GReal_t x) {
			fDistribution[bin*N+dim] = x;
	}

	//----------------
	//IterationDuration


	inline const mc_host_vector<GReal_t>& GetIterationDuration() const {
		return fIterationDuration;
	}


	inline void SetIterationDuration(
			const mc_host_vector<GReal_t>& iterationDuration) {
		fIterationDuration = iterationDuration;
	}

	//----------------
	//IterationResult

	inline const mc_host_vector<GReal_t>& GetIterationResult() const {
		return fIterationResult;
	}

	inline void SetIterationResult(const mc_host_vector<GReal_t>& iterationResult) {
		fIterationResult = iterationResult;
	}

	//----------------
	//Iterations

	inline GUInt_t GetIterations() const {
		return fIterations;
	}

	inline void SetIterations(GUInt_t iterations) {
		fIterations = iterations;
	}

	//----------------
	//IterationSigma

	inline const mc_host_vector<GReal_t>& GetIterationSigma() const {
		return fIterationSigma;
	}


	inline void SetIterationSigma(const mc_host_vector<GReal_t>& iterationSigma) {
		fIterationSigma = iterationSigma;
	}

	//----------------
	//ItNum


	inline GUInt_t GetItNum() const {
		return fItNum;
	}

	inline void SetItNum(GUInt_t itNum) {
		fItNum = itNum;
	}

	//----------------
	//ItStart

	inline GUInt_t GetItStart() const {
		return fItStart;
	}


	inline void SetItStart(GUInt_t itStart) {
		fItStart = itStart;
	}

	//----------------
	//Jacobian


	inline GReal_t GetJacobian() const {
		return fJacobian;
	}


	inline void SetJacobian(GReal_t jacobian) {
		fJacobian = jacobian;
	}

	//----------------
	//MaxError


	inline GReal_t GetMaxError() const {
		return fMaxError;
	}


	inline void SetMaxError(GReal_t maxError) {
		fMaxError = maxError;
	}

	//----------------
	//Mode


	inline GInt_t GetMode() const {
		return fMode;
	}


	inline void SetMode(GInt_t mode) {
		fMode = mode;
	}

	//----------------
	//NBins

	inline size_t GetNBins() const {
		return fNBins;
	}


	inline void SetNBins(size_t nBins) {
		fNBins = nBins;
	}

	//----------------
	//NBinsMax

	inline size_t GetNBinsMax() const {
		return fNBinsMax;
	}


	inline void SetNBinsMax(size_t nBinsMax) {
		fNBinsMax = nBinsMax;
	}

	//----------------
	//NBoxes


	inline size_t GetNBoxes() const {
		return fNBoxes;
	}


	inline void SetNBoxes(size_t nBoxes) {
		fNBoxes = nBoxes;
	}

	//----------------
	//NDimensions


	inline size_t GetNDimensions() const {
		return fNDimensions;
	}


	inline void SetNDimensions(size_t nDimensions) {
		fNDimensions = nDimensions;
	}

	//----------------
	//OStream

	inline std::ostream& GetOStream()  {
		return fOStream;
	}

	//----------------
	//Result


	inline GReal_t GetResult() const {
		return fResult;
	}


	inline void SetResult(GReal_t result) {
		fResult = result;
	}

	//----------------
	//Samples


	inline GUInt_t GetSamples() const {
		return fSamples;
	}


	inline void SetSamples(GUInt_t samples) {
		fSamples = samples;
	}

	//----------------
	//Sigma


	inline GReal_t GetSigma() const {
		return fSigma;
	}


	inline void SetSigma(GReal_t sigma) {
		fSigma = sigma;
	}

	//----------------
	//Stage


	inline GInt_t GetStage() const {
		return fStage;
	}


	inline void SetStage(GInt_t stage) {
		fStage = stage;
	}

	//----------------
	//SumOfWeights


	inline GReal_t GetSumOfWeights() const {
		return fSumOfWeights;
	}


	inline void SetSumOfWeights(GReal_t sumOfWeights) {
		fSumOfWeights = sumOfWeights;
	}

	//----------------
	//UseRelativeError


	inline GBool_t IsUseRelativeError() const {
		return fUseRelativeError;
	}


	inline void SetUseRelativeError(GBool_t useRelativeError) {
		fUseRelativeError = useRelativeError;
	}

	//----------------
	//Verbose


	inline GInt_t GetVerbose() const {
		return fVerbose;
	}


	inline void SetVerbose(GInt_t verbose) {
		fVerbose = verbose;
	}

	//----------------
	//Volume


	inline GReal_t GetVolume() const {
		return fVolume;
	}


	inline void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	//----------------
	//Weight


	inline const mc_host_vector<GReal_t>& GetWeight() const {
		return fWeight;
	}


	inline void SetWeight(const mc_host_vector<GReal_t>& weight) {
		fWeight = weight;
	}


	inline void SetWeight(GUInt_t i, GReal_t weight) {
		fWeight[i] = weight;
	}

	//----------------
	//WeightedIntSum


	inline GReal_t GetWeightedIntSum() const {
		return fWeightedIntSum;
	}


	inline void SetWeightedIntSum(GReal_t weightedIntSum) {
		fWeightedIntSum = weightedIntSum;
	}

	//----------------
	//Xi


	inline const mc_host_vector<GReal_t>& GetXi() const {
		return fXi;
	}


	inline void SetXi(const mc_host_vector<GReal_t>& xi) {
		fXi = xi;
	}


	inline void SetXi(GInt_t i, GReal_t xi) {
		fXi[i] = xi;
	}

	//----------------
	//	Xin


	inline const mc_host_vector<GReal_t>& GetXin() const {
		return fXin;
	}


	inline void SetXin(const mc_host_vector<GReal_t>& xin) {
		fXin = xin;
	}


	inline void SetXin(GUInt_t i, GReal_t xin) {
		fXin[i] = xin;
	}

	//----------------
	//Store...


	inline void StoreIterationResult(const GReal_t integral,
			const GReal_t sigma) {
		fIterationResult.push_back(integral);
		fIterationSigma.push_back(sigma);
	}


	inline void StoreCumulatedResult(const GReal_t integral,
			const GReal_t sigma) {
		fCumulatedResult.push_back(integral);
		fCumulatedSigma.push_back(sigma);
	}


	inline void StoreIterationDuration(const GReal_t timing) {
		fIterationDuration.push_back(timing);
	}


	inline const mc_host_vector<GReal_t>& GetXLow() const {
		return fXLow;
	}


	inline void SetXLow(const mc_host_vector<GReal_t>& xLow) {
		fXLow = xLow;
	}


	inline const mc_host_vector<GReal_t>& GetXUp() const {
		return fXUp;
	}


	inline void SetXUp(const mc_host_vector<GReal_t>& xUp) {
		fXUp = xUp;
	}



	inline void CopyStateToDevice()
	{
		thrust::copy(fXi.begin(), fXi.end(), fDeviceXi.begin());
		thrust::copy( fDistribution.begin(),
						  						  fDistribution.end(), fDeviceDistribution.begin());

	}

	inline void CopyStateToHost()
	{

		thrust::copy( fDeviceDistribution.begin(),
				  						  fDeviceDistribution.end(), fDistribution.begin());
	}


	inline void SendGridToDevice()
		{

					  thrust::copy(fDeltaX.begin(), fDeltaX.end(),
							  fDeviceDeltaX.begin());

					  //checar
					  thrust::copy(fXLow.begin(),
							  fXLow.end(), fDeviceXLow.begin());



		}


	const mc_device_vector<GReal_t>& GetDeviceDeltaX() const {
		return fDeviceDeltaX;
	}

	void SetDeviceDeltaX(const mc_device_vector<GReal_t>& deviceDeltaX) {
		fDeviceDeltaX = deviceDeltaX;
	}


	const mc_device_vector<GReal_t>& GetDeviceXi() const {
		return fDeviceXi;
	}

	void SetDeviceXi(const mc_device_vector<GReal_t>& deviceXi) {
		fDeviceXi = deviceXi;
	}

	const mc_device_vector<GReal_t>& GetDeviceXLow() const {
		return fDeviceXLow;
	}

	void SetDeviceXLow(const mc_device_vector<GReal_t>& deviceXLow) {
		fDeviceXLow = deviceXLow;
	}

	const mc_device_vector<GReal_t>& GetDeviceDistribution() const {
		return fDeviceDistribution;
	}

	std::mutex* GetMutex()  {
		return &fMutex;
	}

	GInt_t fVerbose;
std::ostream &fOStream;
private:
	/* grid */
	size_t fNDimensions;
	size_t fNBinsMax;
	size_t fNBins;
	size_t fNBoxes; /* these are both counted along the axes */

	//host
	mc_host_vector<GReal_t> fXUp;
	mc_host_vector<GReal_t> fXLow;
	mc_host_vector<GReal_t> fXi;
	mc_host_vector<GReal_t> fXin;
	mc_host_vector<GReal_t> fDeltaX;
	mc_host_vector<GReal_t> fWeight;
	//mc_host_vector<GReal_t> fX;
	//mc_host_vector<GFloat_t> fDistribution;
	mc_host_vector<GReal_t> fDistribution;
	mc_host_vector<GReal_t> fIterationResult; ///< vector with the result per iteration
	mc_host_vector<GReal_t> fIterationSigma; ///< vector with the result per iteration
	mc_host_vector<GReal_t> fCumulatedResult; ///< vector of cumulated results per iteration
	mc_host_vector<GReal_t> fCumulatedSigma; ///< vector of cumulated sigmas per iteration
	mc_host_vector<GReal_t> fIterationDuration; ///< vector with the time per iteration
	//mc_host_vector<GUInt_t> fBox;

	//device

	mc_device_vector<GReal_t> fDeviceDistribution;
	mc_device_vector<GReal_t> fDeviceXLow;//initgrid
	mc_device_vector<GReal_t> fDeviceXi;//CopyStateToDevice
	mc_device_vector<GReal_t> fDeviceDeltaX;//initgrid

	GReal_t fVolume;
	/* control variables */
	GReal_t fAlpha;
	GInt_t fMode;
	GUInt_t fIterations;
	GInt_t fStage;

	/* scratch variables preserved between calls to vegas1/2/3  */
	GReal_t fJacobian;
	GReal_t fWeightedIntSum;
	GReal_t fSumOfWeights;
	GReal_t fChiSum;
	GReal_t fChiSquare;
	GReal_t fResult;
	GReal_t fSigma;

	GUInt_t fItStart;
	GUInt_t fItNum;
	GUInt_t fSamples;
	size_t  fCallsPerBox; ///< number of call per box
	size_t  fCalls;
	GReal_t fMaxError; ///< max error
	GBool_t fUseRelativeError; ///< use relative error as convergence criteria

	std::mutex fMutex;


};


}
#endif /* VEGASSTATE_H_ */

#include <hydra/detail/VegasState.inl>
