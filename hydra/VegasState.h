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

	__host__
	inline GReal_t GetAlpha() const {
		return fAlpha;
	}

	__host__
	inline void SetAlpha(GReal_t alpha) {
		fAlpha = alpha;
	}

	//-----------------------------
	//CallsPerBox

	__host__
	inline size_t GetCallsPerBox() const {
		return fCallsPerBox;
	}

	__host__
	inline void SetCallsPerBox(size_t callsPerBox) {
		fCallsPerBox = callsPerBox;
	}

	//-----------------------------
	//ChiSquare

	__host__
	inline GReal_t GetChiSquare() const {
		return fChiSquare;
	}

	__host__
	inline void SetChiSquare(GReal_t chiSquare) {
		fChiSquare = chiSquare;
	}

	//-----------------------------
	//ChiSum

	__host__
	inline GReal_t GetChiSum() const {
		return fChiSum;
	}

	__host__
	inline void SetChiSum(GReal_t chiSum) {
		fChiSum = chiSum;
	}

	//-----------------------------
	//CumulatedResult

	__host__
	inline const mc_host_vector<GReal_t>& GetCumulatedResult() const {
		return fCumulatedResult;
	}

	__host__
	inline void SetCumulatedResult(const mc_host_vector<GReal_t>& cumulatedResult) {
		fCumulatedResult = cumulatedResult;
	}

	//-----------------------------
	//CumulatedSigma

	__host__
	inline const mc_host_vector<GReal_t>& GetCumulatedSigma() const {
		return fCumulatedSigma;
	}

	__host__
	inline void SetCumulatedSigma(const mc_host_vector<GReal_t>& cumulatedSigma) {
		fCumulatedSigma = cumulatedSigma;
	}

	//----------------
	//DeltaX

	__host__
	inline const mc_host_vector<GReal_t>& GetDeltaX() const {
		return fDeltaX;
	}

	__host__
	inline void SetDeltaX(const mc_host_vector<GReal_t>& deltaX) {
		fDeltaX = deltaX;
	}

	__host__
	inline void SetDeltaX(GUInt_t i, GReal_t dx) {
			fDeltaX[i] = dx;
	}

	//----------------
	//Distribution

	__host__
	inline const mc_host_vector<GReal_t>& GetDistribution() const {
		return fDistribution;
	}

	__host__
	inline void SetDistribution(const mc_host_vector<GReal_t>& distribution) {
		fDistribution = distribution;
	}

	__host__
	inline void SetDistribution(GUInt_t i,  GReal_t x) {
			fDistribution[i] = x;
	}

	__host__
	inline void SetDistribution(GUInt_t bin, GUInt_t dim,  GReal_t x) {
			fDistribution[bin*N+dim] = x;
	}

	//----------------
	//IterationDuration

	__host__
	inline const mc_host_vector<GReal_t>& GetIterationDuration() const {
		return fIterationDuration;
	}

	__host__
	inline void SetIterationDuration(
			const mc_host_vector<GReal_t>& iterationDuration) {
		fIterationDuration = iterationDuration;
	}

	//----------------
	//IterationResult

	__host__
	inline const mc_host_vector<GReal_t>& GetIterationResult() const {
		return fIterationResult;
	}

	__host__
	inline void SetIterationResult(const mc_host_vector<GReal_t>& iterationResult) {
		fIterationResult = iterationResult;
	}

	//----------------
	//Iterations

	__host__
	inline GUInt_t GetIterations() const {
		return fIterations;
	}

	__host__
	inline void SetIterations(GUInt_t iterations) {
		fIterations = iterations;
	}

	//----------------
	//IterationSigma

	__host__
	inline const mc_host_vector<GReal_t>& GetIterationSigma() const {
		return fIterationSigma;
	}

	__host__
	inline void SetIterationSigma(const mc_host_vector<GReal_t>& iterationSigma) {
		fIterationSigma = iterationSigma;
	}

	//----------------
	//ItNum

	__host__
	inline GUInt_t GetItNum() const {
		return fItNum;
	}

	__host__
	inline void SetItNum(GUInt_t itNum) {
		fItNum = itNum;
	}

	//----------------
	//ItStart

	__host__
	inline GUInt_t GetItStart() const {
		return fItStart;
	}

	__host__
	inline void SetItStart(GUInt_t itStart) {
		fItStart = itStart;
	}

	//----------------
	//Jacobian

	__host__
	inline GReal_t GetJacobian() const {
		return fJacobian;
	}

	__host__
	inline void SetJacobian(GReal_t jacobian) {
		fJacobian = jacobian;
	}

	//----------------
	//MaxError

	__host__
	inline GReal_t GetMaxError() const {
		return fMaxError;
	}

	__host__
	inline void SetMaxError(GReal_t maxError) {
		fMaxError = maxError;
	}

	//----------------
	//Mode

	__host__
	inline GInt_t GetMode() const {
		return fMode;
	}

	__host__
	inline void SetMode(GInt_t mode) {
		fMode = mode;
	}

	//----------------
	//NBins

	__host__
	inline size_t GetNBins() const {
		return fNBins;
	}

	__host__
	inline void SetNBins(size_t nBins) {
		fNBins = nBins;
	}

	//----------------
	//NBinsMax

	__host__
	inline size_t GetNBinsMax() const {
		return fNBinsMax;
	}

	__host__
	inline void SetNBinsMax(size_t nBinsMax) {
		fNBinsMax = nBinsMax;
	}

	//----------------
	//NBoxes

	__host__
	inline size_t GetNBoxes() const {
		return fNBoxes;
	}

	__host__
	inline void SetNBoxes(size_t nBoxes) {
		fNBoxes = nBoxes;
	}

	//----------------
	//NDimensions

	__host__
	inline size_t GetNDimensions() const {
		return fNDimensions;
	}

	__host__
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

	__host__
	inline GReal_t GetResult() const {
		return fResult;
	}

	__host__
	inline void SetResult(GReal_t result) {
		fResult = result;
	}

	//----------------
	//Samples

	__host__
	inline GUInt_t GetSamples() const {
		return fSamples;
	}

	__host__
	inline void SetSamples(GUInt_t samples) {
		fSamples = samples;
	}

	//----------------
	//Sigma

	__host__
	inline GReal_t GetSigma() const {
		return fSigma;
	}

	__host__
	inline void SetSigma(GReal_t sigma) {
		fSigma = sigma;
	}

	//----------------
	//Stage

	__host__
	inline GInt_t GetStage() const {
		return fStage;
	}

	__host__
	inline void SetStage(GInt_t stage) {
		fStage = stage;
	}

	//----------------
	//SumOfWeights

	__host__
	inline GReal_t GetSumOfWeights() const {
		return fSumOfWeights;
	}

	__host__
	inline void SetSumOfWeights(GReal_t sumOfWeights) {
		fSumOfWeights = sumOfWeights;
	}

	//----------------
	//UseRelativeError

	__host__
	inline GBool_t IsUseRelativeError() const {
		return fUseRelativeError;
	}

	__host__
	inline void SetUseRelativeError(GBool_t useRelativeError) {
		fUseRelativeError = useRelativeError;
	}

	//----------------
	//Verbose

	__host__
	inline GInt_t GetVerbose() const {
		return fVerbose;
	}

	__host__
	inline void SetVerbose(GInt_t verbose) {
		fVerbose = verbose;
	}

	//----------------
	//Volume

	__host__
	inline GReal_t GetVolume() const {
		return fVolume;
	}

	__host__
	inline void SetVolume(GReal_t volume) {
		fVolume = volume;
	}

	//----------------
	//Weight

	__host__
	inline const mc_host_vector<GReal_t>& GetWeight() const {
		return fWeight;
	}

	__host__
	inline void SetWeight(const mc_host_vector<GReal_t>& weight) {
		fWeight = weight;
	}

	__host__
	inline void SetWeight(GUInt_t i, GReal_t weight) {
		fWeight[i] = weight;
	}

	//----------------
	//WeightedIntSum

	__host__
	inline GReal_t GetWeightedIntSum() const {
		return fWeightedIntSum;
	}

	__host__
	inline void SetWeightedIntSum(GReal_t weightedIntSum) {
		fWeightedIntSum = weightedIntSum;
	}

	//----------------
	//Xi

	__host__
	inline const mc_host_vector<GReal_t>& GetXi() const {
		return fXi;
	}

	__host__
	inline void SetXi(const mc_host_vector<GReal_t>& xi) {
		fXi = xi;
	}

	__host__
	inline void SetXi(GInt_t i, GReal_t xi) {
		fXi[i] = xi;
	}

	//----------------
	//	Xin

	__host__
	inline const mc_host_vector<GReal_t>& GetXin() const {
		return fXin;
	}

	__host__
	inline void SetXin(const mc_host_vector<GReal_t>& xin) {
		fXin = xin;
	}

	__host__
	inline void SetXin(GUInt_t i, GReal_t xin) {
		fXin[i] = xin;
	}

	//----------------
	//Store...

	__host__
	inline void StoreIterationResult(const GReal_t integral,
			const GReal_t sigma) {
		fIterationResult.push_back(integral);
		fIterationSigma.push_back(sigma);
	}

	__host__
	inline void StoreCumulatedResult(const GReal_t integral,
			const GReal_t sigma) {
		fCumulatedResult.push_back(integral);
		fCumulatedSigma.push_back(sigma);
	}

	__host__
	inline void StoreIterationDuration(const GReal_t timing) {
		fIterationDuration.push_back(timing);
	}

	__host__
	inline const mc_host_vector<GReal_t>& GetXLow() const {
		return fXLow;
	}

	__host__
	inline void SetXLow(const mc_host_vector<GReal_t>& xLow) {
		fXLow = xLow;
	}

	__host__
	inline const mc_host_vector<GReal_t>& GetXUp() const {
		return fXUp;
	}

	__host__
	inline void SetXUp(const mc_host_vector<GReal_t>& xUp) {
		fXUp = xUp;
	}


	__host__
	inline void CopyStateToDevice()
	{
		thrust::copy(fXi.begin(), fXi.end(), fDeviceXi.begin());
		thrust::copy( fDistribution.begin(),
						  						  fDistribution.end(), fDeviceDistribution.begin());

	}
	__host__
	inline void CopyStateToHost()
	{

		thrust::copy( fDeviceDistribution.begin(),
				  						  fDeviceDistribution.end(), fDistribution.begin());
	}

	__host__
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
	size_t fCallsPerBox; ///< number of call per box

	GReal_t fMaxError; ///< max error
	GBool_t fUseRelativeError; ///< use relative error as convergence criteria

	std::mutex fMutex;


};


}
#endif /* VEGASSTATE_H_ */

#include <hydra/detail/VegasState.inl>
