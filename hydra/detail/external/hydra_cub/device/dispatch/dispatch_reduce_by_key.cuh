/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceReduceByKey provides device-wide, parallel operations for
 *       reducing segments of values residing within device-accessible memory.
 */

#pragma once

#include <hydra/detail/external/hydra_cub/agent/agent_reduce_by_key.cuh>
#include <hydra/detail/external/hydra_cub/config.cuh>
#include <hydra/detail/external/hydra_cub/device/dispatch/dispatch_scan.cuh>
#include <hydra/detail/external/hydra_cub/grid/grid_queue.cuh>
#include <hydra/detail/external/hydra_cub/thread/thread_operators.cuh>
#include <hydra/detail/external/hydra_cub/util_deprecated.cuh>
#include <hydra/detail/external/hydra_cub/util_device.cuh>
#include <hydra/detail/external/hydra_cub/util_math.cuh>

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

#include <hydra/detail/external/hydra_libcudacxx/nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Multi-block reduce-by-key sweep kernel entry point
 *
 * @tparam AgentReduceByKeyPolicyT
 *   Parameterized AgentReduceByKeyPolicyT tuning policy type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param d_unique_out
 *   Pointer to the output sequence of unique keys (one key per run)
 *
 * @param d_values_in
 *   Pointer to the input sequence of corresponding values
 *
 * @param d_aggregates_out
 *   Pointer to the output sequence of value aggregates (one aggregate per run)
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs encountered
 *   (i.e., the length of d_unique_out)
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   KeyT equality operator
 *
 * @param reduction_op
 *   ValueT reduction operator
 *
 * @param num_items
 *   Total number of items to select from
 */
template <typename ChainedPolicyT,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReduceByKeyPolicyT::BLOCK_THREADS)) __global__
  void DeviceReduceByKeyKernel(KeysInputIteratorT d_keys_in,
                               UniqueOutputIteratorT d_unique_out,
                               ValuesInputIteratorT d_values_in,
                               AggregatesOutputIteratorT d_aggregates_out,
                               NumRunsOutputIteratorT d_num_runs_out,
                               ScanTileStateT tile_state,
                               int start_tile,
                               EqualityOpT equality_op,
                               ReductionOpT reduction_op,
                               OffsetT num_items)
{
  using AgentReduceByKeyPolicyT = typename ChainedPolicyT::ActivePolicy::ReduceByKeyPolicyT;

  // Thread block type for reducing tiles of value segments
  using AgentReduceByKeyT = AgentReduceByKey<AgentReduceByKeyPolicyT,
                                             KeysInputIteratorT,
                                             UniqueOutputIteratorT,
                                             ValuesInputIteratorT,
                                             AggregatesOutputIteratorT,
                                             NumRunsOutputIteratorT,
                                             EqualityOpT,
                                             ReductionOpT,
                                             OffsetT,
                                             AccumT>;

  // Shared memory for AgentReduceByKey
  __shared__ typename AgentReduceByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentReduceByKeyT(temp_storage,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    d_aggregates_out,
                    d_num_runs_out,
                    equality_op,
                    reduction_op)
    .ConsumeRange(num_items, tile_state, start_tile);
}

namespace detail 
{

template <class AccumT, class KeyOutputT>
struct device_reduce_by_key_policy_hub
{
  static constexpr int MAX_INPUT_BYTES = CUB_MAX(sizeof(KeyOutputT), sizeof(AccumT));
  static constexpr int COMBINED_INPUT_BYTES = sizeof(KeyOutputT) + sizeof(AccumT);

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 6;
    static constexpr int ITEMS_PER_THREAD =
      (MAX_INPUT_BYTES <= 8)
        ? 6
        : CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
                  CUB_MAX(1,
                          ((NOMINAL_4B_ITEMS_PER_THREAD * 8) + COMBINED_INPUT_BYTES - 1) /
                            COMBINED_INPUT_BYTES));

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             ITEMS_PER_THREAD,
                             BLOCK_LOAD_DIRECT,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             detail::default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  using MaxPolicy = Policy350;
};

}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceReduceByKey
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam SelectedPolicy 
 *   Implementation detail, do not specify directly, requirements on the 
 *   content of this type are subject to breaking change.
 */
template <typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT = detail::accumulator_t<ReductionOpT,
                                                  cub::detail::value_t<ValuesInputIteratorT>,
                                                  cub::detail::value_t<ValuesInputIteratorT>>,
          typename SelectedPolicy =                //
          detail::device_reduce_by_key_policy_hub< //
            AccumT,                                //
            cub::detail::non_void_value_t<         //
              UniqueOutputIteratorT,               //
              cub::detail::value_t<KeysInputIteratorT>>>>
struct DispatchReduceByKey
{
  //-------------------------------------------------------------------------
  // Types and constants
  //-------------------------------------------------------------------------

  // The input values type
  using ValueInputT = cub::detail::value_t<ValuesInputIteratorT>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

  void *d_temp_storage;
  size_t &temp_storage_bytes;
  KeysInputIteratorT d_keys_in;
  UniqueOutputIteratorT d_unique_out;
  ValuesInputIteratorT d_values_in;
  AggregatesOutputIteratorT d_aggregates_out;
  NumRunsOutputIteratorT d_num_runs_out;
  EqualityOpT equality_op;
  ReductionOpT reduction_op;
  OffsetT num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchReduceByKey(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      KeysInputIteratorT d_keys_in,
                      UniqueOutputIteratorT d_unique_out,
                      ValuesInputIteratorT d_values_in,
                      AggregatesOutputIteratorT d_aggregates_out,
                      NumRunsOutputIteratorT d_num_runs_out,
                      EqualityOpT equality_op,
                      ReductionOpT reduction_op,
                      OffsetT num_items,
                      cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_unique_out(d_unique_out)
      , d_values_in(d_values_in)
      , d_aggregates_out(d_aggregates_out)
      , d_num_runs_out(d_num_runs_out)
      , equality_op(equality_op)
      , reduction_op(reduction_op)
      , num_items(num_items)
      , stream(stream)
  {}

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  template <typename ActivePolicyT, typename ScanInitKernelT, typename ReduceByKeyKernelT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke(ScanInitKernelT init_kernel,
                                                          ReduceByKeyKernelT reduce_by_key_kernel)
  {
    using AgentReduceByKeyPolicyT = typename ActivePolicyT::ReduceByKeyPolicyT;
    const int block_threads = AgentReduceByKeyPolicyT::BLOCK_THREADS;
    const int items_per_thread = AgentReduceByKeyPolicyT::ITEMS_PER_THREAD;

    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      if (CubDebug(error = cudaGetDevice(&device_ordinal)))
      {
        break;
      }

      // Number of input tiles
      int tile_size = block_threads * items_per_thread;
      int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0])))
      {
        break; // bytes needed for tile status descriptors
      }

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void *allocations[1] = {};
      if (CubDebug(
            error =
              AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_state;
      if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0])))
      {
        break;
      }

      // Log init_kernel configuration
      int init_grid_size = CUB_MAX(1, cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              (long long)stream);
#endif

      // Invoke init_kernel to initialize tile descriptors
      HYDRA_THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size,
                                                              INIT_KERNEL_THREADS,
                                                              0,
                                                              stream)
        .doit(init_kernel, tile_state, num_tiles, d_num_runs_out);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Get SM occupancy for reduce_by_key_kernel
      int reduce_by_key_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(reduce_by_key_sm_occupancy,
                                          reduce_by_key_kernel,
                                          block_threads)))
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      if (CubDebug(
            error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal)))
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
      {
// Log reduce_by_key_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking %d reduce_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                block_threads,
                (long long)stream,
                items_per_thread,
                reduce_by_key_sm_occupancy);
#endif

        // Invoke reduce_by_key_kernel
        HYDRA_THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(scan_grid_size,
                                                                block_threads,
                                                                0,
                                                                stream)
          .doit(reduce_by_key_kernel,
                d_keys_in,
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                tile_state,
                start_tile,
                equality_op,
                reduction_op,
                num_items);

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    return Invoke<ActivePolicyT>(DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
                                 DeviceReduceByKeyKernel<MaxPolicyT,
                                                         KeysInputIteratorT,
                                                         UniqueOutputIteratorT,
                                                         ValuesInputIteratorT,
                                                         AggregatesOutputIteratorT,
                                                         NumRunsOutputIteratorT,
                                                         ScanTileStateT,
                                                         EqualityOpT,
                                                         ReductionOpT,
                                                         OffsetT,
                                                         AccumT>);
  }

  /**
   * Internal dispatch routine
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of corresponding values
   *
   * @param[out] d_aggregates_out
   *   Pointer to the output sequence of value aggregates
   *   (one aggregate per run)
   *
   * @param[out] d_num_runs_out
   *   Pointer to total number of runs encountered
   *   (i.e., the length of d_unique_out)
   *
   * @param[in] equality_op
   *   KeyT equality operator
   *
   * @param[in] reduction_op
   *   ValueT reduction operator
   *
   * @param[in] num_items
   *   Total number of items to select from
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      DispatchReduceByKey dispatch(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_unique_out,
                                   d_values_in,
                                   d_aggregates_out,
                                   d_num_runs_out,
                                   equality_op,
                                   reduction_op,
                                   num_items,
                                   stream);

      // Dispatch
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    d_aggregates_out,
                    d_num_runs_out,
                    equality_op,
                    reduction_op,
                    num_items,
                    stream);
  }
};

CUB_NAMESPACE_END
