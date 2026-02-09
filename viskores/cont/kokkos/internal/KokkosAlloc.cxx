//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#include <viskores/cont/kokkos/internal/KokkosAlloc.h>

#include <viskores/cont/ErrorBadAllocation.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Logging.h>
#include <viskores/cont/kokkos/internal/KokkosTypes.h>

#include <sstream>

namespace viskores
{
namespace cont
{
namespace kokkos
{
namespace internal
{

void* Allocate(std::size_t size)
{
  if (!Kokkos::is_initialized())
  {
    VISKORES_LOG_F(viskores::cont::LogLevel::Info,
                   "Allocating device memory before Kokkos has been initialized. Calling "
                   "viskores::cont::Initialize.");
    viskores::cont::Initialize();
  }
  try
  {
#if defined(KOKKOS_HAS_SHARED_SPACE)
    return Kokkos::kokkos_malloc<Kokkos::SharedSpace>(size);
#else
    return Kokkos::kokkos_malloc<ExecutionSpace::memory_space>(size);
#endif
  }
  catch (...) // the type of error thrown is not well documented
  {
    std::ostringstream err;
    err << "Failed to allocate " << size << " bytes on Kokkos device";
    throw viskores::cont::ErrorBadAllocation(err.str());
  }
}

void Free(void* ptr)
{
  if (Kokkos::is_initialized())
  {
    GetExecutionSpaceInstance().fence();
#if defined(KOKKOS_HAS_SHARED_SPACE)
    Kokkos::kokkos_free<Kokkos::SharedSpace>(ptr);
#else
    Kokkos::kokkos_free<ExecutionSpace::memory_space>(ptr);
#endif
  }
  else
  {
    // It is possible that a Buffer instance might try to free its Kokkos data after
    // Kokkos has been finalized. If that is the case, silently do nothing.
  }
}

bool IsUnifiedMemoryPointer(const void* ptr)
{
#if defined(KOKKOS_HAS_SHARED_SPACE)
  // There seems to be no direct way to query whether a pointer is a unified memory
  // pointer in Kokkos. Since we use SharedSpace for (de)allocation and Kokkos views,
  // if unified memory is available, i.e., if KOKKOS_HAS_SHARED_SPACE is true, we
  // can assume non-null pointer are unified memory pointers.
  return ptr != nullptr;
#else
  return false;
#endif
}

void* Reallocate(void* ptr, std::size_t newSize)
{
  try
  {
    return Kokkos::kokkos_realloc<ExecutionSpace::memory_space>(ptr, newSize);
  }
  catch (...)
  {
    std::ostringstream err;
    err << "Failed to re-allocate " << newSize << " bytes on Kokkos device";
    throw viskores::cont::ErrorBadAllocation(err.str());
  }
}

}
}
}
} // viskores::cont::kokkos::internal
