#ifndef HLS_TORCHTRAITS_H
#define HLS_TORCHTRAITS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace torch {
namespace Torch {
namespace OpTrait {

template <typename ConcreteType>
class IsOutOfPlaceVariant
    : public ::mlir::OpTrait::TraitBase<ConcreteType, IsOutOfPlaceVariant> {};
} // namespace OpTrait
} // namespace Torch
} // namespace torch
} // namespace mlir

#endif
