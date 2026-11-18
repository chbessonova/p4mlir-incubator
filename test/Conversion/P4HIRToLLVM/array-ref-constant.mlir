// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm | FileCheck %s

// A valid constant beyond the signed 29-bit GEP immediate range uses an SSA
// index, but still needs no guard on its access.
// CHECK-LABEL: @large_constant(
// CHECK-NOT: llvm.icmp
// CHECK: llvm.getelementptr inbounds %{{.*}}[0, %{{.*}}]
// CHECK-NOT: llvm.cond_br
// CHECK: llvm.load
!b8i = !p4hir.bit<8>
!b64i = !p4hir.bit<64>
!arr = !p4hir.array<268435457 x !b8i>
module {
  p4hir.func @large_constant(%a: !p4hir.ref<!arr>) -> !b8i {
    %i = p4hir.const #p4hir.int<268435456> : !b64i
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b64i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}
